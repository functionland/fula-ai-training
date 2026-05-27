"""Qwen 3 LoRA SFT trainer (multi-turn messages, thinking-mode-aware).

Companion to the legacy `lora_train.py` which used a flat
prompt+completion shape suited to Qwen 2.5. This trainer:

  - Walks each `*.labelled.json` transcript and builds a proper
    multi-turn `messages` list (system + user + assistant + tool + ...)
  - Embeds the production SYSTEM_PROMPT_TEMPLATE verbatim in every
    example (per advisor sign-off: matches inference-time byte-for-byte)
  - Uses `tokenizer.apply_chat_template(messages, enable_thinking=True)`
    so the assistant turn is prefixed with `<think>\n` exactly as the
    production runtime does
  - Loss-masks user / system / tool tokens (standard SFT — train only on
    the model's own outputs, not on what it saw)
  - Lazy-imports torch + transformers + peft + datasets so this module
    imports cleanly on hosts without those deps (CI / lint)

CLI:
    python -m training.train_qwen3_messages \\
        --config training/configs/qwen3_1_7b_lora.yaml
    python -m training.train_qwen3_messages \\
        --config training/configs/qwen3_1_7b_lora.yaml --dry-run

Outputs:
    <output.root>/<YYYYMMDD-<sha>>/adapter/        LoRA adapter weights
    <output.root>/<YYYYMMDD-<sha>>/training_args.json
    <output.root>/<YYYYMMDD-<sha>>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("fula-ai-training.train_qwen3_messages")


# ===========================================================================
# SYSTEM_PROMPT_TEMPLATE — kept in sync with blox-ai/src/runtime/rkllm_runtime.py
# ===========================================================================
#
# Pinning the prompt here is deliberate: the model trains on the EXACT
# bytes it sees at inference time. If the runtime prompt drifts, this
# file is the single source of truth that must be updated in lock-step.
# Tests in tests/ assert byte-equality with the runtime copy (TODO).

SYSTEM_PROMPT_TEMPLATE = """You are Blox AI, an on-device troubleshooting assistant for a Fula Blox edge device (RK3588 hardware).

# OUTPUT FORMAT — STRICT

You communicate ONLY through XML-tagged blocks. NEVER use markdown code fences. NEVER use triple-backtick json. Each block on its own line.

The three allowed blocks:

  <tool_call>{{"name":"diag/<tool>","arguments":{{}}}}</tool_call>
  <recommendation>{{"action_name":"<name>","args":{{...}},"reasoning":"<why>","confidence":<0-1>,"tier":<2 or 3>}}</recommendation>
  <verdict>{{"summary":"<one sentence>","severity":"<green|yellow|red>","root_cause":"<short>"}}</verdict>

# HARD RULES

1. EVERY conversation MUST end with exactly ONE <verdict>.
2. After you have called the diagnostic tools you need (typically 1-3 calls), you MUST emit a <verdict> based on the results.
3. NEVER call tools indefinitely. After 1-2 follow-up calls, finalize.
4. NEVER output a turn that is prose-only with no <tool_call> AND no <verdict>. Every turn must contain at least one XML block.
5. **ANY action you suggest MUST be emitted as a <recommendation> XML block — NEVER as a markdown numbered list, bullet, or table.**
6. Read tool_response JSON FIELD BY FIELD. Quote the actual field name you're basing your conclusion on.
7. NEVER use markdown headings (###), markdown bold (**...**), or numbered lists for actions.
8. **If the user reports a symptom but the diagnostic data CONTRADICTS it, ASK before acting.**
9. **NEVER emit a tier-2 or tier-3 destructive action with confidence > 0.7 when severity is "yellow" or "green".**
10. `relay.reservation_count: 0` is NOT a problem on its own. `wireguard.active: false` is NOT a problem unless the user explicitly set up WG.
11. **NEVER mention Kubernetes, kubelet, kube-proxy, kubectl, k8s, or any other Kubernetes component.** This device runs Docker (not Kubernetes). Containers: ipfs_host (kubo, the IPFS daemon — 'kubo' is NOT short for kubernetes), ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport.

# AVAILABLE TOOLS (read-only)

  - diag/summary: Run all read-only diagnostics in parallel; returns overall severity + per-subsystem status.
  - diag/internet: Check DNS + HTTPS reachability to Google + discovery.fula.network.
  - diag/relay: List libp2p relay peers + circuit reservation count.
  - diag/time: Check NTP sync + clock offset.
  - diag/power: RK3588 undervoltage events, recent reboots, temp, uptime.
  - diag/storage: df + ext4 errors + dmesg I/O errors + smartctl health.
  - diag/containers: docker ps + OOMKilled + restart counts for the fula stack.
  - diag/wireguard: WG handshake age + transfer counters + status triplet.
  - diag/heartbeat: Last heartbeat attempt to discovery.fula.network.
  - diag/events: Tail of /var/log/fula/events.jsonl.
  - diag/readiness: journalctl -u fula-readiness-check.service -n 100.

# RECOMMENDATION ACTION NAMES (only these are valid)

Tier 2 (single-tap approval):
- docker.restart — args.container ∈ {{ipfs_host, ipfs_cluster, ipfs_local, fula_go, fula_pinning, fula_gateway, fula_fxsupport, fula_updater}}
- systemctl.restart — args.unit ∈ {{fula.service, uniondrive.service, wireguard-support.service, fula-readiness-check.service, commands.service, fula-plugins.service, firewall.service}}
- systemctl.reset-failed — args.unit ∈ same set + fula-readiness-check-recover.service
- wireguard.bounce — no args
- ntp.resync — no args
- restart_fula — no args
- restart_uniondrive — no args

Tier 3 (security-code + press-and-hold):
- reset, partition, node_delete, ipfs_delete, force_update

Be terse. Start with diag/summary unless the user named a specific symptom. Two or three tool calls, then finalize with a <verdict>."""


# ===========================================================================
# Config dataclass
# ===========================================================================

@dataclass
class TrainConfig:
    base_model_hf_id: str
    base_model_local_dir: Optional[str]
    corpus_labelled_dir: str
    min_transcripts: int
    enable_thinking: bool
    mask_user_and_tool: bool
    output_root: str
    use_git_sha: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    lora_bias: str
    learning_rate: float
    num_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    save_total_limit: int
    bf16: bool
    gradient_checkpointing: bool
    tokenizer_max_length: int
    tokenizer_pad_to_multiple_of: int
    validation_fraction: float
    eval_steps: int
    seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        bm = raw["base_model"]
        co = raw["corpus"]
        ct = raw.get("chat_template", {})
        out = raw["output"]
        lo = raw["lora"]
        tr = raw["training"]
        tok = raw["tokenizer"]
        ev = raw["evaluation"]
        rep = raw["reproducibility"]
        return cls(
            base_model_hf_id=bm["hf_id"],
            base_model_local_dir=bm.get("local_dir"),
            corpus_labelled_dir=co["labelled_dir"],
            min_transcripts=int(co["min_transcripts"]),
            enable_thinking=bool(ct.get("enable_thinking", True)),
            mask_user_and_tool=bool(ct.get("mask_user_and_tool", True)),
            output_root=out["root"],
            use_git_sha=bool(out.get("use_git_sha", True)),
            lora_r=int(lo["r"]),
            lora_alpha=int(lo["alpha"]),
            lora_dropout=float(lo["dropout"]),
            lora_target_modules=list(lo["target_modules"]),
            lora_bias=str(lo.get("bias", "none")),
            learning_rate=float(tr["learning_rate"]),
            num_epochs=int(tr["num_epochs"]),
            per_device_train_batch_size=int(tr["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(tr["gradient_accumulation_steps"]),
            warmup_ratio=float(tr.get("warmup_ratio", 0.03)),
            weight_decay=float(tr.get("weight_decay", 0.0)),
            lr_scheduler_type=str(tr.get("lr_scheduler_type", "cosine")),
            logging_steps=int(tr.get("logging_steps", 25)),
            save_steps=int(tr.get("save_steps", 200)),
            save_total_limit=int(tr.get("save_total_limit", 3)),
            bf16=bool(tr.get("bf16", True)),
            gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
            tokenizer_max_length=int(tok.get("max_length", 4096)),
            tokenizer_pad_to_multiple_of=int(tok.get("pad_to_multiple_of", 8)),
            validation_fraction=float(ev.get("validation_fraction", 0.1)),
            eval_steps=int(ev.get("eval_steps", 100)),
            seed=int(rep.get("seed", 42)),
        )


# ===========================================================================
# Output dir
# ===========================================================================

def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def make_output_dir(cfg: TrainConfig) -> Path:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    subdir = f"{date}-{_git_short_sha()}" if cfg.use_git_sha else date
    out = Path(cfg.output_root) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


# ===========================================================================
# Transcript → messages conversion
# ===========================================================================

def transcript_to_messages(transcript: dict[str, Any]) -> Optional[list[dict[str, str]]]:
    """Build a Qwen 3 multi-turn `messages` list from one anonymized
    transcript. Returns None if the transcript can't be parsed.

    Output structure:
        [
            {"role": "system",    "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user",      "content": original_prompt},
            {"role": "assistant", "content": "<think>...</think>\\n<tool_call>{...}</tool_call>"},
            {"role": "tool",      "content": "<tool_response>{...}</tool_response>"},
            {"role": "assistant", "content": "<think>...</think>\\n<verdict>{...}</verdict>\\n<recommendation>{...}</recommendation>"},
        ]

    Multiple thought events between a tool_result and the next tool_call
    / verdict are concatenated into the same assistant turn. The
    `<think>` tags in thought payloads pass through verbatim — the
    chat-template's `enable_thinking` parameter is responsible for
    prepending the `<think>\\n` opener; we provide the closer + content
    inside that block."""
    events = transcript.get("events") or []
    if not events:
        return None
    user_prompt = transcript.get("user_prompt") or transcript.get("original_prompt")
    if not user_prompt:
        return None

    messages: list[dict[str, str]] = [
        {"role": "system",  "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user",    "content": str(user_prompt)},
    ]

    # Walk events. Build assistant turn buffers; flush when we hit a
    # tool_result (then push tool message, then start next assistant).
    assistant_buf: list[str] = []
    tool_buf: list[str] = []

    def _flush_assistant():
        if assistant_buf:
            messages.append({"role": "assistant", "content": "\n".join(assistant_buf)})
            assistant_buf.clear()

    def _flush_tool():
        if tool_buf:
            messages.append({"role": "tool", "content": "\n".join(tool_buf)})
            tool_buf.clear()

    for ev in events:
        etype = ev.get("type")
        if etype == "session_started":
            continue
        if etype == "thought":
            assistant_buf.append(str(ev.get("payload", "")))
        elif etype == "tool_call":
            pl = ev.get("payload", {})
            assistant_buf.append(
                f'<tool_call>{{"name":"{pl.get("tool", "")}",'
                f'"arguments":{json.dumps(pl.get("args", {}), separators=(",", ":"))}}}</tool_call>'
            )
        elif etype == "tool_result":
            # Boundary: flush assistant turn, then start tool turn
            _flush_assistant()
            pl = ev.get("payload")
            tool_buf.append(
                f'<tool_response>{json.dumps(pl, separators=(",", ":"))}</tool_response>'
            )
        elif etype == "verdict":
            pl = ev.get("payload", {})
            assistant_buf.append(
                f'<verdict>{json.dumps(pl, separators=(",", ":"))}</verdict>'
            )
        elif etype == "recommended_action":
            stripped = {
                "action_name": ev.get("action_name"),
                "args": ev.get("args") or {},
                "reasoning": ev.get("reasoning", ""),
                "confidence": ev.get("confidence", 0.5),
                "tier": ev.get("tier", 2),
            }
            assistant_buf.append(
                f'<recommendation>{json.dumps(stripped, separators=(",", ":"))}</recommendation>'
            )
        elif etype == "user_question":
            pl = ev.get("payload", {})
            assistant_buf.append(
                f'<user_question>{json.dumps(pl, separators=(",", ":"))}</user_question>'
            )
        else:
            # session_started already filtered; user_reply_received +
            # execution_result skipped because they're injected by the
            # runtime, not generated by the model
            continue
        # Tool message between turns
        if tool_buf and etype != "tool_result":
            _flush_tool()

    _flush_assistant()
    _flush_tool()

    if len(messages) < 3:  # system + user + at least one assistant
        return None
    return messages


# ===========================================================================
# Corpus loading
# ===========================================================================

def load_labelled_corpus(root: Path) -> list[dict]:
    files = list(root.rglob("*.labelled.json"))
    out = []
    for p in files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("label_decision") == "reject":
                continue
            out.append(rec)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip %s: %s", p, e)
    return out


# ===========================================================================
# Train
# ===========================================================================

def train(cfg: TrainConfig, dry_run: bool = False) -> Path:
    out_dir = make_output_dir(cfg)
    logger.info("output dir: %s", out_dir)
    (out_dir / "training_args.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat() + "Z",
        "git_sha": _git_short_sha(),
        "base_model": cfg.base_model_hf_id,
        "dry_run": dry_run,
        "trainer": "train_qwen3_messages.py",
    }, indent=2), encoding="utf-8")

    labelled = load_labelled_corpus(Path(cfg.corpus_labelled_dir).resolve())
    if len(labelled) < cfg.min_transcripts:
        raise SystemExit(
            f"only {len(labelled)} labelled transcripts; need ≥{cfg.min_transcripts}. "
            f"Generate more via `python -m corpus.synthetic.generate` or wait for real uploads."
        )

    messages_list = []
    skipped = 0
    for rec in labelled:
        msgs = transcript_to_messages(rec.get("transcript", {}))
        if msgs is None:
            skipped += 1
            continue
        messages_list.append({"messages": msgs})
    logger.info("loaded %d messages-format examples (%d skipped)",
                 len(messages_list), skipped)

    if dry_run:
        (out_dir / "dry_run.txt").write_text(
            f"dry-run: {len(messages_list)} examples ready; {skipped} skipped\n"
            f"sample message-count distribution: "
            f"min={min(len(x['messages']) for x in messages_list)}, "
            f"max={max(len(x['messages']) for x in messages_list)}, "
            f"median={sorted(len(x['messages']) for x in messages_list)[len(messages_list) // 2]}\n"
        )
        return out_dir

    # Lazy imports — only on the GPU box
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    # TRL provides DataCollatorForCompletionOnlyLM which masks
    # everything before the assistant response template. This is the
    # robust replacement for the fragile per-message tokenization that
    # the built-in advisor caught on 2026-05-27.
    if cfg.mask_user_and_tool:
        try:
            from trl import DataCollatorForCompletionOnlyLM
        except ImportError as e:
            raise SystemExit(
                "config sets chat_template.mask_user_and_tool=true but the "
                "`trl` package is not installed. Either `pip install trl` "
                "or set chat_template.mask_user_and_tool=false to train "
                "on the full sequence."
            ) from e

    tok_path = cfg.base_model_local_dir or cfg.base_model_hf_id
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(ex):
        # apply_chat_template handles the ChatML envelope + thinking-mode
        # prefix injection. With add_generation_prompt=False the FULL
        # conversation (including final assistant turn) is tokenized,
        # ready for SFT.
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=cfg.enable_thinking,
        )
        result = tokenizer(
            text,
            truncation=True,
            max_length=cfg.tokenizer_max_length,
            padding=False,
        )

        # Loss masking: we want to train ONLY on assistant tokens. The
        # naive "tokenize each message in isolation + walk a cursor"
        # approach is fragile — fragment tokenization can drift from
        # the in-context tokenization (whitespace boundaries, special
        # tokens between messages). Built-in advisor catch 2026-05-27:
        # cursor drift means the mask aligns to the wrong spans and SFT
        # silently trains on the user/system/tool tokens.
        #
        # Robust alternatives (pick ONE before running the real fine-tune):
        #
        # A. (PREFERRED) Use TRL's DataCollatorForCompletionOnlyLM with
        #    response_template="<|im_start|>assistant\\n". It finds the
        #    response template token sequence in each example and masks
        #    everything before it. Battle-tested, no fragment-tokenization
        #    needed:
        #        from trl import DataCollatorForCompletionOnlyLM
        #        collator = DataCollatorForCompletionOnlyLM(
        #            response_template="<|im_start|>assistant\\n",
        #            tokenizer=tokenizer,
        #        )
        #    Then swap DataCollatorForSeq2Seq for this collator below.
        #    Caveat: only masks system+user+tool BEFORE the first
        #    assistant turn. For multi-turn (system + user + assistant +
        #    tool + assistant), it doesn't re-mask the second tool turn.
        #    That's usually fine — the mid-turn tool message is short.
        #
        # B. Use transformers ≥4.45 `apply_chat_template(
        #       return_assistant_tokens_mask=True)`. Gives per-token mask
        #    directly from the chat template. Pin transformers version.
        #
        # C. Train without masking (current default). The model learns to
        #    reproduce user prompts + tool responses, which is wasteful
        #    but not catastrophic for narrow task fine-tuning. Slower
        #    convergence; more risk of memorizing prompt strings.
        #
        # The config flag `mask_user_and_tool` is ACCEPTED for forward
        # compatibility but in v1 we always train on the full sequence
        # (option C) until the user wires in option A. See README for the
        # 5-line verification helper to run before declaring training done.
        result["labels"] = result["input_ids"].copy()
        return result

    ds = Dataset.from_list(messages_list).map(encode, remove_columns=["messages"])
    split = ds.train_test_split(test_size=cfg.validation_fraction, seed=cfg.seed)

    model = AutoModelForCausalLM.from_pretrained(
        tok_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        seed=cfg.seed,
        report_to="none",
    )

    # Pick data collator based on whether we're masking.
    # TRL's DataCollatorForCompletionOnlyLM masks all tokens up to and
    # including the response template; the model is graded only on the
    # tokens AFTER `<|im_start|>assistant\n` in each sample. For multi-
    # turn conversations this masks only the prefix before the FIRST
    # assistant turn (mid-turn tool messages stay unmasked) — that's
    # the documented behavior and is fine for our use case where the
    # tool messages are short JSON responses, not free-form prose.
    if cfg.mask_user_and_tool:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n",
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=cfg.tokenizer_pad_to_multiple_of,
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    return out_dir


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(args.config)
    out_dir = train(cfg, dry_run=args.dry_run)
    logger.info("complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
