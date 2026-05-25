"""19.4 — LoRA fine-tune Qwen 2.5 3B on labelled transcripts.

Wraps HuggingFace transformers + PEFT. Designed to run on a GPU box
(>=24 GB VRAM recommended for the default config). Imports of torch/
transformers/peft/datasets are LAZY so this module imports cleanly
on hosts without those deps installed (lets the test suite + CI
import + lint without needing the full training stack).

CLI:
    python -m training.lora_train --config training/configs/qwen3b_lora.yaml

Outputs:
    <output.root>/<YYYYMMDD-<git_sha>>/adapter/    LoRA adapter weights
    <output.root>/<YYYYMMDD-<git_sha>>/training_args.json
    <output.root>/<YYYYMMDD-<git_sha>>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


logger = logging.getLogger("fula-ai-training.lora_train")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    base_model_hf_id: str
    base_model_local_dir: Optional[str]
    corpus_labelled_dir: str
    min_transcripts: int
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


# ---------------------------------------------------------------------------
# Output dir naming
# ---------------------------------------------------------------------------

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
    date = datetime.utcnow().strftime("%Y%m%d")
    if cfg.use_git_sha:
        subdir = f"{date}-{_git_short_sha()}"
    else:
        subdir = date
    out = Path(cfg.output_root) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_labelled_corpus(root: Path) -> list[dict]:
    """Walk corpus/labelled/ for *.labelled.json files. Each is one
    labelled transcript (per labeller/app.py output shape)."""
    files = list(root.rglob("*.labelled.json"))
    out = []
    for p in files:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip %s: %s", p, e)
    return out


def format_training_example(labelled: dict) -> Optional[dict]:
    """Convert one labelled transcript into a {prompt, completion} pair
    for SFT. Skips transcripts the labeller marked as 'reject' (the
    model's behavior was wrong end-to-end — training on it would teach
    the wrong thing)."""
    if labelled.get("label_decision") == "reject":
        return None
    events = labelled.get("transcript", {}).get("events") or []
    user_prompt = next(
        (e.get("payload") for e in events
         if e.get("type") == "session_started"), None,
    )
    # Build the assistant transcript as our SFT target. For now, simply
    # concatenate all assistant-side events (thought/tool_call/verdict/
    # recommendation) with the XML tags the production prompt teaches.
    parts = []
    for e in events:
        t = e.get("type")
        if t == "thought":
            parts.append(str(e.get("payload", "")))
        elif t == "tool_call":
            pl = e.get("payload", {})
            parts.append(
                f'<tool_call>{{"name":"{pl.get("tool", "")}",'
                f'"arguments":{json.dumps(pl.get("args", {}), separators=(",", ":"))}}}</tool_call>'
            )
        elif t == "verdict":
            pl = e.get("payload", {})
            parts.append(f'<verdict>{json.dumps(pl, separators=(",", ":"))}</verdict>')
        elif t == "recommended_action":
            stripped = {
                "action_name": e.get("action_name"),
                "args": e.get("args") or {},
                "reasoning": e.get("reasoning", ""),
                "confidence": e.get("confidence", 0.5),
                "tier": e.get("tier", 2),
            }
            parts.append(f'<recommendation>{json.dumps(stripped, separators=(",", ":"))}</recommendation>')
    if not parts:
        return None
    completion = "\n".join(parts)
    prompt = (
        labelled.get("transcript", {}).get("original_prompt")
        or "Diagnose this device."
    )
    return {"prompt": prompt, "completion": completion}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, dry_run: bool = False) -> Path:
    """Run the LoRA fine-tune. Returns the output dir.

    `dry_run=True` does the dataset / config plumbing but skips the
    actual `trainer.train()` call — useful for CI smoke tests that
    don't have a GPU."""
    out_dir = make_output_dir(cfg)
    logger.info("output dir: %s", out_dir)

    # Save config + git state alongside the run for forensics
    (out_dir / "training_args.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(json.dumps({
        "started_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_short_sha(),
        "base_model": cfg.base_model_hf_id,
        "dry_run": dry_run,
    }, indent=2), encoding="utf-8")

    labelled = load_labelled_corpus(Path(cfg.corpus_labelled_dir).resolve())
    if len(labelled) < cfg.min_transcripts:
        raise SystemExit(
            f"only {len(labelled)} labelled transcripts; need ≥{cfg.min_transcripts}. "
            f"Wait for more uploads + labelling, or lower min_transcripts."
        )
    examples = [format_training_example(L) for L in labelled]
    examples = [e for e in examples if e is not None]
    logger.info("training on %d examples (after rejected filter)", len(examples))

    if dry_run:
        (out_dir / "dry_run.txt").write_text("dry-run complete; no model touched")
        return out_dir

    # Lazy imports — only on the GPU box that has these installed
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )

    tok_path = cfg.base_model_local_dir or cfg.base_model_hf_id
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(ex):
        text = f"{ex['prompt']}\n{ex['completion']}{tokenizer.eos_token}"
        return tokenizer(
            text,
            truncation=True,
            max_length=cfg.tokenizer_max_length,
            padding=False,
        )

    ds = Dataset.from_list(examples).map(encode, remove_columns=["prompt", "completion"])
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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the plumbing but skip the actual training step. "
                             "Useful on hosts without a GPU.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)
    cfg = TrainConfig.from_yaml(args.config)
    out_dir = train(cfg, dry_run=args.dry_run)
    logger.info("complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
