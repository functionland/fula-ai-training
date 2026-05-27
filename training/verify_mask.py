"""Quick verification helper — run BEFORE the full fine-tune.

Tokenizes one example with the chat template + your chosen mask logic,
then decodes back the non-masked positions. If the mask is correct, the
decoded string should be EXACTLY the concatenated assistant message
content (with <think>...</think> blocks + structured XML).

If the decoded string includes any user prompt text, `<tool_response>`
content, or system-prompt fragments, the mask is broken — the model
will train on the wrong tokens.

Usage:
    python -m training.verify_mask \\
        --config training/configs/qwen3_1_7b_lora.yaml \\
        --sample 1

Exit code:
    0 — mask appears correct (decoded == assistant content)
    1 — mask drift detected (decoded includes non-assistant fragments)
    2 — config / corpus load error

Built-in advisor catch 2026-05-27: silent loss-mask bugs are the #1
reason SFT runs produce models that look like they trained but didn't
learn the target behavior. Run this BEFORE the multi-hour GPU run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from training.train_qwen3_messages import (
    TrainConfig, load_labelled_corpus, transcript_to_messages,
)


logger = logging.getLogger("verify_mask")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Index of the labelled example to verify (default: 0)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    try:
        cfg = TrainConfig.from_yaml(args.config)
    except Exception as e:  # noqa: BLE001
        logger.error("config load failed: %s", e)
        return 2

    try:
        labelled = load_labelled_corpus(Path(cfg.corpus_labelled_dir).resolve())
    except Exception as e:  # noqa: BLE001
        logger.error("corpus load failed: %s", e)
        return 2

    if not labelled:
        logger.error("no labelled records found in %s", cfg.corpus_labelled_dir)
        return 2
    if args.sample >= len(labelled):
        logger.error("--sample=%d exceeds corpus size %d", args.sample, len(labelled))
        return 2

    msgs = transcript_to_messages(labelled[args.sample]["transcript"])
    if msgs is None:
        logger.error("sample %d transcript could not be parsed into messages", args.sample)
        return 2

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed; run on GPU box or install transformers")
        return 2

    tok_path = cfg.base_model_local_dir or cfg.base_model_hf_id
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        enable_thinking=cfg.enable_thinking,
    )
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]

    print("=" * 70)
    print(f"# Sample {args.sample}: {labelled[args.sample].get('upload_id', '?')}")
    print(f"# Messages: {len(msgs)} (system + user + N assistant/tool)")
    print(f"# Full token count: {len(full_ids)}")
    print(f"# Apply-chat-template length: {len(full_text)} chars")
    print("=" * 70)
    print("# Full rendered text (truncated to 2000 chars):")
    print(full_text[:2000])
    if len(full_text) > 2000:
        print(f"... [{len(full_text) - 2000} more chars]")
    print("=" * 70)

    # Expected assistant-only content (what mask should isolate)
    assistant_only = "\n".join(
        m["content"] for m in msgs if m["role"] == "assistant"
    )
    print("# What the mask SHOULD isolate (assistant content only):")
    print(assistant_only[:1500])
    if len(assistant_only) > 1500:
        print(f"... [{len(assistant_only) - 1500} more chars]")
    print("=" * 70)
    print(
        f"# If your collator/mask produces a decoded string equal to "
        f"the assistant-only content above (modulo "
        f"<|im_start|>assistant/<|im_end|> markers), the mask is "
        f"correct. If it includes the user prompt or <tool_response> "
        f"content, the mask is broken — switch to TRL's "
        f"DataCollatorForCompletionOnlyLM (see train_qwen3_messages.py "
        f"docstring) before running the full fine-tune."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
