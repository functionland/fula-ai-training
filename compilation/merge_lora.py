"""19.6a — merge LoRA adapter into base model weights.

Output: training/output/<run>/merged/  (HF format; ready for RKLLM convert)

Lazy imports for the same reason as lora_train.py — modules import
cleanly on hosts without torch/transformers/peft.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger("fula-ai-training.merge_lora")


def merge(adapter_dir: Path, base_model: str, output_dir: Path,
          dry_run: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        (output_dir / "dry_run.txt").write_text(
            f"would merge {adapter_dir} into {base_model} -> {output_dir}\n"
        )
        return output_dir
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("merged: %s", output_dir)
    return output_dir


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True,
                        help="Path to the LoRA adapter dir (from lora_train.py output)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write merged HF weights (input to rkllm_convert.sh)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)
    merge(Path(args.adapter), args.base_model, Path(args.output_dir),
          dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
