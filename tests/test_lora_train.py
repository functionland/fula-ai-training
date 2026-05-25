"""19.4 — lora_train tests. The heavy torch/transformers imports happen
INSIDE train(); module imports cleanly without them. Tests cover the
non-GPU plumbing (config parsing, corpus loading, dataset shaping)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.lora_train import (
    TrainConfig, format_training_example, load_labelled_corpus,
    make_output_dir,
)


@pytest.fixture
def sample_config_yaml(tmp_path) -> str:
    """Minimal valid YAML matching the qwen3b_lora.yaml structure."""
    yaml_text = """
base_model:
  hf_id: Qwen/Qwen2.5-3B-Instruct
  local_dir: null
corpus:
  labelled_dir: corpus/labelled
  min_transcripts: 100
output:
  root: training/output
  use_git_sha: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj]
  bias: none
training:
  learning_rate: 2.0e-4
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  warmup_ratio: 0.03
  weight_decay: 0.0
  lr_scheduler_type: cosine
  logging_steps: 25
  save_steps: 200
  save_total_limit: 3
  bf16: true
  gradient_checkpointing: true
tokenizer:
  max_length: 4096
  pad_to_multiple_of: 8
evaluation:
  validation_fraction: 0.1
  eval_steps: 100
reproducibility:
  seed: 42
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    return str(p)


def test_config_parses_complete(sample_config_yaml):
    cfg = TrainConfig.from_yaml(sample_config_yaml)
    assert cfg.base_model_hf_id == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.lora_target_modules == ["q_proj", "k_proj"]
    assert cfg.bf16 is True
    assert cfg.seed == 42


def test_format_training_example_with_full_transcript():
    labelled = {
        "label_decision": "accept",
        "transcript": {
            "original_prompt": "device is slow",
            "events": [
                {"type": "session_started", "payload": "x"},
                {"type": "thought", "payload": "checking summary"},
                {"type": "tool_call",
                 "payload": {"tool": "diag/summary", "args": {}}},
                {"type": "verdict",
                 "payload": {"summary": "all good", "severity": "green",
                             "root_cause": "no_issue"}},
                {"type": "recommended_action",
                 "action_name": "docker.restart",
                 "args": {"container": "ipfs_host"},
                 "reasoning": "kubo wedge",
                 "confidence": 0.8,
                 "tier": 2},
            ],
        },
    }
    out = format_training_example(labelled)
    assert out is not None
    assert "device is slow" == out["prompt"]
    assert "<tool_call>" in out["completion"]
    assert "<verdict>" in out["completion"]
    assert "<recommendation>" in out["completion"]
    assert "checking summary" in out["completion"]


def test_format_training_example_skips_rejected():
    """label_decision=reject means the model's behavior was WRONG.
    Training on it would teach the wrong thing — skip."""
    labelled = {
        "label_decision": "reject",
        "transcript": {"events": []},
    }
    assert format_training_example(labelled) is None


def test_format_training_example_skips_empty():
    """No events to learn from -> skip."""
    labelled = {
        "label_decision": "accept",
        "transcript": {"events": []},
    }
    assert format_training_example(labelled) is None


def test_load_labelled_corpus_walks_jsonl(tmp_path):
    d = tmp_path / "labelled"
    d.mkdir()
    (d / "a.labelled.json").write_text(json.dumps({
        "upload_id": "a", "label_decision": "accept", "transcript": {},
    }))
    (d / "b.labelled.json").write_text(json.dumps({
        "upload_id": "b", "label_decision": "reject", "transcript": {},
    }))
    items = load_labelled_corpus(d)
    assert len(items) == 2
    ids = {x["upload_id"] for x in items}
    assert ids == {"a", "b"}


def test_load_labelled_corpus_skips_malformed(tmp_path):
    d = tmp_path / "labelled"
    d.mkdir()
    (d / "good.labelled.json").write_text(json.dumps({"upload_id": "g"}))
    (d / "bad.labelled.json").write_text("{not json")
    items = load_labelled_corpus(d)
    assert len(items) == 1


def test_make_output_dir_creates(tmp_path, sample_config_yaml):
    cfg = TrainConfig.from_yaml(sample_config_yaml)
    cfg.output_root = str(tmp_path)
    cfg.use_git_sha = False
    out = make_output_dir(cfg)
    assert out.exists()
    assert out.is_dir()
    # Subdir is a date
    import re
    assert re.match(r"^\d{8}$", out.name)
