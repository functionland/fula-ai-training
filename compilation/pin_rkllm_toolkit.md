# RKLLM Toolkit version pin

The Rockchip RKLLM-Toolkit version determines the binary format of `.rkllm` model files. The on-device `librkllmrt.so` must match major.minor. **A toolkit upgrade is a breaking change for any device running an older runtime.**

## Currently pinned

| Component | Version | Source |
|---|---|---|
| RKLLM-Toolkit (build-host) | 1.1.4 | https://github.com/airockchip/rknn-llm/tree/release-v1.1.4 |
| On-device librkllmrt.so | 1.1.4 | extracted from `functionland/loyal-agent:latest` image (proven working) |

These MUST stay in sync. If you upgrade the build-host toolkit, also upgrade the device runtime in a coordinated canary release.

## Why 1.1.4

- The community Qwen 2.5 3B model we're seeding from (`c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1`) was built against toolkit 1.1.1; runtime 1.1.4 is backward-compatible with it (verified end-to-end on lab `pi@192.168.68.107` 2026-05-25).
- 1.2.x has unrelated regressions on this kernel build (Rockchip vendor 6.1.115-rk35xx).

## How to install on a build host

Rockchip distributes the toolkit as a wheel from their developer portal:

1. Register at https://www.rockchip.com/ — get developer credentials.
2. Download `rkllm_toolkit-<version>-cp310-cp310-linux_x86_64.whl` matching the pin above.
3. `pip install rkllm_toolkit-<version>-cp310-cp310-linux_x86_64.whl`
4. Verify: `python3 -c "from rkllm.api import RKLLM; print('ok')"`

Requires CUDA 11.8 + 24+ GB GPU memory for Qwen 3B conversion.

## When to bump

- New base model architecture not supported by current toolkit (e.g. Llama 4 / Qwen 3).
- Performance regression on RK3588 NPU that newer toolkit claims to fix (benchmark BEFORE adopting).

Never bump without:
1. Successfully converting a known-good Qwen 2.5 3B model with the new toolkit.
2. Loading the resulting `.rkllm` on the lab device + running 5+ canonical test scenarios end-to-end.
3. Decision on whether the on-device runtime also needs to bump (it usually does for major versions).

Document any bump in this file + the parent fula-ota plugin's `PLAYBOOK.md`.
