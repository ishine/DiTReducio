# DiTReducio: Training-Free Calibration and Acceleration for DiT-based Text-to-Speech

[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-blue)](https://2026.aclweb.org/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-brightgreen)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green)]()

Official implementation of **DiTReducio**, a training-free calibration and acceleration framework for DiT-based text-to-speech models. DiTReducio identifies temporal and branch redundancy in DiT inference and applies progressive compression strategies to achieve significant speedup with minimal quality loss.

## News

- **2026/04**: DiTReducio has been accepted at **ACL 2026 Findings**.

## Overview

DiTReducio introduces two complementary compression strategies:

- **Temporal Skipping (TS)**: Caches module outputs from the preceding timestep and reuses them when temporally redundant, avoiding redundant recomputation.

- **Branch Skipping (BS)**: Under Classifier-Free Guidance (CFG), skips the unconditional branch and reconstructs it via a **Branch Residual** mechanism that preserves essential guidance details.

The framework operates through a **three-phase progressive calibration**:

1. **Check Phase**: Identifies highly temporally redundant layer-step pairs by analyzing attention pattern similarity with diagonal matrices.
2. **Pre-Calibration Phase**: Applies TS to the marked pairs for preliminary strategy assignment.
3. **Calibration Phase**: Performs greedy search over both TS and BS for all remaining pairs using a dynamic threshold.

After calibration, the resulting **strategy table** is saved and can be loaded for plug-and-play accelerated inference.

## Results

Performance on **LibriSpeech-PC-test-clean** (averaged over 5 seeds):

| Model | Metric | T0 (Baseline) | T1 | T2 | T3 | **T4** | T5 | T6 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F5-TTS** | SIM-o | 0.640 | 0.640 | 0.637 | 0.629 | **0.618** | 0.610 | 0.590 |
| | WER (%) | 2.636 | 2.655 | 2.564 | 2.643 | **2.634** | 2.661 | 2.900 |
| | RTF | 0.178 | 0.165 | 0.149 | 0.138 | **0.129** | 0.120 | 0.112 |
| | Ops Ratio (%) | 100.00 | 82.59 | 66.38 | 55.09 | **45.58** | 39.26 | 34.42 |
| **MegaTTS 3** | SIM-o | 0.750 | 0.750 | 0.748 | 0.743 | **0.734** | 0.691 | 0.626 |
| | WER (%) | 3.112 | 3.112 | 3.110 | 3.073 | **3.095** | 3.133 | 3.030 |
| | RTF | 0.396 | 0.395 | 0.359 | 0.287 | **0.224** | 0.176 | 0.156 |
| | Ops Ratio (%) | 100.00 | 98.87 | 88.02 | 68.19 | **48.94** | 33.88 | 27.52 |

**T4** represents the optimal balance point. DiTReducio achieves **1.37x** speedup for F5-TTS and **1.76x** for MegaTTS 3 at T4 with no significant quality degradation.

## Installation

```bash
cd DiTReducio
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Backend Dependencies

DiTReducio requires the upstream TTS model code:

- **F5-TTS**: Clone from [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) and set the path in config
- **MegaTTS 3**: Clone from [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3) and set the path in config

Or use the provided setup script:

```bash
bash scripts/fetch_backends.sh
```

## Quick Start

### 1. Configure

Copy and edit the example config:

```bash
cp configs/f5tts.example.yaml configs/local.f5tts.yaml
# Edit paths in configs/local.f5tts.yaml
```

Key path fields:

| Field | Description |
|---|---|
| `paths.backend_code_root` | F5-TTS or MegaTTS 3 code root directory |
| `paths.backend_ckpt_root` | Model weights directory |
| `paths.strategy_output_root` | Strategy table output directory |
| `paths.inference_output_root` | Inference audio output directory |

### 2. Calibrate

Run the three-phase calibration to generate a strategy table:

```bash
# F5-TTS (delta=0.20 corresponds to T4)
python -m ditreducio.cli.calibrate --backend f5tts --config configs/local.f5tts.yaml --delta 0.2

# MegaTTS 3
python -m ditreducio.cli.calibrate --backend megatts3 --config configs/local.megatts3.yaml --delta 0.8
```

### 3. Accelerated Inference

Load the saved strategy table and run accelerated inference:

```bash
# F5-TTS
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --delta 0.2

# MegaTTS 3
python -m ditreducio.cli.infer --backend megatts3 --config configs/local.megatts3.yaml --delta 0.8
```

### Ablation Presets

```bash
# Disable pre-check and pre-calibration
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --preset no_pre --delta 0.2

# Only use Branch Skipping
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --preset only_bs --delta 0.2

# Only use Temporal Skipping
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --preset only_ts --delta 0.2

# Branch Skipping with conditional replacement
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --preset bs_cond_replace --delta 0.2

# Branch Skipping with unconditional replacement
python -m ditreducio.cli.infer --backend f5tts --config configs/local.f5tts.yaml --preset bs_uncond_replace --delta 0.2
```

## Experiments

### Threshold Sweep (T0–T6)

```bash
# Full sweep: calibrate + infer + eval for all thresholds
CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_f5.py \
    --backend_root /path/to/F5-TTS \
    --f5tts_ckpt /path/to/model_1250000.safetensors \
    --vocoder_path /path/to/vocos-mel-24khz \
    --data_root /path/to/LibriSpeech \
    --dataset clean

# Custom thresholds
python scripts/run_sweep_f5.py \
    --backend_root /path/to/F5-TTS \
    --f5tts_ckpt /path/to/model_1250000.safetensors \
    --vocoder_path /path/to/vocos-mel-24khz \
    --data_root /path/to/LibriSpeech \
    --deltas 0.0 0.05 0.1 0.15 0.2 0.25 0.3
```

### Evaluation (WER + SIM-o)

Evaluation models are **auto-downloaded** on first use (faster-whisper-large-v3 and ECAPA-TDNN WavLM checkpoint). To use local weights instead, specify the paths:

```bash
# Auto-download evaluation models
python scripts/eval_metrics.py \
    --gen_dir <dir> --lst_file <lst> --librispeech_root <root> --device cuda

# Use local evaluation model weights
python scripts/eval_metrics.py \
    --gen_dir <dir> --lst_file <lst> --librispeech_root <root> \
    --whisper_ckpt /path/to/faster-whisper-large-v3 \
    --ecapa_ckpt /path/to/wavlm_large_finetune.pth \
    --device cuda
```

### Data Preparation

```bash
# Generate cross-sentence lst from LibriSpeech JSON
python scripts/gen_lst.py --data_root /path/to/LibriSpeech
python scripts/gen_lst_other.py --data_root /path/to/LibriSpeech-other
```

## Repository Structure

```
DiTReducio/
  configs/             # YAML configuration files
  src/ditreducio/
    cli/               # calibrate / infer command-line interface
    core/              # Type definitions, config loading, registry
    calibration/       # Shared calibration logic:
                       #   accessor.py  — TransformerView abstraction
                       #   hooks.py     — shared calibration flow
                       #   metrics.py   — compression loss
                       #   timer.py     — CUDA timing
                       #   util.py      — threshold_q, seed_everything
    methods/           # TS / BS compression method implementations
    ablation/          # Ablation preset definitions
    backends/          # Backend adapters and runtime code:
                       #   f5tts_adapter.py  / megatts3_adapter.py — subprocess adapters
                       #   f5tts/       — cli.py, hooks.py, flops_tracker.py, ecapa_tdnn.py
                       #   megatts3/    — cli.py, hooks.py, flops_tracker.py
  scripts/             # Experiment scripts:
                       #   run_sweep_f5.py   — threshold sweep (T0–T6)
                       #   eval_infer_f5.py  — batch inference
                       #   eval_metrics.py   — WER + SIM-o evaluation
                       #   gen_lst.py        — data preparation
  outputs/             # Experiment outputs
    strategies/        # Strategy table JSON files
    audio/             # Generated audio files
    metrics/           # CSV metric files
  paper.txt            # Paper source (experiment definitions)
```

## Architecture

The codebase uses a **TransformerView** abstraction to share calibration logic across backends with different model architectures:

```
F5-TTS:   model.transformer.transformer_blocks[i].attn / .ff
MegaTTS3: model.dit.encoder.layers[i].attention / .feed_forward
                       ↓ TransformerView ↓
           Shared calibration hooks (calibration_reset, calibration_preparation,
           pre_calibration, calibration, speedup, ...)
```

Backend-specific code (efficient forward implementations, attention hooks, FLOPs tracking) lives in `backends/f5tts/` and `backends/megatts3/`, while shared calibration flow is in `calibration/`.

## Citation

```bibtex
@inproceedings{ditreducio2026,
  title     = {DiTReducio: Training-Free Calibration and Acceleration for DiT-based Text-to-Speech},
  author    = {},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

## License

This project is licensed under the Apache License 2.0.

## Acknowledgements

We build upon [F5-TTS](https://github.com/SWivid/F5-TTS) and [MegaTTS3](https://github.com/bytedance/MegaTTS3) for the baseline TTS models. Our approach is inspired by [DiTFastAttn](https://github.com/deep-diver/DiTFastAttn) for attention-based acceleration in diffusion transformers.
