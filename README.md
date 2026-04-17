# DiTReducio: Training-Free Calibration and Acceleration for DiT-based Text-to-Speech

<a href="https://arxiv.org/abs/2509.09748"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>

Official implementation of **DiTReducio**, a training-free calibration and acceleration framework for DiT-based text-to-speech models. DiTReducio identifies temporal and branch redundancy in DiT inference and applies compression strategies via progressive calibration to achieve speedup without training cost.

## News

- **2026/04**: 🎉 Our paper has been accepted to **ACL 2026 Findings**! 

## Overview

DiTReducio is a training-free acceleration framework that eliminates redundant computations in DiT-based TTS models via progressive calibration. 

### Core Compression Strategies
- **Temporal Skipping (TS)**: Caches module outputs at a given timestep and reuses them in subsequent steps to avoid temporally redundant computation.
- **Branch Skipping (BS)**: Skips the redundant unconditional branch in Classifier-Free Guidance and reconstructs it via a **Branch Residual** mechanism to preserve essential guidance details.

### Three-Phase Progressive Calibration
1. **Check Phase**: Identifies highly temporally redundant layer-step pairs by detecting **diagonal-like attention patterns**.
2. **Pre-Calibration Phase**: Selectively applies TS to marked pairs to ensure a superior strategy combination and avoid suboptimal compression.
3. **Calibration Phase**: Systematically applies both TS and BS across all layer-step pairs.

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
bash scripts/fetch_backends.sh <target-root>
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
# F5-TTS
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

## Experiments

### Threshold Sweep (T0–T6)

```bash
# Full sweep: calibrate + infer + eval for all thresholds
python scripts/run_sweep_f5.py \
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

## Citation

```bibtex
@article{huo2025ditreducio,
  title={Ditreducio: A training-free acceleration for dit-based tts via progressive calibration},
  author={Huo, Yanru and Jiang, Ziyue and Tang, Zuoli and Hong, Qingyang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2509.09748},
  year={2025}
}
```

## Acknowledgements

Our approach is inspired by [DiTFastAttn](https://github.com/thu-nics/DiTFastAttn) for training-free acceleration in diffusion transformers. We build upon [F5-TTS](https://github.com/SWivid/F5-TTS) and [MegaTTS3](https://github.com/bytedance/MegaTTS3) for the baseline TTS models. 
