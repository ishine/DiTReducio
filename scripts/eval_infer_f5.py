"""Batch inference for F5-TTS on LibriSpeech cross-sentence pairs.

Generates audio for each pair, measures RTF, and saves per-sample timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch

# ── add backend to path ────────────────────────────────────────────────────
BACKEND_ROOT = os.environ.get(
    "F5TTS_BACKEND_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "backends", "F5-TTS"),
)
sys.path.insert(0, BACKEND_ROOT)
sys.path.insert(0, os.path.join(BACKEND_ROOT, "src"))
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "src"))

# need to import after path setup
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from ditreducio.backends.f5tts.hooks import make_view, pre_calibration_hook
from ditreducio.calibration.hooks import (
    calibration_preparation,
    calibration_reset,
    calibration_reset_step,
    speedup,
)
from ditreducio.calibration.util import seed_everything
from omegaconf import OmegaConf
from hydra.utils import get_class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--lst_file", required=True)
    p.add_argument("--librispeech_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--methods_path", default="methods")
    p.add_argument("--nfe_step", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--model", default="F5TTS_v1_Base")
    p.add_argument(
        "--model_cfg",
        default=os.path.join(BACKEND_ROOT, "src/f5_tts/configs/F5TTS_v1_Base.yaml"),
    )
    p.add_argument(
        "--ckpt_file",
        default=os.environ.get("F5TTS_CKPT", ""),
        help="Path to F5-TTS model checkpoint (or set F5TTS_CKPT env var)",
    )
    p.add_argument(
        "--vocoder_path",
        default=os.environ.get("VOCODER_PATH", ""),
        help="Path to vocoder checkpoint (or set VOCODER_PATH env var)",
    )
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--nots", action="store_true")
    p.add_argument("--nobs", action="store_true")
    p.add_argument("--nopre", action="store_true")
    p.add_argument(
        "--bs_mode", default="residual", choices=["residual", "cond_replace"]
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # ── load models ─────────────────────────────────────────────────────
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=args.vocoder_path,
        device=args.device,
    )

    model_cfg = OmegaConf.load(args.model_cfg)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ema_model = load_model(
        model_cls, model_arc, args.ckpt_file, mel_spec_type="vocos", device=args.device
    )
    ema_model.transformer.bs_mode = args.bs_mode

    # Build view for shared calibration API
    view = make_view(ema_model)

    # ── load strategy if delta > 0 ──────────────────────────────────────
    delta = args.delta if args.delta > 0 else None
    if delta is not None:
        ema_model.transformer.nots = args.nots
        ema_model.transformer.nobs = args.nobs
        speedup(
            view, steps=args.nfe_step, delta=delta, methods_path=args.methods_path
        )
    else:
        calibration_preparation(view, steps=args.nfe_step)

    # ── read lst ────────────────────────────────────────────────────────
    with open(args.lst_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.max_samples > 0:
        lines = lines[: args.max_samples]

    # ── batch infer ─────────────────────────────────────────────────────
    total_infer_time = 0.0
    total_audio_dur = 0.0
    n_processed = 0
    timing_rows = []

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) != 6:
            continue
        src_id, src_dur, src_text, tgt_id, tgt_dur, tgt_text = parts
        out_path = os.path.join(args.output_dir, f"{tgt_id}.flac")
        if args.skip_existing and os.path.exists(out_path):
            continue

        # source audio path
        spk, chap, _ = src_id.split("-")
        ref_audio = os.path.join(args.librispeech_root, spk, chap, f"{src_id}.flac")
        if not os.path.exists(ref_audio):
            print(f"[WARN] ref audio not found: {ref_audio}")
            continue

        ref_text = src_text
        gen_text = " " + tgt_text

        # preprocess
        try:
            ref_audio_pp, ref_text_pp = preprocess_ref_audio_text(ref_audio, ref_text)
        except Exception as e:
            print(f"[WARN] preprocess failed for {src_id}: {e}")
            continue

        # infer
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            generated_segments = []
            reg1 = r"(?=\[\w+\])"
            chunks = re.split(reg1, gen_text)
            reg2 = r"\[(\w+)\]"
            for text in chunks:
                if not text.strip():
                    continue
                text = re.sub(reg2, "", text)
                audio_seg, sr, _ = infer_process(
                    ref_audio_pp,
                    ref_text_pp,
                    text.strip(),
                    ema_model,
                    vocoder,
                    mel_spec_type="vocos",
                    target_rms=0.1,
                    cross_fade_duration=0.15,
                    nfe_step=args.nfe_step,
                    cfg_strength=args.cfg_strength,
                    sway_sampling_coef=args.sway_sampling_coef,
                    speed=1.0,
                    fix_duration=None,
                    device=args.device,
                )
                generated_segments.append(audio_seg)
                calibration_reset_step(view)

            if generated_segments:
                final_wave = np.concatenate(generated_segments)
                sf.write(out_path, final_wave, sr)
                audio_dur = len(final_wave) / sr
            else:
                continue

            torch.cuda.synchronize()
            infer_time = time.perf_counter() - t0

            # skip first 2 for warmup
            if i >= 2:
                total_infer_time += infer_time
                total_audio_dur += audio_dur
                n_processed += 1

            timing_rows.append(
                {
                    "id": tgt_id,
                    "infer_time": f"{infer_time:.4f}",
                    "audio_dur": f"{audio_dur:.4f}",
                }
            )
            calibration_reset_step(view)

            if i % 50 == 0:
                print(
                    f"[{i}/{len(lines)}] {tgt_id} infer={infer_time:.3f}s dur={audio_dur:.3f}s"
                )

        except Exception as e:
            print(f"[ERR] inference failed for {tgt_id}: {e}")
            calibration_reset_step(view)
            continue

    # ── save timing ─────────────────────────────────────────────────────
    rtf = total_infer_time / total_audio_dur if total_audio_dur > 0 else float("inf")
    print(f"\n====f5tts-rtf-delta{delta}-step{args.nfe_step}-seed{args.seed}====")
    print(f"RTF: {rtf:.4f}")
    print(f"Samples: {n_processed}")

    # save timing csv
    timing_path = os.path.join(args.output_dir, "timing.csv")
    with open(timing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "infer_time", "audio_dur"])
        w.writeheader()
        w.writerows(timing_rows)

    # save RTF
    with open(os.path.join(args.output_dir, "rtf.txt"), "w") as f:
        f.write(f"{rtf:.6f}")


if __name__ == "__main__":
    main()
