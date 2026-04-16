"""Evaluate SIM-o and WER for generated audio against LibriSpeech references.

Usage:
    python scripts/eval_metrics.py --gen_dir <dir> --lst_file <lst> --librispeech_root <root> \
        --whisper_ckpt <path> --device cuda
"""

from __future__ import annotations

import argparse
import os
import string
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from jiwer import wer as jiwer_wer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gen_dir", required=True, help="Directory with generated .flac files"
    )
    p.add_argument("--lst_file", required=True, help="Cross-sentence lst file")
    p.add_argument(
        "--librispeech_root",
        required=True,
        help="LibriSpeech root (test-clean or test-other)",
    )
    p.add_argument(
        "--whisper_ckpt",
        default="",
        help="Path to local faster-whisper model dir (empty = auto-download large-v3)",
    )
    p.add_argument(
        "--ecapa_ckpt",
        default="",
        help="Path to local ECAPA-TDNN checkpoint .pth file (empty = auto-download)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_sim", action="store_true")
    p.add_argument("--skip_wer", action="store_true")
    return p.parse_args()


def load_test_pairs(lst_file: str, librispeech_root: str, gen_dir: str):
    """Load (gen_wav, ref_wav, gen_text) tuples from lst file."""
    pairs = []
    with open(lst_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            src_id, _, src_text, tgt_id, _, tgt_text = parts
            ref_spk, ref_chap, _ = src_id.split("-")
            ref_wav = os.path.join(
                librispeech_root, ref_spk, ref_chap, f"{src_id}.flac"
            )
            gen_wav = os.path.join(gen_dir, f"{tgt_id}.flac")
            if os.path.exists(gen_wav) and os.path.exists(ref_wav):
                pairs.append((gen_wav, ref_wav, " " + tgt_text))
    return pairs


def _download_ecapa_ckpt(save_dir: str | None = None) -> str:
    """Download ECAPA-TDNN WavLM finetuned checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = "UniSpeech/wavlm_large_finetune"
    filename = "wavlm_large_finetune.pth"
    if save_dir:
        return hf_hub_download(repo_id, filename, local_dir=save_dir)
    return hf_hub_download(repo_id, filename)


def eval_sim(pairs, ecapa_ckpt: str, device: str):
    """Compute speaker similarity using ECAPA-TDNN + WavLM."""
    from ditreducio.backends.f5tts.ecapa_tdnn import ECAPA_TDNN_SMALL

    ckpt_path = ecapa_ckpt
    if not ckpt_path:
        # Auto-download from HuggingFace
        ckpt_path = _download_ecapa_ckpt()
    elif os.path.isdir(ckpt_path):
        for fn in ("wavlm_large_finetune.pth", "classifier.ckpt", "model.ckpt"):
            candidate = os.path.join(ckpt_path, fn)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, weights_only=True, map_location=lambda storage, loc: storage)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    sims = []
    for gen_wav, ref_wav, _ in tqdm(pairs, desc="SIM-o"):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(ref_wav)
        wav1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)(wav1).to(
            device
        )
        wav2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)(wav2).to(
            device
        )
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)
        sims.append(F.cosine_similarity(emb1, emb2)[0].item())

    avg_sim = np.mean(sims) if sims else 0.0
    return round(avg_sim, 3)


def eval_wer(pairs, whisper_ckpt: str, device: str):
    """Compute WER using faster-whisper."""
    from faster_whisper import WhisperModel

    model_size = "large-v3"
    compute_type = "float16" if device.startswith("cuda") else "int8"
    kwargs = {"device": device, "compute_type": compute_type}
    if whisper_ckpt and os.path.isdir(whisper_ckpt):
        model_size = whisper_ckpt
    asr_model = WhisperModel(model_size, **kwargs)

    punctuation_all = string.punctuation
    wers = []
    for gen_wav, _, truth in tqdm(pairs, desc="WER"):
        segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
        hypo = " ".join(seg.text for seg in segments)

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")
        truth = truth.lower().replace("  ", " ").strip()
        hypo = hypo.lower().replace("  ", " ").strip()

        wers.append(jiwer_wer(truth, hypo))

    avg_wer = np.mean(wers) * 100 if wers else 0.0
    return round(avg_wer, 3)


def main():
    args = parse_args()
    pairs = load_test_pairs(args.lst_file, args.librispeech_root, args.gen_dir)
    print(f"Found {len(pairs)} pairs to evaluate")

    sim_score = None
    wer_score = None

    if not args.skip_sim:
        sim_score = eval_sim(
            pairs, args.ecapa_ckpt, args.device
        )
        print(f"SIM-o: {sim_score}")

    if not args.skip_wer:
        wer_score = eval_wer(pairs, args.whisper_ckpt, args.device)
        print(f"WER: {wer_score}%")

    # Also read RTF if available
    rtf_path = os.path.join(args.gen_dir, "rtf.txt")
    rtf = None
    if os.path.exists(rtf_path):
        rtf = float(open(rtf_path).read().strip())
        print(f"RTF: {rtf:.4f}")

    # Print summary for parsing
    print(f"\n=== METRICS ===")
    if rtf is not None:
        print(f"RTF: {rtf:.4f}")
    if wer_score is not None:
        print(f"WER: {wer_score}%")
    if sim_score is not None:
        print(f"SIM-o: {sim_score}")


if __name__ == "__main__":
    main()
