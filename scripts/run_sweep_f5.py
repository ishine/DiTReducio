"""F5-TTS threshold sweep (T0-T6) — calibrate + infer + eval per threshold.

Usage:
    python scripts/run_sweep_f5.py \
        --backend_root /path/to/F5-TTS \
        --f5tts_ckpt /path/to/model.safetensors \
        --vocoder_path /path/to/vocos-mel-24khz \
        --data_root /path/to/LibriSpeech \
        [--deltas 0 0.05 0.1 0.15 0.2 0.25 0.3]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NFE_STEP = 32
SEED = 888


def _run(
    cmd: list[str], cwd: str | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def _env_with_backend(backend_root: str) -> dict:
    env = os.environ.copy()
    paths = [
        backend_root,
        os.path.join(backend_root, "src"),
        os.path.join(PROJECT, "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(paths) + (f":{existing}" if existing else "")
    env.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
    return env


def _fast_cli_args(
    delta: float,
    calibration_mode: bool,
    output_dir: str,
    methods_path: str,
    args,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        PYTHON,
        os.path.join(
            PROJECT,
            "src/ditreducio/backends/f5tts/cli.py",
        ),
        "--model",
        "F5TTS_v1_Base",
        "--model_cfg",
        args.model_cfg,
        "--ckpt_file",
        args.f5tts_ckpt,
        "--ref_audio",
        args.ref_audio,
        "--ref_text",
        "Some call me nature, others call me mother nature.",
        "--gen_text",
        "The quick brown fox jumps over the lazy dog.",
        "--output_dir",
        output_dir,
        "--nfe_step",
        str(NFE_STEP),
        "--cfg_strength",
        "2.0",
        "--sway_sampling_coef",
        "-1.0",
        "--speed",
        "1.0",
        "--device",
        "cuda",
        "--seed",
        str(SEED),
        "--methods_path",
        methods_path,
        "--load_vocoder_from_local",
        "--vocoder_local_path",
        args.vocoder_path,
        "--bs-mode",
        "residual",
    ]
    if calibration_mode:
        cmd.append("-q")
    if delta > 0:
        cmd.extend(["-d", str(delta)])
    if extra:
        cmd.extend(extra)
    return cmd


def run_sweep(deltas: list[float], output_base: str, dataset: str, args):
    """Run calibrate+infer for each delta, then eval."""
    lst_file = args.test_clean_lst if dataset == "clean" else args.test_other_lst
    libri_root = args.librispeech_clean if dataset == "clean" else args.librispeech_other

    results = []
    for delta in deltas:
        tag = f"d{delta:.2f}"
        out_dir = os.path.join(output_base, tag)
        methods_dir = os.path.join(output_base, "strategies", tag)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(methods_dir, exist_ok=True)

        # ── calibrate ───────────────────────────────────────────────────
        if delta > 0:
            print(f"\n{'=' * 60}")
            print(f"[SWEEP] Calibrating delta={delta} ...")
            cmd = _fast_cli_args(
                delta,
                calibration_mode=True,
                output_dir=out_dir,
                methods_path=methods_dir,
                args=args,
            )
            r = _run(cmd, cwd=methods_dir, env=_env_with_backend(args.backend_root))
            if r.returncode != 0:
                print(f"[ERR] calibrate delta={delta} failed: {r.stderr[-500:]}")
                continue
            print(f"[OK] calibrate delta={delta}")

        # ── batch infer (via eval_infer script) ─────────────────────────
        print(f"\n[SWEEP] Running batch inference delta={delta} on {dataset} ...")
        infer_cmd = [
            PYTHON,
            os.path.join(PROJECT, "scripts/eval_infer_f5.py"),
            "--delta",
            str(delta),
            "--lst_file",
            lst_file,
            "--librispeech_root",
            libri_root,
            "--output_dir",
            out_dir,
            "--methods_path",
            methods_dir,
            "--nfe_step",
            str(NFE_STEP),
            "--device",
            "cuda",
            "--seed",
            str(SEED),
        ]
        r = _run(infer_cmd, env=_env_with_backend(args.backend_root))
        if r.returncode != 0:
            print(f"[ERR] infer delta={delta} failed: {r.stderr[-500:]}")
            # try to continue with eval anyway
        print(r.stdout[-300:] if r.stdout else "(no stdout)")

        # ── eval ────────────────────────────────────────────────────────
        print(f"\n[SWEEP] Evaluating delta={delta} on {dataset} ...")
        eval_cmd = [
            PYTHON,
            os.path.join(PROJECT, "scripts/eval_metrics.py"),
            "--gen_dir",
            out_dir,
            "--lst_file",
            lst_file,
            "--librispeech_root",
            libri_root,
            "--device",
            "cuda",
        ]
        if args.whisper_ckpt:
            eval_cmd.extend(["--whisper_ckpt", args.whisper_ckpt])
        if args.ecapa_ckpt:
            eval_cmd.extend(["--ecapa_ckpt", args.ecapa_ckpt])
        r = _run(eval_cmd, env=_env_with_backend(args.backend_root))
        if r.returncode != 0:
            print(f"[ERR] eval delta={delta} failed: {r.stderr[-500:]}")
            continue

        # parse metrics from eval output
        metrics = _parse_eval_output(r.stdout)
        metrics["delta"] = delta
        metrics["dataset"] = dataset
        results.append(metrics)
        print(
            f"[OK] delta={delta}: RTF={metrics.get('rtf', '?')} WER={metrics.get('wer', '?')} SIM={metrics.get('sim', '?')}"
        )

    return results


def _parse_eval_output(stdout: str) -> dict:
    """Parse key metrics from eval_metrics.py output."""
    m = {}
    for line in stdout.splitlines():
        if line.startswith("RTF:"):
            m["rtf"] = float(line.split(":")[1].strip())
        elif line.startswith("WER:"):
            m["wer"] = float(line.split(":")[1].strip().replace("%", ""))
        elif line.startswith("SIM-o:"):
            m["sim"] = float(line.split(":")[1].strip())
    return m


def save_csv(results: list[dict], path: str):
    if not results:
        return
    fieldnames = ["dataset", "delta", "rtf", "wer", "sim"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"[CSV] saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="F5-TTS threshold sweep (T0-T6)")
    parser.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS)
    parser.add_argument(
        "--output_base", default=os.path.join(PROJECT, "outputs/sweep_f5")
    )
    parser.add_argument("--dataset", choices=["clean", "other", "both"], default="both")
    parser.add_argument("--csv_dir", default=os.path.join(PROJECT, "outputs/metrics"))

    # Path arguments (required or inferred from backend_root)
    parser.add_argument(
        "--backend_root", required=True,
        help="Path to F5-TTS code root",
    )
    parser.add_argument(
        "--f5tts_ckpt", required=True,
        help="Path to F5-TTS model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--vocoder_path", required=True,
        help="Path to vocoder checkpoint directory",
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Path to LibriSpeech data root containing test-clean/ and cross-sentence .lst files",
    )
    parser.add_argument(
        "--model_cfg", default=None,
        help="Path to F5TTS model config YAML (default: <backend_root>/src/f5_tts/configs/F5TTS_v1_Base.yaml)",
    )
    parser.add_argument(
        "--ref_audio", default=None,
        help="Path to reference audio for calibration (default: <backend_root>/src/f5_tts/infer/examples/basic/basic_ref_en.wav)",
    )
    parser.add_argument(
        "--whisper_ckpt", default="",
        help="Path to local faster-whisper model dir (empty = auto-download)",
    )
    parser.add_argument(
        "--ecapa_ckpt", default="",
        help="Path to local ECAPA-TDNN checkpoint .pth file (empty = auto-download)",
    )
    args = parser.parse_args()

    # Derive defaults from backend_root
    if not args.model_cfg:
        args.model_cfg = os.path.join(
            args.backend_root, "src/f5_tts/configs/F5TTS_v1_Base.yaml"
        )
    if not args.ref_audio:
        args.ref_audio = os.path.join(
            args.backend_root, "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
        )

    # Derive LibriSpeech paths from data_root
    args.test_clean_lst = os.path.join(args.data_root, "LibriSpeech/test_clean_cross_sentence.lst")
    args.test_other_lst = os.path.join(args.data_root, "LibriSpeech-other/test_other_cross_sentence.lst")
    args.librispeech_clean = os.path.join(args.data_root, "LibriSpeech/test-clean")
    args.librispeech_other = os.path.join(args.data_root, "LibriSpeech-other/test-other")

    os.makedirs(args.csv_dir, exist_ok=True)
    all_results = []

    datasets = ["clean", "other"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        r = run_sweep(args.deltas, args.output_base, ds, args)
        all_results.extend(r)
        save_csv(r, os.path.join(args.csv_dir, f"main_table_f5_{ds}.csv"))

    save_csv(all_results, os.path.join(args.csv_dir, "main_table_f5.csv"))
    print("\n[DONE] Sweep complete.")


if __name__ == "__main__":
    main()
