# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import argparse
import sys
import librosa
import numpy as np
import torch
import math
import warnings
import logging
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from langdetect import detect as classify_language
from pydub import AudioSegment
import pyloudnorm as pyln

from ditreducio.backends.megatts3.hooks import make_view, pre_calibration_hook
from ditreducio.calibration.hooks import (
    build_need_cache_output,
    calibration,
    calibration_preparation,
    calibration_reset,
    calibration_reset_step,
    pre_calibration,
    pre_calibration_check,
    speedup,
)
from ditreducio.calibration.util import threshold_q, seed_everything
from ditreducio.backends.megatts3.flops_tracker import (
    start_flops_tracking,
    end_flops_tracking,
)

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import (
    g2p,
    align,
    make_dur_prompt,
    dur_pred,
    prepare_inputs_for_dit,
)
from tts.utils.audio_utils.io import (
    save_wav,
    to_wav_bytes,
    convert_to_wav_bytes,
    combine_audio_segments,
)
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import split_text_chinesev2, chunk_text_english
from tts.utils.commons.hparams import hparams, set_hparams


if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def cut_wav(wav_path, max_len=28):
    audio = AudioSegment.from_file(wav_path)
    audio = audio[: int(max_len * 1000)]
    audio.export(wav_path, format="wav")


class MegaTTS3DiTInfer:
    def __init__(
        self,
        device=None,
        ckpt_root="./checkpoints",
        dit_exp_name="diffusion_transformer",
        frontend_exp_name="aligner_lm",
        wavvae_exp_name="wavvae",
        dur_ckpt_path="duration_lm",
        g2p_exp_name="g2p",
        precision=torch.float16,
        **kwargs,
    ):
        self.sr = 24000
        self.fm = 8
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.precision = precision

        # build models
        self.dit_exp_name = os.path.join(ckpt_root, dit_exp_name)
        self.frontend_exp_name = os.path.join(ckpt_root, frontend_exp_name)
        self.wavvae_exp_name = os.path.join(ckpt_root, wavvae_exp_name)
        self.dur_exp_name = os.path.join(ckpt_root, dur_ckpt_path)
        self.g2p_exp_name = os.path.join(ckpt_root, g2p_exp_name)
        self.build_model(self.device)

        # init text normalizer
        self.zh_normalizer = ZhNormalizer(
            overwrite_cache=False, remove_erhua=False, remove_interjections=False
        )
        self.en_normalizer = EnNormalizer(overwrite_cache=False)
        # loudness meter
        self.loudness_meter = pyln.Meter(self.sr)

    def build_model(self, device):
        set_hparams(exp_name=self.dit_exp_name, print_hparams=False)

        """ Load Dict """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dict_candidates = [
            Path(current_dir) / "utils" / "text_utils" / "dict.json",
            Path(current_dir).parent / "utils" / "text_utils" / "dict.json",
        ]
        tts_module = sys.modules["tts"]
        tts_file = getattr(tts_module, "__file__", None)
        if tts_file:
            tts_root = Path(tts_file).resolve().parent
        else:
            tts_root = Path(next(iter(tts_module.__path__))).resolve()
        dict_candidates.append(tts_root / "utils" / "text_utils" / "dict.json")

        dict_path = None
        for candidate in dict_candidates:
            if candidate.exists():
                dict_path = candidate
                break
        if dict_path is None:
            tried = "\n".join(str(path) for path in dict_candidates)
            raise FileNotFoundError(f"dict.json not found. Tried:\n{tried}")
        ling_dict = json.load(open(dict_path, encoding="utf-8-sig"))
        self.ling_dict = {
            k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov="<UNK>")
            for k in ["phone", "tone"]
        }
        self.token_encoder = token_encoder = self.ling_dict["phone"]
        ph_dict_size = len(token_encoder)

        """ Load Duration LM """
        from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor

        hp_dur_model = self.hp_dur_model = set_hparams(
            f"{self.dur_exp_name}/config.yaml", global_hparams=False
        )
        hp_dur_model["frames_multiple"] = hparams["frames_multiple"]
        self.dur_model = ARDurPredictor(
            hp_dur_model,
            hp_dur_model["dur_txt_hs"],
            hp_dur_model["dur_model_hidden_size"],
            hp_dur_model["dur_model_layers"],
            ph_dict_size,
            hp_dur_model["dur_code_size"],
            use_rot_embed=hp_dur_model.get("use_rot_embed", False),
        )
        self.length_regulator = LengthRegulator()
        load_ckpt(self.dur_model, f"{self.dur_exp_name}", "dur_model")
        self.dur_model.eval()
        self.dur_model.to(device)

        """ Load Diffusion Transformer """
        from tts.modules.llm_dit.dit import Diffusion

        self.dit = Diffusion()
        load_ckpt(self.dit, f"{self.dit_exp_name}", "dit", strict=False)
        self.dit.eval()
        self.dit.to(device)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1

        """ Load Frontend LM """
        from tts.modules.aligner.whisper_small import Whisper

        self.aligner_lm = Whisper()
        load_ckpt(self.aligner_lm, f"{self.frontend_exp_name}", "model")
        self.aligner_lm.eval()
        self.aligner_lm.to(device)
        self.kv_cache = None
        self.hooks = None

        """ Load G2P LM"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        g2p_tokenizer = AutoTokenizer.from_pretrained(
            self.g2p_exp_name, padding_side="right"
        )
        g2p_tokenizer.padding_side = "right"
        self.g2p_model = (
            AutoModelForCausalLM.from_pretrained(self.g2p_exp_name).eval().to(device)
        )
        self.g2p_tokenizer = g2p_tokenizer
        self.speech_start_idx = g2p_tokenizer.encode("<Reserved_TTS_0>")[0]

        """ Wav VAE """
        self.hp_wavvae = hp_wavvae = set_hparams(
            f"{self.wavvae_exp_name}/config.yaml", global_hparams=False
        )
        from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3

        self.wavvae = WavVAE_V3(hparams=hp_wavvae)
        if os.path.exists(f"{self.wavvae_exp_name}/model_only_last.ckpt"):
            load_ckpt(
                self.wavvae,
                f"{self.wavvae_exp_name}/model_only_last.ckpt",
                "model_gen",
                strict=True,
            )
            self.has_vae_encoder = True
        else:
            load_ckpt(
                self.wavvae,
                f"{self.wavvae_exp_name}/decoder.ckpt",
                "model_gen",
                strict=False,
            )
            self.has_vae_encoder = False
        self.wavvae.eval()
        self.wavvae.to(device)
        self.vae_stride = hp_wavvae.get("vae_stride", 4)
        self.hop_size = hp_wavvae.get("hop_size", 4)

    def preprocess(self, audio_bytes, latent_file=None, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        """ Load wav """
        wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
        # Pad wav if necessary
        ws = hparams["win_size"]
        if len(wav) % ws < ws - 1:
            wav = np.pad(
                wav, (0, ws - 1 - (len(wav) % ws)), mode="constant", constant_values=0.0
            ).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode="constant", constant_values=0.0).astype(
            np.float32
        )
        self.loudness_prompt = self.loudness_meter.integrated_loudness(
            wav.astype(float)
        )

        """ obtain alignments with aligner_lm """
        ph_ref, tone_ref, mel2ph_ref = align(self, wav)

        with torch.inference_mode():
            """ Forward WaveVAE to obtain: prompt latent """
            if self.has_vae_encoder:
                wav = torch.FloatTensor(wav)[None].to(self.device)
                vae_latent = self.wavvae.encode_latent(wav)
                vae_latent = vae_latent[:, : mel2ph_ref.size(1) // 4]
            else:
                assert latent_file is not None, (
                    "Please provide latent_file in WaveVAE decoder-only mode"
                )
                vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                vae_latent = vae_latent[:, : mel2ph_ref.size(1) // 4]

            """ Duration Prompting """
            self.dur_model.hparams["infer_top_k"] = topk_dur if topk_dur > 1 else None
            incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(
                self, mel2ph_ref, ph_ref, tone_ref
            )

        return {
            "ph_ref": ph_ref,
            "tone_ref": tone_ref,
            "mel2ph_ref": mel2ph_ref,
            "vae_latent": vae_latent,
            "incremental_state_dur_prompt": incremental_state_dur_prompt,
            "ctx_dur_tokens": ctx_dur_tokens,
        }

    def forward(
        self,
        resource_context,
        input_text,
        time_step,
        p_w,
        t_w,
        dur_disturb=0.1,
        dur_alpha=1.0,
        **kwargs,
    ):
        device = self.device

        ph_ref = resource_context["ph_ref"].to(device)
        tone_ref = resource_context["tone_ref"].to(device)
        mel2ph_ref = resource_context["mel2ph_ref"].to(device)
        vae_latent = resource_context["vae_latent"].to(device)
        ctx_dur_tokens = resource_context["ctx_dur_tokens"].to(device)
        incremental_state_dur_prompt = resource_context["incremental_state_dur_prompt"]

        with torch.inference_mode():
            """ Generating """
            wav_pred_ = []
            language_type = classify_language(input_text)
            if language_type == "en":
                input_text = self.en_normalizer.normalize(input_text)
                text_segs = chunk_text_english(input_text, max_chars=130)
            else:
                input_text = self.zh_normalizer.normalize(input_text)
                text_segs = split_text_chinesev2(input_text, limit=60)

            for seg_i, text in enumerate(text_segs):
                """ G2P """
                ph_pred, tone_pred = g2p(self, text)

                """ Duration Prediction """
                mel2ph_pred = dur_pred(
                    self,
                    ctx_dur_tokens,
                    incremental_state_dur_prompt,
                    ph_pred,
                    tone_pred,
                    seg_i,
                    dur_disturb,
                    dur_alpha,
                    is_first=seg_i == 0,
                    is_final=seg_i == len(text_segs) - 1,
                )

                inputs = prepare_inputs_for_dit(
                    self,
                    mel2ph_ref,
                    mel2ph_pred,
                    ph_ref,
                    tone_ref,
                    ph_pred,
                    tone_pred,
                    vae_latent,
                )
                # Speech dit inference
                with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                    x = self.dit.inference(
                        inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]
                    ).float()

                # WavVAE decode
                x[:, : vae_latent.size(1)] = vae_latent
                wav_pred = self.wavvae.decode(x)[0, 0].to(torch.float32)

                """ Post-processing """
                # Trim prompt wav
                wav_pred = (
                    wav_pred[vae_latent.size(1) * self.vae_stride * self.hop_size :]
                    .cpu()
                    .numpy()
                )
                # Norm generated wav to prompt wav's level
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                loudness_pred = self.loudness_meter.integrated_loudness(
                    wav_pred.astype(float)
                )
                wav_pred = pyln.normalize.loudness(
                    wav_pred, loudness_pred, self.loudness_prompt
                )
                if np.abs(wav_pred).max() >= 1:
                    wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95

                # Apply hamming window
                wav_pred_.append(wav_pred)

            wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(float)
            return to_wav_bytes(wav_pred, self.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav", type=str)
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--time_step",
        type=int,
        default=32,
        help="Inference steps of Diffusion Transformer",
    )
    parser.add_argument("--p_w", type=float, default=1.6, help="Intelligibility Weight")
    parser.add_argument("--t_w", type=float, default=2.5, help="Similarity Weight")
    parser.add_argument(
        "--delta", "-d", type=float, default=None, help="Compression Threshold"
    )
    parser.add_argument("-q", action="store_true", help="Calibration Mode")
    parser.add_argument("--seed", type=int, default=888, help="Random seed")
    parser.add_argument(
        "--methods_path",
        type=str,
        default="methods",
        help="Not use precheck and precalibration",
    )
    parser.add_argument(
        "--track_flops",
        action="store_true",
        help="Track and display real-time FLOPs during inference",
    )
    # for ablation
    parser.add_argument(
        "--nots",
        type=bool,
        default=False,
        help="Not use ts",
    )

    parser.add_argument(
        "--nobs",
        type=bool,
        default=False,
        help="Not use bs",
    )

    parser.add_argument(
        "--nopre",
        type=bool,
        default=False,
        help="Not use precheck and precalibration",
    )
    parser.add_argument(
        "--bs-mode",
        type=str,
        default="residual",
        choices=["residual", "cond_replace", "uncond_replace"],
        help="Branch skipping mode",
    )
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    (
        wav_path,
        input_text,
        out_path,
        time_step,
        p_w,
        t_w,
        delta,
        calibration_mode,
        track_flops,
    ) = (
        args.input_wav,
        args.input_text,
        args.output_dir,
        args.time_step,
        args.p_w,
        args.t_w,
        args.delta,
        args.q,
        args.track_flops,
    )

    delta = None if delta == 0 else delta
    methods_root = args.methods_path

    infer_ins = MegaTTS3DiTInfer()
    infer_ins.dit.encoder.bs_mode = args.bs_mode

    # Build view for shared calibration API
    view = make_view(infer_ins)

    # for ablation
    nots, nobs, nopre = args.nots, args.nobs, args.nopre
    if nots:
        nopre = True
    infer_ins.dit.encoder.nots = nots
    infer_ins.dit.encoder.nobs = nobs
    infer_ins.dit.encoder.nopre = nopre
    print(f"not use ts: {nots}, not use obs: {nobs}, not use precheck: {nopre}")

    with open(wav_path, "rb") as file:
        file_content = file.read()

    latent_file = args.latent_file
    if latent_file is None:
        latent_candidate = os.path.splitext(wav_path)[0] + ".npy"
        if os.path.exists(latent_candidate):
            latent_file = latent_candidate

    print(f"[fast_cli - main]:Wav Path:{wav_path}\n Input Text:{input_text}")
    resource_context = infer_ins.preprocess(file_content, latent_file=latent_file)

    calibrate_hook = None
    if calibration_mode:
        # Pre-calibration
        pre_hooks = pre_calibration_check(view, pre_calibration_hook, steps=time_step)
        # First run
        _ = infer_ins.forward(
            resource_context, input_text, time_step=time_step, p_w=p_w, t_w=t_w
        )

        # Remove hooks
        for hook in pre_hooks:
            hook.remove()
        # Collect similarities for each timestep and block
        similarities = []
        for blocki in range(len(infer_ins.dit.encoder.layers)):
            attn = infer_ins.dit.encoder.layers[blocki].attention
            similarities_list_i = list(attn.diagonal_similarities.values())
            similarities.extend(similarities_list_i)

        similarities = torch.tensor(similarities, device="cpu").numpy()

        # Set threshold
        threshold = threshold_q(similarities, ratio=0.1) if not nopre else 1.2

        print(f"[fast_cli - main] Pre Calibration Percentile Threshold: {threshold}")

        ts_first_cnt = 0

        for blocki in range(len(infer_ins.dit.encoder.layers)):
            attn = infer_ins.dit.encoder.layers[blocki].attention
            attn.ts_first = {}
            for step, similarity in attn.diagonal_similarities.items():
                attn.ts_first[step] = False
                if similarity >= threshold:
                    attn.ts_first[step] = True
                    ts_first_cnt += 1
            del attn.diagonal_similarities

        print(f"TS First Count: {ts_first_cnt}")

        # Start pre-calibration
        pre_calibrate_hook = pre_calibration(
            view, steps=time_step, threshold=delta
        )
        # Pre-calibrate
        _ = infer_ins.forward(
            resource_context, input_text, time_step=time_step, p_w=p_w, t_w=t_w
        )
        pre_calibrate_hook.remove()
        # If in calibration mode, call calibrate method
        calibrate_hook = calibration(view, steps=time_step, threshold=delta)
    elif calibration_mode is False and delta is not None:
        speedup(view, steps=time_step, delta=delta, methods_path=methods_root)
    else:
        calibration_preparation(view, steps=time_step)
    # Start FLOPs tracking if enabled
    if track_flops:
        flops_tracker = start_flops_tracking(infer_ins)
        print("FLOPs tracking enabled...")

    # Count TS occurrences
    ts_cnt = 0
    for blocki, block in enumerate(infer_ins.dit.encoder.layers):
        for method in block.attention.steps_method:
            if method == "TS":
                ts_cnt += 1

    print(f"TS Count: {ts_cnt}")

    # Hooks for calculation
    hooks = []
    if calibrate_hook is not None:
        hooks.append(calibrate_hook)

    wav_bytes = infer_ins.forward(
        resource_context, input_text, time_step=time_step, p_w=p_w, t_w=t_w
    )

    # End FLOPs tracking and display results
    if track_flops:
        flops_summary = end_flops_tracking()
        from ditreducio.backends.megatts3.flops_tracker import (
            get_flops_tracker,
        )

        tracker = get_flops_tracker()
        tracker.print_summary()

    for hook in hooks:
        hook.remove()

    if calibration_mode:
        transformer = infer_ins.dit.encoder

        # Save calibrated methods
        to_save_methods = {"methods": []}
        for blocki, block in enumerate(transformer.layers):
            to_save_methods["methods"].append(block.attention.steps_method)
        to_save_methods["need_cached_output"] = build_need_cache_output(
            to_save_methods["methods"]
        )
        os.makedirs(methods_root, exist_ok=True)
        methods_path = f"{methods_root}/{time_step}_{delta}.json"
        with open(methods_path, "w") as file:
            import json

            file.write(json.dumps(to_save_methods))

    sys.stdout = sys.__stdout__
    print(f"| Saving results to {out_path}/[P]{input_text[:20]}.wav")
    os.makedirs(out_path, exist_ok=True)
    save_wav(wav_bytes, f"{out_path}/out_{delta}.wav")
