#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import soundfile as sf
import torch

from vllm.engine.arg_utils import nullable_str
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav

logger = init_logger('vllm.omni')

parser = argparse.ArgumentParser()
parser.add_argument('--code2wav-model',
                    type=str,
                    default=os.path.expanduser("~/models/omni-v4/code2wav"))
parser.add_argument('--input-json',
                    type=str,
                    default=os.path.expanduser("~/vllm/generated-codes.json"))
parser.add_argument('--voice-type', type=nullable_str, default='default')
parser.add_argument("--batched-chunk", type=int, default=None)
parser.add_argument("--frequency", type=str, default='50hz', choices=['50hz'])
parser.add_argument('--sample-rate', type=int, default=24000)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--concurrency', type=int, default=1)
parser.add_argument('--enable-torch-compile', action='store_true')
parser.add_argument('--enable-torch-compile-first-chunk', action='store_true')
parser.add_argument("--odeint-method",
                    type=str,
                    default="rk4",
                    choices=["euler", "rk4"])
parser.add_argument('--multi-waveforms', action='store_true')
parser.add_argument('--output-dir',
                    type=str,
                    default='.',
                    help="Audio output directory")

args = parser.parse_args()


def process_code(
    code: List[int],
    code2wav: Qwen2Code2wav,
    code2wav_cond: torch.Tensor,
    code2wav_ref_mel: torch.Tensor,
    code2wav_y_all: torch.Tensor,
    code2wav_steps: int,
    device: torch.device,
) -> List[np.ndarray]:
    # start the code2wav thread
    all_code = torch.tensor(code, dtype=torch.long, device=device).reshape(1, -1)
    progress, prev_generated, waveforms = 0, None, []
    for i in range(all_code.size(1)):
        finished = i == all_code.size(1) - 1
        chunk_code_length = i * (2 if args.frequency == "50hz" else
                                 4) - code2wav.future_cache_size
        code = all_code[:, :i+1]
        if (chunk_code_length > 0
                and chunk_code_length % code2wav.chunk_size == 0) or finished:
            logger.info("process_chunk | %d | codec_embed_size: %d | code2wav.future_cache_size: %d | chunk_code_length: %d | code2wav.chunk_size: %d | progress: %d | finished: %r | code.shape: %r",
                        i, code2wav.codec_embed_size, code2wav.future_cache_size, chunk_code_length, code2wav.chunk_size, progress, finished, code.shape)
            if progress == 0 and finished:
                process_chunk = code2wav.process_little_chunk
            else:
                process_chunk = code2wav.process_chunk

            start_chunk_time = time.perf_counter()

            prev_generated, audio = process_chunk(
                code2wav_cond,
                code2wav_ref_mel,
                codec_all=code,
                y_all=code2wav_y_all,
                i=progress,
                steps=code2wav_steps,
                prev_generated=prev_generated,
                finished=finished,
            )
            end_chunk_time = time.perf_counter()
            print(
                f'Chunk {progress} took {end_chunk_time - start_chunk_time} seconds'
            )
            progress += 1
            waveforms.append(audio)
    return [waveform.detach().cpu().numpy() for waveform in waveforms]

def load_code2wav(model_path):
    dit_model, bigvgan_model = {}, {}
    safetensors = sorted(
        glob.glob(os.path.join(model_path, '*.safetensors')))
    legacy_weights = False
    for key, value in safetensors_weights_iterator(safetensors,
                                                   use_tqdm_on_load=True):
        legacy_weights = legacy_weights or 'input_embed.spk_encoder.fc.conv.weight' in key
        if legacy_weights:
            break
    for key, value in safetensors_weights_iterator(safetensors,
                                                   use_tqdm_on_load=True):
        if key.startswith('token2wav.code2wav_bigvgan_model.'):
            if 'generator' not in bigvgan_model:
                bigvgan_model['generator'] = {}
            bigvgan_model['generator'][key.replace(
                'token2wav.code2wav_bigvgan_model.', '')] = value
        if key.startswith('token2wav.code2wav_dit_model.'):
            key = key.replace('token2wav.code2wav_dit_model.',
                              'transformer.')
            if key.startswith('transformer.input_embed.spk_encoder'):
                if legacy_weights:
                    dit_model[key] = value
                else:
                    dit_model[key.replace('.bias', '.conv.bias').replace(
                        '.weight', '.conv.weight')] = value
            elif '.ff.ff.0.weight' in key or '.ff.ff.0.bias' in key:
                dit_model[key.replace('.ff.ff.0.weight',
                                      '.ff.ff.0.0.weight').replace(
                                          '.ff.ff.0.bias',
                                          '.ff.ff.0.0.bias')] = value
            elif '.ff.ff.3.weight' in key or '.ff.ff.3.bias' in key:
                dit_model[key.replace('.ff.ff.3.weight',
                                      '.ff.ff.2.weight').replace(
                                          '.ff.ff.3.bias',
                                          '.ff.ff.2.bias')] = value
            else:
                dit_model[key] = value
    return dit_model, bigvgan_model

def load_spk_dict(model_path, device):
    code2wav_conds, code2wav_ref_mels = {}, {}

    if not os.path.exists(os.path.join(model_path, 'spk_dict.pt')):
        return code2wav_conds, code2wav_ref_mels

    for key, value in torch.load(os.path.join(model_path,
                                              'spk_dict.pt')).items():
        code2wav_conds[key] = value["cond"].to(device)
        code2wav_ref_mels[key] = value["ref_mel"].to(device)
    return code2wav_conds, code2wav_ref_mels

def main():
    # code2wav model
    model_path = args.code2wav_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dit_model, bigvgan_model = load_code2wav(model_path)

    code2wav_conds, code2wav_ref_mels = load_spk_dict(model_path, device)
    print(f"speakers: {code2wav_conds.keys()}")
    if "default" not in code2wav_conds:
        code2wav_conds["default"] = code2wav_conds[list(code2wav_conds.keys())[0]]
    if "default" not in code2wav_ref_mels:
        code2wav_ref_mels["default"] = code2wav_ref_mels[list(code2wav_ref_mels.keys())[0]]
    if args.voice_type not in code2wav_conds:
        print(f"voice type {args.voice_type} not found, using default")
        args.voice_type = "default"
    code2wav_cond = code2wav_conds[args.voice_type]
    code2wav_ref_mel = code2wav_ref_mels[args.voice_type]
    print(f"code2wav_cond {code2wav_cond.shape}")
    print(f"code2wav_ref_mel {code2wav_ref_mel.shape}")

    if args.batched_chunk is None:
        if args.frequency == "50hz":
            args.batched_chunk = 2
        else:
            args.batched_chunk = 1
    args.frequency = args.frequency

    code2wav_steps: int = 10
    code2wav_bs_mel: int = 24 if args.frequency == "50hz" else 32
    code2wav = Qwen2Code2wav(dit_ckpt=dit_model,
                             bigvgan_ckpt=bigvgan_model,
                             steps=code2wav_steps,
                             bs_mel=code2wav_bs_mel,
                             odeint_method=args.odeint_method,
                             batched_chunk=args.batched_chunk,
                             frequency=args.frequency,
                             device=device,
                             with_weight_norm=False)

    if args.enable_torch_compile:
        code2wav.enable_torch_compile(args.enable_torch_compile_first_chunk)

    # read the inputs
    with open(args.input_json) as f:
        code = json.load(f)

    code2wav_y_all = torch.randn(args.concurrency,
                                 1,
                                 32768,
                                 80,
                                 device=device,
                                 dtype=code2wav_cond.dtype)
    print(f"code2wav_y_all shape: {code2wav_y_all.shape}, type: {code2wav_y_all.dtype}")

    # warmup
    start_time = time.perf_counter()
    for _ in range(args.warmup):
        process_code(
            code,
            code2wav,
            code2wav_cond,
            code2wav_ref_mel,
            code2wav_y_all[0],
            code2wav_steps,
            device,
        )
    print(f"Code2wav warmup {args.warmup} times "
          f"took {time.perf_counter() - start_time} seconds "
          f"for {len(code)} tokens")

    # concurrency
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for i in range(args.concurrency):
            futures.append(
                executor.submit(
                    process_code,
                    code,
                    code2wav,
                    code2wav_cond,
                    code2wav_ref_mel,
                    code2wav_y_all[i],
                    code2wav_steps,
                    device,
                ))

        waveforms = []
        for future in futures:
            waveforms.append(future.result())
        waveforms = waveforms[0]

    end_time = time.perf_counter()
    cost_time = end_time - start_time
    print(f"Code2wav for {args.concurrency} times "
          f"took {cost_time} seconds "
          f"for {len(code)} tokens, {len(waveforms)} waveforms")

    tmp_wav_path = os.path.join(args.output_dir, "code2wav.wav")
    print(f'Writting waveforms to {tmp_wav_path}')
    if args.multi_waveforms:
        for i, waveform in enumerate(waveforms):
            wav_path =f"{tmp_wav_path[:-4]}-{i}.wav" 
            sf.write(wav_path, waveform, samplerate=args.sample_rate)
            sf.info(wav_path,verbose=True)

    sf.write(tmp_wav_path,
                 np.concatenate(waveforms),
                 samplerate=args.sample_rate)
    info = sf.info(tmp_wav_path,verbose=True)
    if args.concurrency == 1:
        print(f"wav duration: {info.duration} s | cost: {cost_time} s | RTF: {cost_time/info.duration}")

    end_write_time = time.perf_counter()
    print(f'Writing waveforms took {end_write_time - end_time} seconds')


if __name__ == '__main__':
    main()
