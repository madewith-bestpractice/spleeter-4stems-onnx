"""Spleeter 4stems via ONNX: drums, bass, vocals, other.

Same pipeline as separate.py, four U-Nets instead of two. The models are ours --
sherpa-onnx only ever published 2stems, because their U-Net hardcodes
LeakyReLU/ReLU and Spleeter's 4stems config asks for ELU. See convert/.
"""

import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

import stft as S
from separate import EPS, EXPONENT, F, T, analyse, extend

STEMS = ("vocals", "drums", "bass", "other")
MODEL_DIR = "convert/4stems"


def separate4(wave, mode="average", quiet=False):
    n = len(wave)
    spec, net_in, frames = analyse(wave)

    estimates, infer = {}, 0.0
    for name in STEMS:
        sess = ort.InferenceSession(
            f"{MODEL_DIR}/{name}.fp16.onnx", providers=["CPUExecutionProvider"]
        )
        t0 = time.perf_counter()
        out = sess.run(["y"], {"x": net_in})[0]
        dt = time.perf_counter() - t0
        infer += dt
        if not quiet:
            print(f"    {name:<8} {dt:5.2f}s")
        estimates[name] = out.reshape(out.shape[0], -1, F)[:, :frames]

    total = sum(e ** EXPONENT for e in estimates.values()) + EPS
    out = {}
    for name, e in estimates.items():
        mask = (e ** EXPONENT + EPS / len(estimates)) / total
        masked = spec * extend(mask, mode)
        out[name] = np.stack(
            [S.istft(masked[c], n) for c in range(masked.shape[0])], axis=-1
        )
    return out, infer


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "average"
    wave, sr = sf.read("atomic.wav", dtype="float64")
    print(f"in: {wave.shape} @ {sr} Hz ({len(wave)/sr:.1f}s)  mask_extension={mode}\n")

    t0 = time.perf_counter()
    stems, infer = separate4(wave, mode)
    wall = time.perf_counter() - t0

    print()
    for name, y in stems.items():
        sf.write(f"out4_{name}.wav", y, sr)
        print(f"  {name:<8} peak {np.abs(y).max():.3f} -> out4_{name}.wav")

    # Four masks that sum to 1 must still reconstruct the mix exactly.
    err = np.abs(sum(stems.values()) - wave).max()
    res = 10 * np.log10(
        np.mean((sum(stems.values()) - wave) ** 2) / np.mean(wave ** 2) + 1e-20
    )
    print(f"\n  sum(4 stems) vs mix: max err {err:.2e}   residual {res:.1f} dB")

    audio = len(wave) / sr
    print(f"  inference {infer:.2f}s  total {wall:.2f}s  for {audio:.0f}s "
          f"-> {audio/infer:.0f}x realtime (inference)")


if __name__ == "__main__":
    main()
