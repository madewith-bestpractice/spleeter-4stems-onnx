"""Spleeter 2stems via ONNX, with the STFT outside the graph.

This is the spike for Sampeltu's stem feature. It exists to answer four things
no source could tell us: does the export run, what does the mask maths have to
be, does it sound right, and how fast is it.

The model is sherpa-onnx's conversion of Deezer's Spleeter (k2-fsa/sherpa-onnx,
tag source-separation-models). Two independent U-Nets -- one per stem -- each
taking [channels, splits, T=512, F=1024] magnitude and returning an estimate of
that stem's magnitude. No complex numbers cross the graph boundary, which is
the whole reason this ports cleanly and Demucs does not.
"""

import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

import stft as S

T = 512          # frames per split, fixed by the export
F = 1024         # bins the net models: 1024 of 2049, i.e. up to ~11 kHz
EPS = 1e-10
EXPONENT = 2     # Spleeter's separation_exponent
MODEL_DIR = "sherpa-onnx-spleeter-2stems-fp16"
STEMS = ("vocals", "accompaniment")


def analyse(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """[n, 2] -> (complex stft [2, frames, 2049], net input [2, splits, T, F])."""
    spec = np.stack([S.stft(wave[:, c]) for c in range(wave.shape[1])])
    frames = spec.shape[1]
    splits = int(np.ceil(frames / T))
    mag = np.zeros((spec.shape[0], splits * T, F), dtype=np.float32)
    mag[:, :frames] = np.abs(spec[:, :, :F])
    return spec, mag.reshape(spec.shape[0], splits, T, F), frames


def masks(estimates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Spleeter's soft ratio mask. Sums to exactly 1 across stems by
    construction -- which is why the stems must sum back to the mix, and why
    that makes such a good assertion."""
    total = sum(e ** EXPONENT for e in estimates.values()) + EPS
    return {
        name: (e ** EXPONENT + EPS / len(estimates)) / total
        for name, e in estimates.items()
    }


def extend(mask: np.ndarray, mode: str) -> np.ndarray:
    """The net only masks 1024 of 2049 bins. Everything above ~11 kHz needs a
    value, and the choice is audible: `zeros` discards the top, `average`
    carries the per-frame mean up into it."""
    extra = S.BINS - F
    if mode == "zeros":
        pad = np.zeros(mask.shape[:-1] + (extra,), dtype=mask.dtype)
    elif mode == "average":
        pad = np.repeat(mask.mean(axis=-1, keepdims=True), extra, axis=-1)
    else:
        raise ValueError(mode)
    return np.concatenate([mask, pad], axis=-1)


def separate(wave: np.ndarray, mode: str = "zeros") -> tuple[dict[str, np.ndarray], float]:
    n = len(wave)
    spec, net_in, frames = analyse(wave)

    estimates, infer = {}, 0.0
    for name in STEMS:
        sess = ort.InferenceSession(
            f"{MODEL_DIR}/{name}.fp16.onnx", providers=["CPUExecutionProvider"]
        )
        t0 = time.perf_counter()
        out = sess.run(["y"], {"x": net_in})[0]
        infer += time.perf_counter() - t0
        estimates[name] = out.reshape(out.shape[0], -1, F)[:, :frames]

    out = {}
    for name, mask in masks(estimates).items():
        masked = spec * extend(mask, mode)
        out[name] = np.stack(
            [S.istft(masked[c], n) for c in range(masked.shape[0])], axis=-1
        )
    return out, infer


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "zeros"
    wave, sr = sf.read("atomic.wav", dtype="float64")
    print(f"in: {wave.shape} @ {sr} Hz  ({len(wave)/sr:.1f}s)   mask_extension={mode}")

    t0 = time.perf_counter()
    stems, infer = separate(wave, mode)
    wall = time.perf_counter() - t0

    for name, y in stems.items():
        sf.write(f"out_{name}_{mode}.wav", y, sr)
        print(f"  {name:<14} peak {np.abs(y).max():.3f} -> out_{name}_{mode}.wav")

    # The masks sum to 1, so the stems must sum back to the mix. Anything else
    # means the plumbing is wrong -- this is the assertion, not the vibe check.
    total = sum(stems.values())
    err = np.abs(total - wave).max()
    rms = 10 * np.log10(np.mean((total - wave) ** 2) / np.mean(wave ** 2) + 1e-20)
    print(f"\n  sum(stems) vs mix:  max err {err:.2e}   residual {rms:.1f} dB")

    audio = len(wave) / sr
    print(f"\n  inference {infer:.2f}s   total {wall:.2f}s   "
          f"for {audio:.0f}s audio  ->  {audio/wall:.1f}x realtime")


if __name__ == "__main__":
    main()
