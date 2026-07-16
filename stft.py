"""STFT/iSTFT pair, written to be ported to Dart verbatim.

The forward transform must produce the magnitudes Spleeter was trained on:
periodic Hann, frame 4096, hop 1024. Get the window wrong and the net sees a
spectrogram unlike anything in its training set.

Time *alignment* is not part of that contract -- the model is a conv net and is
translation-equivariant in time -- so this pads the start by N_FFT-HOP, which
Spleeter does not do. That padding is what makes every real sample sit under a
full four-window overlap, and it is the difference between reconstruction that
is exact and reconstruction that is merely exact in theory: on the ramp-in a
periodic Hann is ~1e-7, so w^2 is ~1e-13, and dividing by it amplifies float
noise into the only part of the signal that has no redundancy. Pad instead, and
the ill-conditioned region lives entirely in padding we throw away.

The inverse only has to invert the forward, so rather than reproduce Spleeter's
`inverse_stft_window_fn` and its 2/3 WINDOW_COMPENSATION_FACTOR -- two magic
constants that exist to undo TensorFlow-specific normalisation -- this uses the
canonical weighted overlap-add:

    y[n] = SUM_k w[n-kH] x_k[n-kH]  /  SUM_k w[n-kH]^2

the least-squares inverse, which needs no constants at all.
"""

import numpy as np

N_FFT = 4096
HOP = 1024
BINS = N_FFT // 2 + 1  # 2049

# Zeros in front of sample 0, so it gets the same overlap as the interior.
PAD = N_FFT - HOP  # 3072

# Periodic (not symmetric) Hann. np.hanning is symmetric and is NOT this: it
# differs by one sample, and that one sample is the difference between COLA
# holding and not.
WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float64)


def n_frames_for(n: int) -> int:
    return int(np.ceil((PAD + n) / HOP))


def stft(x: np.ndarray) -> np.ndarray:
    """[n_samples] -> [n_frames, BINS] complex."""
    n = len(x)
    n_frames = n_frames_for(n)
    padded = np.zeros((n_frames - 1) * HOP + N_FFT, dtype=np.float64)
    padded[PAD:PAD + n] = x
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    return np.fft.rfft(padded[idx] * WINDOW, axis=-1)


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    """[n_frames, BINS] complex -> [length] real."""
    frames = np.fft.irfft(spec, n=N_FFT, axis=-1)
    n_frames = frames.shape[0]
    total = (n_frames - 1) * HOP + N_FFT
    out = np.zeros(total, dtype=np.float64)
    wsum = np.zeros(total, dtype=np.float64)
    for i in range(n_frames):
        at = i * HOP
        out[at:at + N_FFT] += frames[i] * WINDOW
        wsum[at:at + N_FFT] += WINDOW ** 2
    out = np.divide(out, wsum, out=np.zeros_like(out), where=wsum > 1e-8)
    return out[PAD:PAD + length]


def test_roundtrip() -> None:
    """istft(stft(x)) == x, exactly, with no exempted samples.

    The one assertion that catches window, hop, padding and normalisation bugs
    in a single shot, before a model is anywhere near it.
    """
    rng = np.random.default_rng(0)
    for n in (44100, 44100 * 3 + 137, 5000, 1, 1024):
        x = rng.standard_normal(n)
        y = istft(stft(x), n)
        err = np.max(np.abs(x - y))
        assert err < 1e-9, f"n={n} roundtrip error {err:.3e}"
        print(f"  roundtrip n={n:<10} max err {err:.2e}  OK")
    print("  STFT/iSTFT pair is exact.")


if __name__ == "__main__":
    test_roundtrip()
