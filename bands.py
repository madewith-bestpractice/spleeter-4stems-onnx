"""How much does Spleeter's 11 kHz ceiling actually cost *a volca*?

The received wisdom is that the ceiling is Spleeter's cardinal sin. That is
true of a 44.1 kHz production. This device stores 31,250 Hz mono, so its
Nyquist is 15,625 Hz -- everything above that is discarded by the hardware no
matter which model produced it.

So the band that Spleeter's ceiling actually costs us is only 11,025..15,625 Hz.
Everything above 15,625 is thrown away for free. This measures the split.
"""

import numpy as np
import soundfile as sf

SR = 44100
MODEL_CEIL = 11025.0   # bin 1024 of 2049 at 44.1 kHz -- where the U-Net stops
VOLCA_NYQ = 15625.0    # 31,250 Hz / 2 -- where the hardware stops


def band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    mono = x.mean(axis=1) if x.ndim > 1 else x
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    return float(np.sum(np.abs(spec[sel]) ** 2))


def main() -> None:
    mix, _ = sf.read("atomic.wav", dtype="float64")
    total = band_energy(mix, 0, SR / 2)

    bands = [
        ("0 - 11,025 Hz     modelled by the net", 0, MODEL_CEIL),
        ("11,025 - 15,625   LOST by `zeros`, volca CAN hear", MODEL_CEIL, VOLCA_NYQ),
        ("15,625 - 22,050   volca discards anyway, free", VOLCA_NYQ, SR / 2),
    ]
    print(f"Spectral budget of the mix ({len(mix)/SR:.0f}s of Atomic Dog):\n")
    for label, lo, hi in bands:
        e = band_energy(mix, lo, hi)
        print(f"  {label:<52} {100*e/total:6.3f}%   {10*np.log10(e/total):7.2f} dB")

    above = band_energy(mix, MODEL_CEIL, SR / 2)
    costly = band_energy(mix, MODEL_CEIL, VOLCA_NYQ)
    print(f"\n  Of everything above the net's ceiling, the share the volca could")
    print(f"  actually have reproduced: {100*costly/above:.1f}%")
    print(f"  -> `zeros` costs a volca {10*np.log10(costly/total):.1f} dB of the mix,")
    print(f"     not the {10*np.log10(above/total):.1f} dB it costs a 44.1 kHz render.")


if __name__ == "__main__":
    main()
