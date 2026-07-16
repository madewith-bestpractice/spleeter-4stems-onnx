"""Render the stems the way the device would actually get them.

Listening to a 44.1 kHz stereo vocal isolation tells you how Spleeter does. It
does not tell you how *Sampeltu* would sound, which is the only question that
matters: the app downmixes to mono and resamples to 31,250 Hz before a byte
reaches the hardware. Artifacts audible under critical listening at 44.1/stereo
may or may not survive that, and the only way to know is to listen to the thing
that ships.

Mirrors lib/audio/sample_convert.dart: StereoMix.average, then the windowed-sinc
resample to Volca.sampleRate.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import separate4

VOLCA_SR = 31250
SR = 44100


def to_volca(x: np.ndarray) -> np.ndarray:
    """[n, 2] @ 44100 -> [m] @ 31250 mono. StereoMix.average, then resample."""
    mono = x.mean(axis=1)
    # 44100 -> 31250 is 625/882 exactly. No rounding, no drift.
    return resample_poly(mono, 625, 882)


def main() -> None:
    wave, sr = sf.read("atomic.wav", dtype="float64")
    assert sr == SR

    stems, _ = separate4.separate4(wave, mode="average", quiet=True)
    stems["mix"] = wave

    print("Rendered through the real pipeline (mono, 31,250 Hz) --")
    print("these are what the volca would actually play:\n")
    for name, y in stems.items():
        v = to_volca(y)
        peak = np.abs(v).max()
        # Match the app: 16-bit is what the hardware stores.
        out = f"volca_{name}.wav"
        sf.write(out, np.clip(v, -1, 1), VOLCA_SR, subtype="PCM_16")
        print(f"  {name:<14} {len(v)/VOLCA_SR:5.1f}s  peak {peak:.3f}  -> {out}")

    # What one volca slot can actually hold, for scale.
    print(f"\n  For reference: the device holds ~130s of audio in TOTAL across")
    print(f"  all 200 slots. This 30s excerpt is ~23% of the whole instrument.")


if __name__ == "__main__":
    main()
