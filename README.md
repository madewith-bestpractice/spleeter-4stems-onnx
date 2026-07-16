# stem-spike

Standalone probe for Sampeltu's stem-separation feature. **Outside the app repo
on purpose** — nothing here imports or touches `volca/`.

It exists to answer four questions no published source could:

1. Does sherpa-onnx's Spleeter export actually run? — **yes**
2. What must the mask maths be? — **settled, and proven exact**
3. Does it sound right through the volca pipeline? — **listen to `volca_*.wav`**
4. How fast is it? — **~46× realtime inference, CPU only**

## Results (30s of Atomic Dog, stereo 44.1 kHz, M-series CPU)

| | |
|---|---|
| Inference, 2 stems | **0.65 s** for 30 s audio (~46× realtime) |
| Whole pipeline | 2.1 s (the Python iSTFT loop is the bottleneck, not the net) |
| `sum(stems) − mix`, `average` | **−157.7 dB** — machine precision |
| `sum(stems) − mix`, `zeros` | −23.0 dB — *this is the >11 kHz band being binned* |

## The two things this settled

**Use `mask_extension="average"`, not Spleeter's `zeros` default.** The net only
models 1024 of 2049 bins (up to ~11 kHz). `zeros` discards everything above it.

And it kills an argument that was in the design notes: *"the ceiling costs less
on a volca, because its Nyquist is 15.6 kHz rather than 22."* **False.** Measured
(`bands.py`): **98.6% of the energy above the net's ceiling lives in
11–15.6 kHz** — inside the volca's range. The device's lower Nyquist saves
**0.1 dB**. It was never the Nyquist that saved us; it's `average`, which
reconstructs the band exactly.

(Caveat: the source is an MP3, which lowpasses ~16 kHz, so the near-zero energy
above 15.6 kHz is partly the encoder. Doesn't change the conclusion — the lost
band sits inside the volca's range either way.)

## Files

| | |
|---|---|
| `stft.py` | STFT/iSTFT pair. **Written to be ported to Dart verbatim.** Self-tests. |
| `separate.py` | The pipeline: analyse → U-Net → ratio mask → extend → apply → iSTFT |
| `separate4.py` | The 4-stem version. **This is the one.** |
| `convert/` | TF checkpoint → frozen pb → PyTorch → ONNX, for all four stems |
| `bands.py` | What the 11 kHz ceiling actually costs *this device* |
| `volca_render.py` | Renders stems mono @ 31,250 Hz — what the hardware would really play |
| `convert/inspect_graph.py`, `trace_input.py`, `bisect_layers.py` | The debugging trail that found the ELU |

```bash
./.venv/bin/python stft.py              # roundtrip test — run this first
./.venv/bin/python separate4.py average # drums/bass/vocals/other
./.venv/bin/python separate.py average  # 2-stem; `zeros` to hear that decision
./.venv/bin/python bands.py
./.venv/bin/python volca_render.py      # -> volca_{mix,vocals,accompaniment}.wav
```

## Why the STFT is padded at the start (and Spleeter's isn't)

The forward transform must produce the magnitudes the net was trained on:
periodic Hann, 4096/1024. Time *alignment* is not part of that contract — the
model is a conv net, translation-equivariant in time — so `stft.py` pads the
front by `N_FFT-HOP`, which Spleeter does not.

That padding is load-bearing. Without it, reconstruction is exact in theory and
broken in practice: on the ramp-in a periodic Hann is ~1e-7, so `w²` is ~1e-13,
and WOLA's divide-by-`w²` turns float noise into the only part of the signal
with no redundancy. The first attempt failed the roundtrip by **2.3** for
exactly this reason. Pad, and the ill-conditioned region lives entirely in
padding that gets thrown away — roundtrip error drops to **1.3e-15**, with no
exempted samples.

This is also why `stft.py` doesn't reproduce Spleeter's `inverse_stft_window_fn`
or its 2/3 `WINDOW_COMPENSATION_FACTOR`: those undo TensorFlow-specific
normalisation. The inverse only has to invert *our* forward, so it uses the
canonical least-squares WOLA and needs no constants.

## 4 stems: DONE — drums/bass/vocals/other

`convert/` produces them. **~19.7 MB per stem fp16, 75 MB for all four** — half
of Koala's 150 MB. Inference **1.69 s for 30 s of audio, all four stems (18×
realtime)**, and `sum(4 stems) − mix` is **−153.1 dB**, so the 4-stem pipeline is
provably correct too.

### Why nobody had done this (the ELU trap)

sherpa-onnx publishes 2stems only, and it is not laziness — **their `unet.py`
cannot produce 4stems and nothing about the shapes says so.** Spleeter takes its
activations from config:

```
2stems:  "params": {}                                 -> LeakyReLU(0.2) + ReLU  (defaults)
4stems:  "params": {"conv_activation": "ELU",
                    "deconv_activation": "ELU"}       -> ELU everywhere
```

Identical architecture, identical weight shapes, so `load_state_dict` accepts
the 4stems weights **without a murmur** and returns garbage. It cost a real
debugging session: max err **945**, against a TF output whose entire range was
178 — the error was bigger than the signal. Everything checked out (mapping,
input tensor, inference-mode BN, byte-identical kernel shapes) because the bug
was in neither the weights nor the wiring. With ELU: **945 -> 2.6e-04.**

### The other trap: op-name offsets

`convert_to_torch.py` hardcodes TF op-name offsets for exactly two stems
(`vocals -> conv2d/*`, `accompaniment -> conv2d_7/*`, `bn_offset=12`).
`convert/convert_any.py` derives them from the graph instead — each frozen graph
holds exactly one stem's U-Net, so collect the Const ops and sort:

| stem | conv2d | conv2d_transpose | batch_normalization |
|---|---|---|---|
| vocals | 0–6 | 0–5 | 0–4, **6**–11 |
| drums | 7–13 | 6–11 | 12–16, **18**–23 |
| bass | 14–20 | 12–17 | 24–28, **30**–35 |
| other | 21–27 | 18–23 | 36–40, **42**–47 |

Note the gap: TF allocates 12 BatchNorm indices per stem but only 11 exist
(`conv5` has none), and **freezing prunes the dead one** — so the sorted list is
exactly 11 long and lines up. Verified by reproducing sherpa's hand-written
2stems mapping (3.04e-04), which is the regression test for the generic path.

### Rebuild

```bash
cd convert
python3.12 -m venv .venv312 && ./.venv312/bin/pip install tensorflow torch onnx \
    onnxruntime onnxmltools onnxconverter-common onnxscript numpy
# TF has no 3.14 wheels — 3.12 is not optional.
curl -LO https://github.com/deezer/spleeter/releases/download/v1.4.0/4stems.tar.gz
mkdir -p 4stems && tar xzf 4stems.tar.gz -C 4stems
for s in vocals drums bass other; do
  ./.venv312/bin/python convert_to_pb.py --model-dir ./4stems \
      --output-node-names ${s}_spectrogram/mul \
      --output-filename ./4stems/frozen_${s}_model.pb
  ./.venv312/bin/python convert_any.py --name $s --model-dir ./4stems
done
./.venv312/bin/python export_4stems.py
```

## Superseded: 2 stems

sherpa-onnx only published a 2stems export, but their conversion scripts
(`k2-fsa/sherpa-onnx`, `scripts/spleeter/`) look like they generalise by
parameter, not by porting — `run.sh` selects the stem with an output node name:

```
convert_to_pb.py --output-node-names vocals_spectrogram/mul
convert_to_torch.py --name vocals
export_onnx.py
```

Spleeter's 4stems checkpoint uses the same `{stem}_spectrogram/mul` convention
for `vocals`/`drums`/`bass`/`other`, and the per-stem U-Nets are architecturally
identical (9,826,759 params each — exactly linear across the 2/4/5-stem builds).
So the recipe should be: fetch `4stems.tar.gz` from Deezer's v1.4.0 release, run
the three scripts four times, out come four ONNX files (~78 MB fp16 total).

**Blocker:** `convert_to_pb.py` needs TensorFlow, which has no Python 3.14
wheels, and 3.14 is the only Python on this machine. Needs
`brew install python@3.12` + TF + PyTorch first. Unproven until someone does it —
it is the last real technical risk in the feature.
