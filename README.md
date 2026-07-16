# spleeter-4stems-onnx

**Deezer's [Spleeter](https://github.com/deezer/spleeter) 4stems — drums, bass,
vocals, other — as ONNX.** Plus the STFT/mask pipeline around the graphs, since
the models are only half of a separator.

Deezer ship 4stems as a TensorFlow 1 checkpoint from October 2019.
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ship an excellent ONNX
export of **2stems** and state plainly: *"We only support the `2-stem` model at
present."* This is the missing half.

Weights are on the [Releases](../../releases) page: **19.7 MB per stem in fp16,
75 MB for all four.**

| | |
|---|---|
| Inference, all four stems | **1.69 s** for 30 s of 44.1 kHz stereo (~18× realtime, CPU) |
| `istft(stft(x)) − x` | **1.3e-15**, no exempted samples |
| `sum(4 stems) − mix` | **−153.1 dB** |

That last row is the one to trust. Spleeter's ratio masks sum to exactly 1 by
construction, so the stems *must* reconstruct the mix. They do, to machine
precision — which tests the STFT, the mask, the band extension and the iSTFT in
a single assertion, without anyone listening to anything.

## Why this didn't already exist

Not effort. **sherpa-onnx's `unet.py` cannot produce 4stems, and nothing about
the shapes tells you.** Spleeter reads its activations from config:

```jsonc
// configs/2stems/base_config.json
"model": { "type": "unet.unet", "params": {} }
//                              ^^ defaults: LeakyReLU(0.2) + ReLU

// configs/4stems/base_config.json
"model": { "type": "unet.unet", "params": {
    "conv_activation": "ELU", "deconv_activation": "ELU" } }
```

Identical architecture. Identical weight shapes, kernel for kernel. So
`load_state_dict` accepts the 4stems weights **without a murmur** and returns
garbage.

It presented as a max error of **945 against a TF output whose entire range was
178** — the error larger than the signal. Every hypothesis checked out clean: the
op-name mapping reproduced sherpa's exactly, `strided_slice_3` really was the
tensor entering the U-Net, BatchNorm really was in inference mode (`pred=False`),
the kernel shapes really were byte-identical between 2stems and 4stems. All true,
all irrelevant — the bug was in neither the weights nor the wiring. With ELU:
**945 → 2.6e-04.**

The thing that made it findable is sherpa-onnx's built-in check: their converter
runs the TF graph and the port on the same input and asserts they agree. Without
that, this ships silently wrong and you discover it by ear, much later. It is
kept here, and extended with a regression test — the generic converter must still
reproduce sherpa's hand-written 2stems mapping (3.04e-04).

## Use

```bash
pip install -r requirements.txt
# fetch the four *.fp16.onnx from Releases into convert/4stems/
python separate4.py average     # -> out4_{vocals,drums,bass,other}.wav
python stft.py                  # the roundtrip test; run it first
```

`mask_extension`: **use `average`, not Spleeter's `zeros` default.** The net
models 1024 of 2049 bins (to ~11 kHz); `zeros` discards everything above, which
is a −23 dB hole in the reconstruction. `average` carries the per-frame mean up
and reconstructs exactly. `bands.py` measures it.

## The STFT contract

The forward transform must produce the magnitudes the net was trained on:
**periodic Hann, frame 4096, hop 1024**. `np.hanning` is *symmetric* and is not
this; it differs by one sample, and that one sample is the difference between
COLA holding and not.

Time *alignment* is not part of the contract — the U-Net is convolutional and
translation-equivariant in time — so `stft.py` pads the front by `N_FFT-HOP`,
which Spleeter does not. That padding is load-bearing. Without it reconstruction
is exact in theory and broken in practice: on the ramp-in a periodic Hann is
~1e-7, so `w²` is ~1e-13, and weighted overlap-add's divide-by-`w²` amplifies
float noise in the only region of the signal with no redundancy. The first
attempt failed its roundtrip by **2.3** for exactly this reason. Pad, and the
ill-conditioned region lives entirely in padding that gets discarded.

This is also why `stft.py` doesn't reproduce Spleeter's
`inverse_stft_window_fn` or its 2/3 `WINDOW_COMPENSATION_FACTOR`: those exist to
undo TensorFlow-specific normalisation. The inverse only has to invert *this*
forward, so it uses the canonical least-squares WOLA and needs no constants.

## Rebuilding the models

```bash
cd convert
python3.12 -m venv .venv312            # TF has no 3.14 wheels; 3.12 is not optional
./.venv312/bin/pip install tensorflow torch onnx onnxruntime \
    onnxmltools onnxconverter-common onnxscript numpy
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

### How the op-name mapping is derived

`convert_to_torch.py` hardcodes TF op-name offsets for exactly two stems
(`vocals → conv2d/*`, `accompaniment → conv2d_7/*`, `bn_offset=12`).
`convert_any.py` derives them instead: each frozen graph contains exactly one
stem's U-Net, so collect its `Const` ops and sort by index.

| stem | conv2d | conv2d_transpose | batch_normalization |
|---|---|---|---|
| vocals | 0–6 | 0–5 | 0–4, **6**–11 |
| drums | 7–13 | 6–11 | 12–16, **18**–23 |
| bass | 14–20 | 12–17 | 24–28, **30**–35 |
| other | 21–27 | 18–23 | 36–40, **42**–47 |

Mind the gap. TF allocates **12** BatchNorm indices per stem but only **11**
exist — `conv5` has none — and freezing prunes the dead one, so the sorted list
is exactly 11 long and lines up with the U-Net's 11 BatchNorms. That is observed
(`inspect_graph.py`), not assumed.

`inspect_graph.py`, `trace_input.py` and `bisect_layers.py` are the debugging
trail that found the ELU. They're kept because the next person to touch a
Spleeter graph will want them.

## Licensing

Apache-2.0 (see `LICENSE`), because parts derive from sherpa-onnx. **The model
weights are Deezer's and are MIT.** Full attribution and the provenance argument
for the weights — including the one genuine ambiguity, which is that Spleeter's
README licenses "the code" while the authors' own JOSS paper says "source code
and pre-trained models" — are in [`NOTICE`](NOTICE). Read it before shipping
these in a product.

## Credits

- **[Deezer](https://github.com/deezer/spleeter)** — Spleeter, and for training
  on a licensed catalogue and releasing the weights *because* the data couldn't
  be. That decision is the only reason this model is usable rather than merely
  good.
- **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** (Xiaomi, Fangjun
  Kuang) — the TF→PyTorch→ONNX route, the U-Net port, and the TF-vs-torch
  assertion that made the ELU bug findable instead of silent. This repo is a
  four-stem extension of their work, not a replacement for it.

Built while adding stem separation to a sample manager for the Korg volca sample
2 — hence `volca_render.py`, which renders stems mono at 31,250 Hz to hear what
that hardware would actually play.
