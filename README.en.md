# bitladder

[![ci](https://github.com/Vzzzz/bitladder/actions/workflows/ci.yml/badge.svg)](https://github.com/Vzzzz/bitladder/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vzzzz/bitladder/blob/main/colab/kws.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Русская версия: [README.md](README.md).*

**A ladder from 32 bits down to 1.** Extreme optimization of tiny always-on neural
networks: quantization-aware training (uniform, ternary 1.58-bit, binary), pruning,
per-layer sensitivity analysis, and honest accounting of what is left in bytes,
operations and milliseconds on real hardware. Two applied tracks: a **wake word on a
microcontroller** and **voice enhancement with a minimal parameter count** for a phone
and a Jetson.

The applied target is the wake word for [Kostyan](https://github.com/Vzzzz/skull-assistant/),
a voice assistant in an anatomical skull: an MCU listens all the time and wakes a sleeping
Jetson Orin Nano on the keyword.

Inspired by [ViT-1.58b: Mobile Vision Transformers in the 1-bit Era](https://arxiv.org/abs/2406.18051).

## Results so far (2026-11-03)

Keyword spotting on Google Speech Commands v2 with a DS-CNN 64×4 (23k parameters, 10.6 MMAC),
Colab T4, three seeds where ± σ is given:

| what | number |
|---|---|
| 12-class accuracy, official test set, fp32 | **95.2 %** (Hello Edge, DS-CNN S: 94.4 %) |
| bit ladder of the same model (QAT, first/last layer 8-bit) | 4-bit: **95.2 % in 16 KB**; ternary: **92.5 % in 10.6 KB**; 2-bit: 88.2 % in 11.5 KB; binary: 84.5 % in 9.2 KB |
| a new keyword added to a frozen 64×4 base (branch ≤ 10k params) | 4.2 KB at 6.6 % FRR (4-bit) or 1.5 KB at 8.6 % (ternary), two word pairs |
| a new keyword on a 128×6 base (69 KB at 4 bits, 55 MMAC) | **6.3 KB at 4.3 ± 2.1 %**, 2.2 KB at 5.3 ± 1.8 %, 1.4 KB at 6.6 ± 1.8 % (FAR 1 %, two pairs, 3 seeds) |

*Modular expansion* reproduces [Yandex, Interspeech 2026](https://arxiv.org/abs/2607.19918):
the base network is frozen and only a small branch is trained per new keyword. The
ordering of methods from the paper holds (branch < separate model < head-only); the paper's
4.37 % over five pairs with an int8 branch of about 10 KB is reached with a 6.3 KB branch on
the 128×6 base. What the paper does not have: the main lever is the size of the base, not
of the branch; 4-bit QAT of the base is free (three seeds); distilling the small base from the
big one gives it +0.3 points and its branch −0.9 FRR points at the same bytes; ternary branch
weights cost 2.5 points for half the bytes and hit depthwise convolutions first; mixed
precision from the layer analyzer adds nothing while 4 bits are free. The paper's core-first
arbitration rule does not survive unseen, acoustically close keywords, and calibrating
thresholds on hard negatives (Common Voice) does not fix it. Tables, protocol and open
questions in English: [docs/results/kws_expansion.en.md](docs/results/kws_expansion.en.md);
full working notes in Russian: [docs/results/kws_expansion_2026-10-23.md](docs/results/kws_expansion_2026-10-23.md).

The ternary 12-class model is both smaller and four points more accurate than the 2-bit one,
the ViT-1.58b picture on a model four orders of magnitude smaller; below 4 bits the byte ladder
hits the floor set by the 8-bit first and last layers (about 3.5 KB).

One-shot layer sensitivity of the 12-class model (accuracy drop in points when only that
layer is quantized / pruned, no retraining): 8 and 4 bits are free everywhere; depthwise
layers are fragile; pointwise layers tolerate 50 % unstructured sparsity almost for free.
QAT recovers almost everything the one-shot analysis predicts lost.
[Report](docs/results/kws12_layer_analysis_2026-10-24.md).

![layer sensitivity heatmap](docs/results/kws12_sensitivity_2026-10-24.png)

## Layout

```
bitladder/            library
  quant/              STE weight quantizers (uniform / LSQ, ternary BitNet, binary XNOR), PACT
                      activations, QuantConv2d / QuantLinear, apply_quantization
  measure/            bytes after bit packing, MACs and BOPs, layer analyzer (weight and
                      activation statistics, one-shot sensitivity to bits and pruning)
  models/, data/      tiny models and loaders for the MNIST sandbox
  cli.py, train.py    shared quantization CLI args, training loop
audio/kws/            keyword spotting: log-mel without torchaudio, feature cache, DS-CNN,
                      run.py (training), expand.py (modular expansion), analyze.py,
                      cv_negatives.py (hard negatives from Common Voice)
colab/kws.ipynb       the same scripts in Google Colab for long sweeps
sandbox_mnist/        MNIST warm-up
docs/                 plan, decision log (ADR), literature review, hardware notes, results
tests/                31 unit tests (quantizers, packing, sizes, data pipeline, analyzer, GRU, distillation)
results/              run artifacts (not in git)
```

Documentation is in Russian: [plan](docs/plan.md), [decision log](docs/decision-log.md),
[literature review](docs/PUBLICATIONS.md), [results](docs/results/),
[hardware](docs/hardware.md), [STM32 deployment guide](docs/stm32_deploy.md). `AGENTS.md` is the context file for the coding agent the
project is developed with; it doubles as the up-to-date project status.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # or cu128 / cpu
pip install -e ".[dev]"
pytest -q

# Speech Commands v2 (CC BY 4.0): http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# and speech_commands_test_set_v0.02.tar.gz, extracted into data/
python audio/kws/cache.py --root data/speech_commands_v2 --test-root data/speech_commands_test_v2
python audio/kws/run.py --epochs 30                                   # 12 classes, fp32
python audio/kws/run.py --keywords yes,no,up,down,left,right,on,off --exclude stop,go --unknown-ratio 1.0 --name base8
python audio/kws/expand.py --base results/base8.json --new stop,go --k 16 --separable --epochs 20
python audio/kws/expand.py --base results/base8.json --new stop,go --k 16 --separable --scheme ternary --abits 8 --no-first-last
python audio/kws/analyze.py --run results/base8.json                  # sensitivity table and heatmap
```

Every run writes a JSON and a checkpoint to `results/`: accuracy or FRR, parameters,
bytes after bit packing, MACs and BOPs. Reported numbers were obtained with torch 2.14.1
(cu130, RTX 3080 Laptop) and torch 2.11.0 (cu130, Colab T4); machine-to-machine spread
is larger than seed spread, so compare rows only within one environment.

## Data and licenses

- [Speech Commands v2](https://arxiv.org/abs/1804.03209) (Warden, 2018), CC BY 4.0.
- [Common Voice](https://commonvoice.mozilla.org/) (Mozilla), CC0; used only as hard
  negatives for threshold calibration.
- Modular expansion follows Khaymonenko et al., *Scalable Keyword Spotting via Modular
  Network Expansion*, Interspeech 2026, [arXiv 2607.19918](https://arxiv.org/abs/2607.19918).
- DS-CNN and the KWS protocol follow Zhang et al., *Hello Edge*, [arXiv 1711.07128](https://arxiv.org/abs/1711.07128).

Code is MIT-licensed. If you use it, a link back is appreciated.

## Status

Phase 1, wake-word track. Next steps are listed in [AGENTS.md](AGENTS.md) (in Russian).
