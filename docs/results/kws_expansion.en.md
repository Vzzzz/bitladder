# Modular keyword expansion on DS-CNN: reproduction and a quantization follow-up

Summary in English of the Russian working notes
[kws_expansion_2026-10-23.md](kws_expansion_2026-10-23.md) (full protocol, all tables).
Dates: 2026-10-23 … 2026-10-25. Code: [`audio/kws/expand.py`](../../audio/kws/expand.py),
notebook [`colab/kws.ipynb`](../../colab/kws.ipynb).

## What was reproduced

Paper: Khaymonenko, Saladukha, Rak, Rostov, *Scalable Keyword Spotting via Modular
Network Expansion*, Interspeech 2026, [arXiv 2607.19918](https://arxiv.org/abs/2607.19918).
A frozen base KWS network (BatchNorm statistics included) is extended with a small branch
(≤ 10k parameters) that taps intermediate activations and has its own head for new
keywords; only the branch is trained. Reported: macro FRR 4.37 % on new keywords at a
threshold calibrated to 1 % FAR, averaged over five held-out pairs of Google Speech
Commands v2, versus 6.46 % for a parameter-matched separate model and 22.3 % for a new
head only; base FRR on its own keywords 2.71 %.

Our setup differs in the backbone: DS-CNN 64×4 from Hello Edge (23k parameters,
10.6 MMAC) instead of the paper's SVDF network (150k), same features (40 log-mel,
16 kHz), 95.0 % on the official 12-class test set. Base: 8 keywords + `_silence_` +
`_unknown_` (the other 25 words; the held-out pair is removed from the pool entirely),
30 epochs. Branch: one tap per DS block, 1×1 reduce to *k* channels, concat with the
previous branch block, 3×3 mix (dense or depthwise-separable), BN, hard-swish, head over
new words + background; 20 epochs. Metric: per-word threshold at FAR 1 % on all
non-target clips of the test split, FRR on the new words, macro over the pair, mean ± σ
over three seeds. Colab T4, torch 2.11.

| method (pair {stop, go}, base fp32) | params | macro FRR |
|---|---|---|
| expansion branch, k=8 dense | 6.2k | 7.9 % |
| expansion branch, k=16 dense | 20.4k | 4.6 % |
| separate small DS-CNN | 6.0k | 24.1 % |
| new head on frozen pooled features | 0.2k | 92.1 % |

The ordering of the paper holds (branch < separate < head-only); the base keeps 2.5 %
FRR on its own words. Absolute numbers are worse than the paper's within the 10k budget
(7.9 % vs 4.37 %); the branch was not tuned for DS-CNN.

## What the paper does not have: bits

Base after 4-bit QAT (first/last layer 8-bit, PACT 8-bit activations): 16 040 bytes vs
89 876 in fp32, same accuracy (95.7–96.3 % on both pairs, three seeds). Branch trained
on top of the 4-bit base with its own weight precision (whole branch quantized, 8-bit
activations). FRR without the core-first rule (see below), two held-out pairs, 3 seeds:

| branch | params | weights | bytes | {stop, go} | {on, off} | mean |
|---|---|---|---|---|---|---|
| k=8 dense | 6 171 | fp32 | 24 550 | 6.0 ± 1.4 | 9.0 ± 0.8 | 7.5 |
| k=8 dense | 6 179 | 4-bit | 3 320 | 6.1 ± 1.8 | 9.2 ± 0.2 | 7.6 |
| k=8 dense | 6 179 | ternary | **1 494** | 7.7 ± 1.0 | 9.4 ± 0.8 | 8.6 |
| k=16 separable | 7 075 | fp32 | 28 038 | 5.1 ± 0.5 | 7.6 ± 0.6 | 6.3 |
| k=16 separable | 7 087 | 4-bit | **4 220** | 5.5 ± 0.4 | 7.8 ± 1.0 | **6.6** |
| k=16 separable | 7 087 | ternary | 2 142 | 8.9 ± 0.8 | 10.2 ± 2.0 | 9.6 |
| k=16 dense | 20 403 | fp32 | 81 350 | 2.7 ± 0.3 | 6.9 ± 1.4 | 4.8 |
| k=16 dense | 20 411 | 4-bit | 10 660 | 3.6 ± 0.3 | 7.1 ± 0.4 | 5.3 |
| k=16 dense | 20 411 | ternary | 4 581 | 5.2 ± 0.8 | 7.5 ± 0.3 | 6.3 |

Bytes are counted after bit packing (ternary five per byte, 4-bit two per byte) plus fp16
per-channel scales, biases and BatchNorm. Findings:

1. 4-bit branch weights are free (within σ) at every width. Ternary weights cost 1–4
   points, more when the branch has more capacity or depthwise convolutions.
2. The depthwise-separable branch fits the 10k budget and beats the dense k=8 branch in
   fp32 / 4-bit, but loses to it in ternary: ternarization hurts depthwise layers (nine
   weights per channel) most. A one-shot layer-sensitivity analysis of the base shows the
   same ([layer analysis](kws12_layer_analysis_2026-10-24.md)).
3. Within the 10k budget, a new keyword costs **1.5 KB of flash at 8.6 % FRR** (k=8
   ternary) or **4.2 KB at 6.6 %** (k=16 separable, 4-bit), against roughly 10 KB for an
   int8 branch at the paper's 4.37 %.
4. The second pair is harder (7–10 % vs 3–9 %) and shaped differently: "on" and "off" are
   equally hard, while in the first pair almost all the error is on "go". Averaging over
   pairs, as the paper does, is necessary.

## Two things we could not reproduce (questions to the authors)

**Core-first arbitration.** We implemented "the new head is consulted only when the core
outputs background" with the core firing when any of its words exceeds its own threshold
(calibrated at 1 % FAR). With a base that has never seen the new words, the core fires on
41–74 % of their clips (they are acoustically close to its own words: "go" / "no"), so the
branch is never consulted there and gated FRR is 30–58 % regardless of what the branch
does. Calibrating the core thresholds on Common Voice test windows (as in the paper)
helps a little (74 % → 41 %) but does not fix it; a soft gate
(p_branch · p_core(background)) gives ~30 %. All our headline numbers are therefore the
branch's own thresholded detection without the core-first rule. Question: are the
paper's FRR figures measured with the core-first rule applied, and if so, how does the
core behave on clips of the new keywords?

**Separate-model baseline.** Our parameter-matched separate DS-CNN (6k) gives 24 %
against the paper's 6.46 % for the "Ensemble" row. We did not tune it; the gap is
probably architecture and training, not the method.

Also observed: calibration negatives must be unseen during training. Once the base or
the branch is trained with Common Voice windows as background, thresholds calibrated on
Common Voice collapse (branch threshold 0.03–0.05) and FRR looks spuriously low (0.7 %).

## Reproducibility

Same machine, same torch, new Colab session: every row within σ (e.g. k=16 fp32
2.68 ± 0.31 % vs 2.68 ± 0.28 %). Different machine (RTX 3080 Laptop, torch 2.14): 4.5 %
for the configuration that gives 2.7 % on the T4. Machine-to-machine spread exceeds seed
spread, so rows are compared only within one environment.

## Base size is the lever (one seed, 2026-10-29, preliminary)

Same branch (k=16 depthwise-separable, 4-bit, 9.5 KB) on a DS-CNN 128×6 base (115k
parameters, 69 KB at 4 bits, 55 MMAC) gives 1.7 % FRR versus 7.7 % on the 64×4 base. On the
big base, branches that tap only the last blocks stay cheap: k=16 with 4 taps 6.3 KB → 2.1 %,
k=8 with 3 taps 2.2 KB → 4.1 %, k=8 with 2 taps 1.4 KB → 5.2 %. Time stride 1 on the small
base halves the branch error (4.7 %) at twice the MACs. One seed and one word pair so far;
three seeds and the second pair are queued (section 9 of the Russian report).
