# Predictive Vision

> Rao, R. P. N. & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* Nature Neuroscience 2(1), 79–87. (`nn0199_79.pdf`)

## Idea

The paper proposes that feedback connections carry **predictions** from higher to lower visual areas, while feedforward connections carry only the **residual error** (actual minus predicted activity). Trained on natural images, a hierarchical network of such predictive estimators develops simple-cell-like receptive fields, and its error-detecting neurons show endstopping and other extra-classical receptive-field effects.

## Project plan

### Part 1: Replication

Implement the hierarchical predictive coding model from the paper's equations (Eqs. 1–9) in NumPy, with no library black boxes, and reproduce:

| Target | What it shows |
|---|---|
| Fig. 2 | Level-1 basis vectors become oriented Gabor/edge-like filters; level-2 RFs combine them |
| Fig. 3 | Error-detecting neurons respond strongly to short bars and weakly to long bars (endstopping) |
| Fig. 5 | Removing level-2 feedback eliminates endstopping (82% of neurons in the paper) |

These are the validation checkpoints. The extension is only meaningful once they reproduce.

### Part 2: Extension (the twist)

**Question:** How does the error-neuron response scale with how well the surround predicts the center, and how much of that suppression is due to top-down feedback?

**Hypothesis:** If endstopping reflects prediction error, suppression should grow as the surround becomes a better predictor of the center, and should largely vanish when feedback is removed.

**Method:**
1. Sweep the stimulus context continuously: bar length (1–25 px) and surround orientation relative to center (0°–90°).
2. At each point, measure the error magnitude ‖r − rᵗᵈ‖.
3. Repeat with feedback intact and with feedback removed, over 5 or more random seeds.
4. Report mean ± confidence interval, and the fraction of suppression attributable to feedback.

**Alternative explanation (Discussion):** recurrent lateral inhibition (Eq. 8) instead of top-down feedback. Implemented if time allows, otherwise discussed in text.

**Optional, only if ahead of schedule:** Gaussian vs. sparse kurtotic prior (the paper claims the effects hold under both).

### Out of scope (future directions)

Temporal prediction (the V matrix), error budget across levels, noise-variance sweeps, extra hierarchy levels, motion/MT and other modalities.

## Data

The paper's natural images are not available, so we use the Berkeley Segmentation Dataset (BSDS500) in `data/`:

- `data/images/{train,val,test}`: 200 / 100 / 200 natural-image JPEGs (481×321)
- `data/ground_truth/{train,val,test}`: human segmentation annotations (`.mat`), not needed for this project
- Use: only the images, as natural-image patches. Train on `train`, tune on `val`, evaluate stimuli and statistics on `test`.
- Source: Arbelaez et al., "Contour Detection and Hierarchical Image Segmentation", IEEE TPAMI 2011 (BSDS500)
- Licence: _TBD, check the terms on the Berkeley BSDS page before the report_

## Model summary

- Level 0: image patches, prewhitened with a center-surround difference-of-Gaussians filter
- Level 1: 3 modules, each with 32 feedforward neurons (RFs on 16×16 patches), Gaussian-windowed inputs
- Level 2: 1 module, 128 feedforward neurons, taking the concatenated level-1 responses (96 inputs)
- Linear generative model for endstopping (f(x) = x), tanh for the extra-classical simulations
- Estimate update: gradient descent on E (Eq. 7); basis update: Hebbian-like rule (Eq. 9)
- Parameters from the paper: k1 = 0.5, σ² = 1, σ²_td = 10, α = 1, λ = 0.02, k2 initial 1, divided by 1.015 every 40 inputs

## Status and results (so far)

Reproduce with `.venv/bin/python experiments/<script>.py`:

| Script | Purpose |
|---|---|
| `train_baseline.py` | stage-1 (level 1) then stage-2 (level 2) training; Fig. 2-style RFs |
| `endstopping.py` | per-neuron endstopping, with/without feedback (Fig. 3/5-style) |
| `fig4_autocorrelation.py` | Fig. 4 check: orientation autocorrelation on BSDS500 |
| `sweep_level2.py`, `diagnose_level2.py`, `unexplained_vs_length.py`, `bottleneck_test.py` | level-2 diagnostics |
| `run_extension.py`, `analyze_extension.py` | 5-seed length/orientation sweeps with CIs; produces `figures/extension_results.png` |

**Replication: partial.** The network trains stably, level 2 couples centre and flank modules, and
removing feedback changes the error-neuron response, but the effect is much smaller than the paper's
(28/32 endstopped neurons, 82% removed by ablation). Learned level-1 RFs are noisy rather than clean Gabors
(a linear-Gaussian model only fixes a subspace). BSDS500 has the paper's orientation autocorrelation
structure qualitatively (along > orthogonal, 3-6x) but about half the magnitude (Fig. 4 check).

**Extension (5 seeds, population error norm of the centre module, bar-length sweep):**
suppression index S = (peak - plateau)/peak.

| Stimulus | S with feedback | S without | difference |
|---|---|---|---|
| DoG bar, 0 deg | 25.2 +- 8.3 % | 7.3 +- 1.7 % | +17.8 +- 7.6 (p=0.003) |
| raw bar, 0 deg | 20.3 +- 13.2 % | 0.1 +- 0.1 % | +20.2 +- 13.2 (p=0.013) |
| DoG bar, 90 deg | 18.4 +- 19.9 % | 3.2 +- 0.2 % | +15.1 +- 20.0 (n.s.) |
| raw bar, 90 deg | 5.4 +- 10.4 % | 0.0 +- 0.1 % | +5.4 +- 10.5 (n.s.) |

Surround-orientation sweep: no detectable orientation-contrast effect
(error(same)/error(orthogonal) about 0.95, same with and without feedback), unlike the paper's Fig. 6.
Feedback therefore accounts for roughly 18-20 points of length-dependent suppression for bars
extending across modules (0 deg), none detectable for the orthogonal axis.

**Held-out check and alternative explanation** (`heldout_and_alternative.py`, same 5 trained nets, no re-tuning):
bar widths 2 and 5 px and a 3 px vertical offset. The feedback effect keeps its sign in every condition
(+7 to +20 points; p < 0.05 in 7 of 10 held-out conditions, 0.057-0.077 in 3) but with wide intervals (n = 5 seeds).
A feedback-free divisive normalization (centre / sqrt(1 + beta * flank energy), one parameter fitted on the
tuning conditions) gives suppression of similar size (raw bars: 7-18% vs 7-20% for feedback), so these bar-length
data do not by themselves distinguish top-down prediction from surround normalization.

**Deviations from the paper (must be disclosed in the report):**
- BSDS500 instead of the paper's 5 images; DoG sigmas (1, 1.6), Gaussian window sigma 4 px chosen by us.
- Weights shared across the three level-1 modules; batch updates (batch 4) scaled to mimic per-input steps.
- Euler step size scaled by the spectral norm of U for stability; inference run for 40 steps (200 for tests).
- Learning-rate schedule: k2 = 0.05, not decayed (paper: 1, divided by 1.015 every 40 inputs);
  level-1 init column norm 1, level-2 init norm 5.
- Staged training (level 1 then level 2) as in the paper's Methods, with level 2 fed bottom-up r1.
- Level-1 responses standardised by one fixed constant (`r1_scale`) before level 2; without it U2 collapses.
- Level 2 uses alpha2=0.01, lambda2=0.002, k2=0.5 (paper: 1, 0.02, shared), chosen by a small sweep.
- No gain adaptation of basis vectors (disabled; it killed units).

## Reproducibility

A reader should be able to clone this repository and regenerate every figure in the report. Seeds are fixed and recorded. Validation is done on known cases before any new experiment is trusted.

## Repository layout (planned)

```
predictive-vision/
├── nn0199_79.pdf        # the paper
├── README.md
├── data/                # BSDS500 images and ground truth (present)
├── src/                 # data, model, stimuli
├── experiments/         # scripts that regenerate each figure
├── results/             # trained weights and pickled measurements
├── figures/             # generated plots
└── report/              # final report (to do)
```
