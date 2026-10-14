# Predictive Vision

CG 501 (Computation and Cognition) course project: replicate and extend

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

## Timeline

| When | Work |
|---|---|
| By 20 Oct | Mid-term presentation video (idea, background, methods, analysis plan, timeline, expected outcomes) |
| 21–29 Oct | Baseline model and preprocessing, Fig. 2 reproduced |
| 30 Oct–5 Nov | Endstopping and feedback ablation, Figs. 3 and 5 reproduced |
| 6–12 Nov | Extension: context sweeps, seeds, statistics, plots |
| 13–20 Nov | Report (1500–3000 words), end-term video, clean repository |
| 26 Nov | Individual peer reviews |

Fallback: if endstopping does not reproduce by about 1 Nov, report what did and did not reproduce and why, and run the sweeps on whatever the baseline supports.

## Deliverables

- Mid-term presentation video, due 20 Oct 2026, 10 PM
- End-term presentation video, due 20 Nov 2026, 10 PM
- Codebase and report (PDF), due 20 Nov 2026, 10 PM
- Peer reviews (individual), due 26 Nov 2026, 10 PM

## Reproducibility

A reader should be able to clone this repository and regenerate every figure in the report. Seeds are fixed and recorded. Validation is done on known cases before any new experiment is trusted.

## Repository layout (planned)

```
predictive-vision/
├── nn0199_79.pdf        # the paper
├── README.md
├── data/                # BSDS500 images and ground truth (present)
├── src/                 # model, learning rules, stimuli, analysis
├── experiments/         # scripts that regenerate each figure
├── figures/             # generated plots
└── report/              # final report
```

## Team

_Names and contributions TBD._
