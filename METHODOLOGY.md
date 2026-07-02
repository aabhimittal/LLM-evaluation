# Methodology

This document states exactly what each Caliper instrument computes, the assumptions
behind it, and where it can mislead you. Every estimator here is validated in
`tests/` against simulated models whose ground truth is known.

## 1. Adaptive ability estimation (IRT)

### Model

Each multiple-choice item *i* has discrimination *aᵢ*, difficulty *bᵢ* and a fixed
guessing floor *cᵢ = 1/(number of choices)*. A model with latent ability θ answers
correctly with probability (three-parameter logistic, 3PL):

```
P(correct | θ, aᵢ, bᵢ, cᵢ) = cᵢ + (1 − cᵢ) · σ( aᵢ (θ − bᵢ) )
```

### Ability estimation

Given responses y₁…yₙ, we estimate θ by **MAP** under a standard-normal prior:

```
θ̂ = argmax_θ  Σᵢ [ yᵢ log Pᵢ(θ) + (1−yᵢ) log(1−Pᵢ(θ)) ] − θ²/2
```

The reported standard error is the inverse square root of the observed information
(numeric curvature of the negative log-posterior at θ̂); the 95% CI is θ̂ ± 1.96·SE.
The normal-approximation CI is standard in computerized adaptive testing (CAT) and
accurate once ~10+ informative items have been administered.

### Adaptive item selection

The Fisher information of item *i* at ability θ (3PL form) is

```
Iᵢ(θ) = aᵢ² · (Qᵢ/Pᵢ) · ( (Pᵢ − cᵢ) / (1 − cᵢ) )²,   Qᵢ = 1 − Pᵢ
```

Each round selects among the top-*k* unseen items by information at the current θ̂
("randomesque" exposure control, k=5) and stops when SE ≤ 0.30 (configurable) or the
item budget is exhausted. Because information concentrates near θ̂, **30–50 adaptive
items typically match the precision of hundreds of randomly chosen ones** — the same
principle behind the GRE and tinyBenchmarks.

### Item calibration

`caliper calibrate` fits (aᵢ, bᵢ) from a correctness matrix (models × items) by
**alternating MAP** (joint-mode approximation to marginal maximum likelihood):
abilities re-estimated given items, then per-item (log a, b) by L-BFGS with priors
log a ~ N(0, 0.5²), b ~ N(0, 1.5²); the θ scale is re-standardized each iteration for
identifiability. Parameter recovery on synthetic data: difficulty correlation
r > 0.85 with 48 respondents (see `tests/test_irt.py`).

**Caveat — the bundled bank.** The shipped parameters (`synthetic-demo-v1`) were
calibrated on a *simulated* population, because public per-item correctness data for
real models is gated. The questions are real; the difficulty ordering is not
empirical. Treat demo-mode θ values as a demonstration of the machinery; recalibrate
on a real correctness matrix for research use. IRT abilities are also only comparable
**within one calibration** of one bank.

## 2. Uncertainty-aware LLM-as-judge

Known failure modes of judge models: **position bias** (preferring slot A),
**verbosity bias** (preferring longer), **self-inconsistency** at nonzero temperature.
Caliper's design:

- every comparison is presented in **both orders**, sampled **n=3 times** each at
  temperature 0.6;
- the debiased score for response A is the mean over all votes (win=1, tie=0.5),
  which cancels any additive position preference;
- a **position flip** flag records whether the majority verdict changed with order —
  flipped verdicts are genuinely ambiguous comparisons *or* judge pathology;
- the **audit** aggregates: flip rate across comparisons, and the correlation between
  sign(length difference) and outcome (verbosity bias), plus mean vote agreement.

### Ranking

Match outcomes (possibly fractional) feed a **Bradley–Terry** model
`P(a beats b) = σ(sₐ − s_b)` fit by penalized MLE (L2, sum-zero). Uncertainty comes
from a **nonparametric bootstrap over matches** (default 200 resamples), reported as
95% percentile intervals on the Elo-like scale `1000 + s·400/ln 10`. This mirrors
Chatbot Arena's methodology.

**Caveat.** Debiasing by averaging removes *additive* position bias but not
interactions (e.g. a judge that only favors slot A for long responses). The audit
exposes residual pathologies; a flip rate ≫ 0.1 on clear-cut pairs means get a better
judge.

## 3. Metamorphic robustness

A meaning-preserving transformation of the input should not change the answer.
Operators: surface paraphrase (rule-based rewrites), typo noise (adjacent-character
swaps), casing noise, homoglyph substitution (Latin→Cyrillic lookalikes), distractor
sentence prepending, and **option shuffling** (with the answer key remapped).
Consistency compares the *chosen option's text*, so a model that tracks content
across a shuffle scores consistent. The overall score is the per-item mean
consistency, with a bootstrap CI over items.

**Caveat.** The paraphraser is intentionally shallow (deterministic, dependency-free);
it understates the robustness gap a strong LLM paraphraser would reveal. Homoglyph
noise is arguably adversarial rather than meaning-preserving — read the
per-perturbation table, not just the aggregate.

## 4. Calibration

Alongside each answer the model states a confidence 0–100. We report:

- **ECE** (expected calibration error): Σ_b (n_b/N) |acc_b − conf_b| over 10
  equal-width bins;
- **Brier score**: mean (confidence − correctness)²;
- **Risk–coverage**: sort by confidence descending; risk(κ) is the error rate among
  the top-κ fraction. **AURC** is the area under this curve — low AURC means the
  model's confidence *ranks* its errors well even if the absolute numbers are off.

**Caveat.** Verbalized confidence ≠ token-level probability; models cluster on round
numbers (80, 90, 95), which coarsens the bins. ECE on ~30 items has sampling noise of
a few points; compare models on the same item sample.

## 5. Contamination probes

A model that memorized the benchmark differs from one that knows the subject:

- **Continuation probe**: given the first ~60% of a question's words, ask for the
  exact continuation. Score = token-F1 similarity to the true remainder **minus** the
  similarity to a control remainder (from a different item) — the subtraction removes
  credit for generic fluency. Verbatim continuations are counted separately.
- **Option-recall probe**: ask the model to reproduce the question's original
  multiple-choice options. The *distractors* carry no semantic signal, so recovering
  them near-verbatim (token-F1 > 0.8) indicates memorization.
- `ngram_overlap` is a utility for screening user-supplied corpora (8-gram default).

The combined risk score is a weighted heuristic:
`0.4·clip(2·gap) + 0.3·exact_rate + 0.3·clip(1.5·option_recall)`.

**Caveat — read this one.** These are *probes*, not proof. A clean model with strong
domain knowledge can partially reconstruct famous items; a contaminated model can
paraphrase away from verbatim recall. Elevated risk means **investigate** (e.g. run
the probes on a held-out private set as a baseline), never "guilty".

## 6. The fingerprint

Radar dimensions, all normalized to [0,1], higher = better:

| Dimension | Definition |
|---|---|
| Ability | Φ(θ̂) — the normal CDF of estimated ability (percentile vs. the calibration population) |
| Robustness | overall perturbation consistency |
| Calibration | 1 − min(2·ECE, 1) |
| Selective risk | 1 − min(2·AURC, 1) |
| Cleanliness | 1 − contamination risk |

The radar is a summary; the JSON/HTML report keeps every underlying interval. When in
doubt, trust the intervals over the shape.

## Reproducibility

All randomness is seeded (item sampling, exposure control, bootstrap, simulated
subjects). Two runs with the same seed, model and bank are identical up to provider
nondeterminism. The `ReplayAdapter` records real model interactions to JSON and
replays them exactly, so published fingerprints can be re-derived offline.
