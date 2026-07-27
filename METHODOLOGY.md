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

## 2b. Judge report card (validation against gold labels)

Raw agreement with humans is a misleading headline: if 70% of gold labels favor
A, a judge that always answers A "agrees 70% of the time" while knowing nothing.
`caliper.judge.report_card` reports instead:

- **Cohen's κ** — agreement corrected for chance;
- **accuracy on gold-decisive pairs** — ties excluded;
- **excess bias** — the judge's rate of picking slot A, or the longer answer,
  *minus the humans' rate*. Humans have a verbosity preference too; only the
  excess is the judge's fault;
- **confidence AUC** — does the judge's own vote agreement rank its correct
  verdicts above its wrong ones? A confidently-wrong judge is worse than an
  uncertain one;
- a **letter grade**, downgraded one notch when the position-flip rate exceeds
  25% — averaging may still recover the right ranking, but individual verdicts
  are then unquotable.

## 2c. Anytime-valid sequential duels

**The problem.** Fixed-sample comparison says: run N items, compute a p-value,
decide. In practice everyone watches results stream in and stops when the
answer looks clear — which silently invalidates the p-value. The effect is not
subtle: `tests/test_sequential.py` measures **~42% false positives** at a
nominal 5% level when a z-test is recomputed after every observation.

**The fix.** Bet instead of test. Wealth starts at `K_0 = 1`; each paired
outcome `X_t ∈ [0,1]` (1 = A wins, 0.5 = tie, 0 = B wins) updates

```
K_t = K_{t-1} · (1 + λ_t (X_t − 0.5))
```

with `λ_t` **predictable** (a function of past data only). Under
`H0: E[X] = 0.5` the wealth is a nonnegative martingale, so Ville's inequality
gives

```
P( ∃t : K_t ≥ 1/α ) ≤ α
```

— simultaneously over all t. Peeking after every item and stopping at the first
crossing is therefore exactly as valid as a fixed-N test. Two one-sided
processes (one per direction) run at α/2 each, so the reported threshold is 2/α.

**Betting rule.** `λ_t` follows a regularised aGRAPA: the running mean and
variance are shrunk toward the null with pseudo-counts, and λ is truncated to
[−1, 1]. Validity holds for *any* predictable λ, so these choices only affect
power — but they matter enormously for it. An earlier truncation at 1.9 let a
single early loss multiply wealth by 0.05, from which the process never
recovered; the fix is documented in the module.

**Confidence sequence.** The win rate carries a Robbins normal-mixture interval,
valid uniformly in time (`sigma = 1/2` by Hoeffding's lemma for [0,1] variables).
Verified empirically: uniform-in-time miscoverage ≤ 1.5% at a nominal 5%.

**Pairing.** Both models see the same items in the same order, and concordant
items (both right or both wrong) score 0.5 and leave wealth untouched — the
sequential analogue of McNemar's test, where only discordant pairs carry
information about which model is better.

**Caveat.** Anytime-validity controls false positives, not sample size. Two
genuinely similar models will simply never cross the threshold; the honest
report is "inconclusive within the budget", which the CLI and Space both say.

## 2d. Differential Item Functioning (auditing the benchmark)

Every other section measures a model; this one measures the *benchmark*. An
item shows **DIF** when two groups of **equal ability** have different chances
of answering it correctly. In educational testing this is the standard method
for finding culturally biased questions; pointed at LLM benchmarks it surfaces
items that reward a training recipe, a tokenizer quirk, a formatting habit or
plain contamination instead of the capability the benchmark claims to measure.

**Estimator.** Mantel–Haenszel. Models are matched into ability strata, a
2×2 table (group × correctness) is built per stratum, and the common odds ratio
is tested:

```
alpha_MH = Σ_k (A_k D_k / T_k) / Σ_k (B_k C_k / T_k)
chi2     = (|Σ A_k − Σ E[A_k]| − 0.5)² / Σ Var(A_k)
```

with A/B = reference correct/wrong and C/D = focal correct/wrong. Effect size
uses the ETS delta scale `Δ = −2.35 ln(alpha_MH)`, **positive when the item
favors the focal group**, classified A (|Δ|<1, negligible), B (1–1.5, moderate)
or C (≥1.5, large), with B/C additionally requiring significance.

Two corrections matter and are implemented:

- **rest-score matching** — the studied item is excluded from its own matching
  criterion, otherwise a biased item contaminates the ability it is matched on
  and its own effect attenuates;
- **two-stage purification** — the ability scale is rebuilt from items the first
  pass judged clean, so a handful of biased items cannot skew everything else.

Perfect separation (a group that never gets an item right) is handled with a
Haldane–Anscombe correction, and the χ² is computed independently of the odds
ratio so separation cannot hide a real effect.

**Validated behaviour** (`tests/test_dif.py`): with 8 rigged items among 80,
the auditor recovers the majority of them with few false alarms, and — the test
that matters — a focal group that is simply **stronger overall** produces no
flags, because ability matching absorbs the gap.

**Caveat.** MH needs many respondents (dozens of models per group) for power,
and matching on observed score is imperfect at the extremes. Flags are
hypotheses to inspect, not verdicts.

## 2e. Benchmark saturation and power

**Test information.** `I(θ) = Σ_i I_i(θ)`, and `SE(θ) = 1/√I(θ)`. Passing a
`test_length` sums only the most informative items at each ability — what an
adaptive test of that length actually achieves, which is the only honest basis
for a saturation claim.

**Saturation ceiling.** The ability above which `SE(θ) > se_target`. Above that
line, models cannot be told apart by this bank and reported differences between
them are noise. A boundary that coincides with the edge of the scanned range is
reported as "no ceiling found", not as a ceiling.

**Power.** The smallest detectable ability gap follows the two-sample formula
on the IRT scale: `Δ = (z_{1−α/2} + z_{power}) · √(SE_A² + SE_B²)`.
For the bundled 250-item bank at 40 items per model this is **1.23 logits** —
i.e. this bank cannot certify small differences at all, no matter how the
scores are reported.

**Adaptive efficiency.** Information from Fisher-information selection versus
random sampling, expressed as the number of random items one adaptive item is
worth (≈2.5× on the bundled bank at θ=0).

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

## 6. RAG grounding (faithfulness & relevance)

Retrieval-augmented systems fail in ways a plain benchmark score cannot see: the
model answers fluently but the answer is not *grounded* in the retrieved passages.
Ragas and TruLens score this with a single LLM-judged faithfulness/relevance number.
Caliper measures the same properties but keeps its house rules — **uncertainty on
every number, and localization instead of an aggregate**.

### Faithfulness

For a `(question, contexts)` sample the model answers using only the contexts. We
then:

1. **decompose** the answer into atomic claims (one factual statement each);
2. **verify** each claim against the contexts with an NLI-style judge
   (SUPPORTED / NOT_SUPPORTED), sampled `n_samples` times so the verifier's
   **self-agreement** is recorded;
3. report `faithfulness = supported_claims / total_claims`, with a **bootstrap CI**
   over claims, and — most usefully — the **list of unsupported claims**. A
   hallucination is thus an *address* (this sample, this sentence), not a lower score.

### Answer relevance

An answer that addresses the question should let you reconstruct the question from
it. We ask the model to generate questions *from its own answer*, embed them and the
original question (HF `feature-extraction`, or the simulated hashed embedding), and
report the mean cosine similarity with a CI over the generated questions.

### Context precision

The fraction of retrieved passages the model judges genuinely useful for the
question. Low precision means the retriever is padding the context with distractors.

### Ground truth

`SimulatedRAGSubject` injects a known `hallucination_rate`, `answer_relevance` and
`context_precision`. Supported claims quote the context verbatim, so the token-overlap
verifier marks exactly the fabricated claims as unsupported; the suite recovers
`supported_fraction ≈ 1 − hallucination_rate` (see `tests/test_rag.py`).

**Caveat — read this one.** The verifier is itself a model; a weak verifier
mislabels entailment, which is why **agreement across samples is reported** and why
you should read the unsupported-claims list, not just the aggregate. The answer- and
context-relevance numbers inherit the quality of the embedding / relevance judge.
Faithfulness here is grounding in the *retrieved* context, not truth in the world: a
model faithfully repeating a wrong passage scores high. For the *standard* Ragas /
TruLens numbers, `caliper.rag.bridge` wraps those libraries (optional `[rag]` extra).

## 7. Auditing the verifier — and correcting for it

Faithfulness is measured *by a model*. That verifier is an instrument with its own
error rates, and a raw faithfulness score inherits them silently. The bias is
**systematic**: it does not shrink as you add samples.

### Measuring the verifier

Controls are built from the bank itself, so no extra labelled data is required:

- **positive controls** — sentences taken from a sample's *own* passages. They are
  entailed by construction, so a NOT_SUPPORTED verdict is a miss. The rate of correct
  verdicts estimates **sensitivity** (*se*).
- **negative controls** — sentences from a *different* sample's passages: true
  statements about the world that these passages do not entail. This is exactly the
  distinction faithfulness rests on. The rate of correct rejections estimates
  **specificity** (*sp*).

Both come with bootstrap CIs. Two pathologies are reported alongside: the
**order-flip rate** (does the verdict change when the passages are reordered?) and
the **unparseable rate**.

### Correcting the score

For a test with known error rates, the observed positive rate relates to the true
prevalence *p* by `p_obs = p·se + (1−p)·(1−sp)`. Inverting gives the Rogan–Gladen
(1978) estimator:

```
p_corrected = (p_obs + sp − 1) / (se + sp − 1)
```

Caliper reports this as `faithfulness_corrected`, with an interval obtained by
propagating the endpoints of the *se*/*sp* intervals — so the corrected interval
widens as the verifier becomes less trustworthy, which is the honest behaviour.

**When the correction is refused.** If `se + sp ≤ 1` the verifier is no better than
chance and the correction is undefined; Caliper reports `usable = False` and returns
no corrected number rather than manufacturing precision. This matters: a verifier
that rubber-stamps everything has *sp → 0* and yields a faithfulness score near 1.0
for any model at all.

**Validation.** `tests/test_rag_audit.py` injects known *se*/*sp* into the simulated
fact-checker, confirms the audit recovers them, and asserts that — averaged over
seeds — the corrected estimate lands closer to the true faithfulness than the raw
one. With a rubber-stamping verifier (se 0.95, sp 0.55) the correction roughly halves
the error.

**Caveat.** The correction assumes the controls are representative of the claims
actually being graded. Real answer claims are paraphrases, not verbatim sentences, so
measured sensitivity is an optimistic bound; treat the corrected value as *"what the
score would be with a perfect verifier"*, and read the raw and corrected numbers
together.

## 8. Was the answer earned by retrieval?

High faithfulness does not prove the retrieved passages did any work. A model can
recite an answer from parametric memory that happens to agree with the context and
score perfectly — you could switch the retriever off and the benchmark would not
notice. In Caliper's own demo this shows up starkly: a model with
`context_reliance = 0` scores **faithfulness 0.90** — which any conventional RAG
evaluator would call excellent — while **`earned_by_retrieval` is 0.00**. The claims
really are entailed by the passages; the passages just had nothing to do with
producing them.

This probe asks the counterfactual directly, re-answering each question under
altered context:

| Condition | Metric | Reading |
|---|---|---|
| context removed (closed book) | `parametric_leakage` = sim(full, closed) | high ⇒ the model already "knew" it; retrieval is decorative |
| context swapped for another question's | `context_sensitivity` = 1 − sim(full, foreign) | low ⇒ the model ignores what it retrieves |
| an irrelevant passage added | `distractor_stability` = sim(full, polluted) | low ⇒ one bad retrieval hit derails the answer |

Similarity is cosine over the adapter's embeddings. The headline
`earned_by_retrieval` is `0.5·context_sensitivity + 0.5·(1 − parametric_leakage)`.

This is the retrieval analogue of the contamination probes in §5: same spirit (ask
what the score would be if the thing under test were removed), and the same caveat —
these are **probes, not proof**. A question whose answer is genuinely common
knowledge will show high leakage without anything being wrong; compare models on the
same bank rather than reading one number absolutely.

## 9. The fingerprint

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
