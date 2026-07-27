---
title: Caliper — LLM Measurement Lab
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
short_description: Adaptive IRT evals, anytime-valid duels, benchmark audits
tags:
  - evaluation
  - leaderboard
  - item-response-theory
  - llm-as-judge
  - robustness
---

# 🔬 Caliper — measurement-science evaluation for LLMs

Point estimates lie. Caliper treats LLM evaluation as **measurement science**:

- **Adaptive ability estimation** — 3PL Item Response Theory with Fisher-information
  item selection: a defensible ability estimate *with a confidence interval* from
  ~35 items instead of thousands.
- **Anytime-valid duels** — two models head to head, stopping the moment a winner is
  statistically decided. Peeking after every item is safe (e-values + Ville's
  inequality); a naive repeated z-test in the same setting fires on ~42% of null
  comparisons, this on ~2%.
- **Uncertainty-aware LLM-as-judge** — every comparison runs in both presentation
  orders × multiple samples; position bias cancels in the average and is reported,
  not hidden.
- **Metamorphic robustness** — paraphrase, typos, homoglyphs, distractors, option
  shuffling: does the answer survive surface changes?
- **Calibration** — ECE, Brier, risk–coverage: does the model know what it doesn't know?
- **Contamination probes** — continuation and option-recall tests for benchmark
  memorization.
- **Benchmark diagnostics** — where has this benchmark *saturated*, what is the
  smallest difference it can detect, and which items are **unfair between model
  families** at matched ability (Differential Item Functioning)?

**Demo mode needs no token**: you set a simulated model's true ability, calibration
skew, robustness and contamination, then watch the instruments recover exactly what
you injected. **Live mode** evaluates any chat model on HF Inference Providers with
your own token (session-only, never stored).

Source, methodology and CLI: **[github.com/aabhimittal/LLM-evaluation](https://github.com/aabhimittal/LLM-evaluation)**
