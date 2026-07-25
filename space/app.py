"""Caliper — LLM measurement lab (Hugging Face Space).

Demo mode needs no token: it evaluates simulated subjects with *known*
ability, calibration skew, robustness and contamination, so you can watch
each statistical instrument detect the pathology you injected. Live mode
evaluates any chat model on HF Inference Providers with your own token
(used only for this session, never stored).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np

from caliper.adapters import SimulatedJudge, SimulatedRAGSubject, SimulatedSubject
from caliper.diagnostics import (
    adaptive_efficiency,
    bank_health,
    minimum_detectable_difference,
)
from caliper.dif import detect_dif
from caliper.irt import p_correct, run_adaptive
from caliper.judge import PairwiseJudge
from caliper.rag import RagBank, evaluate_rag
from caliper.report import render_html, render_rag_html, run_fingerprint
from caliper.report.html import _line_chart, _radar_svg
from caliper.robustness import evaluate_robustness
from caliper.robustness.perturb import PERTURBATIONS
from caliper.sequential import run_item_duel
from caliper.types import ItemBank

BANK = ItemBank.bundled()
RAG_BANK = RagBank.bundled()

CSS = """
.svgbox { background: var(--background-fill-primary); border-radius: 8px; padding: 8px; }
.small-note { color: var(--body-text-color-subdued); font-size: 0.85em; }
"""

# The report SVGs reference CSS custom properties that only exist inside the
# standalone report document, and Gradio's HTML component sanitizes <style>
# blocks — so substitute concrete values that read well on light and dark.
_SVG_SUBS = [
    ("var(--series-soft)", "rgba(57,135,229,0.18)"),
    ("var(--band)", "rgba(57,135,229,0.22)"),
    ("var(--series)", "#3987e5"),
    ("var(--grid)", "rgba(137,135,129,0.35)"),
    ("var(--axis)", "#898781"),
    ("var(--ref)", "#898781"),
    ('<text class="val"', '<text fill="currentColor" font-weight="600"'),
    ("<svg ", '<svg fill="#898781" font-family="system-ui, sans-serif" font-size="11" '),
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_JUDGE = "meta-llama/Llama-3.3-70B-Instruct"


def _subject(mode: str, model_id: str, token: str, theta: float, robustness: float,
             skew: float, contaminated: bool):
    if mode.startswith("Demo"):
        return SimulatedSubject(
            theta=theta, bank=BANK, robustness=robustness,
            calibration_skew=skew, contaminated=contaminated, seed=0,
        )
    if not model_id.strip():
        raise gr.Error("Enter a model id for live mode.")
    from caliper.adapters.hf_inference import HFInferenceAdapter

    return HFInferenceAdapter(model=model_id.strip(), token=token.strip() or None)


def _wrap_svg(svg: str) -> str:
    for needle, replacement in _SVG_SUBS:
        svg = svg.replace(needle, replacement)
    return f'<div class="svgbox" style="max-width:660px;margin:0 auto">{svg}</div>'


# ---------------------------------------------------------------- adaptive

def ui_adaptive(mode, model_id, token, theta, robustness, skew, contaminated,
                max_items, progress=gr.Progress()):  # noqa: B008
    try:
        adapter = _subject(mode, model_id, token, theta, robustness, skew, contaminated)
    except Exception as e:  # noqa: BLE001
        yield gr.skip(), f"**Error:** {e}", gr.skip()
        return
    trajectory, band = [], []
    log_lines = []
    try:
        for state in run_adaptive(adapter, BANK, max_items=int(max_items), seed=0):
            e = state.estimate
            trajectory.append((state.step, e.theta))
            band.append((state.step, e.theta - 1.96 * e.se, e.theta + 1.96 * e.se))
            mark = "✓" if state.result.correct else "✗"
            log_lines.append(
                f"`{state.item.id}` {mark} → θ = {e.theta:+.2f} ± {1.96 * e.se:.2f}"
            )
            svg = _line_chart(trajectory, band, x_label="items administered",
                              y_label="θ", aria="Ability convergence")
            status = (
                f"**Step {state.step}** — θ = **{e.theta:+.2f}**, "
                f"95% CI [{e.ci95[0]:+.2f}, {e.ci95[1]:+.2f}]\n\n"
                + "\n".join(f"- {line}" for line in log_lines[-6:])
            )
            yield _wrap_svg(svg), status, gr.skip()
        final = {
            "theta": round(trajectory[-1][1], 3),
            "se": round((band[-1][2] - band[-1][1]) / (2 * 1.96), 3),
            "ci95": [round(band[-1][1], 3), round(band[-1][2], 3)],
            "items_used": len(trajectory),
            "bank_size": len(BANK),
            "note": "Reliable ability estimate from a fraction of the benchmark — "
                    "that is the point of adaptive testing.",
        }
        yield gr.skip(), gr.skip(), final
    except Exception as e:  # noqa: BLE001
        yield gr.skip(), f"**Error during evaluation:** {e}", gr.skip()


# ---------------------------------------------------------------- judge

def ui_judge(mode, judge_model, token, prompt, resp_a, resp_b,
             accuracy, position_bias, verbosity_bias):
    if mode.startswith("Demo"):
        adapter = SimulatedJudge(
            accuracy=accuracy, position_bias=position_bias,
            verbosity_bias=verbosity_bias, seed=0,
        )
    else:
        if not judge_model.strip():
            raise gr.Error("Enter a judge model id for live mode.")
        from caliper.adapters.hf_inference import HFInferenceAdapter

        adapter = HFInferenceAdapter(model=judge_model.strip(), token=token.strip() or None)
    judge = PairwiseJudge(adapter, n_samples=3)
    try:
        verdict = judge.compare(prompt, resp_a, resp_b)
    except Exception as e:
        raise gr.Error(f"Judge call failed: {e}") from e
    flip_note = (
        "⚠️ verdict FLIPPED when response order was swapped — do not trust a "
        "single-order judgment here"
        if verdict.position_flip
        else "verdict stable across both presentation orders"
    )
    summary = (
        f"### Winner: **{verdict.winner}**\n\n"
        f"- debiased P(A wins) = **{verdict.p_a_wins:.2f}** "
        f"(averaged over both orders × 3 samples)\n"
        f"- vote agreement (confidence) = **{verdict.confidence:.2f}**\n"
        f"- {flip_note}\n"
        f"- {verdict.n_votes} parseable votes, {verdict.unparseable} unparseable"
    )
    return summary, verdict.votes


# ---------------------------------------------------------------- robustness

def ui_perturb_preview(question: str):
    rows = ["| perturbation | text |", "|---|---|", f"| *original* | {question} |"]
    for name, fn in PERTURBATIONS.items():
        rows.append(f"| {name} | {fn(question, np.random.default_rng(3))} |")
    return "\n".join(rows)


def ui_robustness(mode, model_id, token, theta, robustness, skew, contaminated, n_items):
    try:
        adapter = _subject(mode, model_id, token, theta, robustness, skew, contaminated)
        report = evaluate_robustness(adapter, BANK, n_items=int(n_items), seed=0)
    except Exception as e:
        raise gr.Error(f"Robustness run failed: {e}") from e
    lines = [
        (
            f"### Consistency: **{report.overall_consistency:.2f}** "
            f"(95% CI {report.ci95[0]:.2f} – {report.ci95[1]:.2f}, {report.n_items} items)"
        ),
        "",
        "| perturbation | consistency |",
        "|---|---|",
    ]
    for k, v in sorted(report.by_perturbation.items(), key=lambda kv: kv[1]):
        lines.append(f"| {k} | {v:.2f} |")
    flips = report.flips[:6]
    if flips:
        lines += ["", "**Example flips** (same question, different surface):", "",
                  "| item | perturbation | baseline answer | perturbed answer |", "|---|---|---|---|"]
        for f in flips:
            lines.append(
                f"| {f['item_id']} | {f['perturbation']} | "
                f"{str(f['baseline'])[:40]} | {str(f['perturbed'])[:40]} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------- fingerprint

def ui_fingerprint(mode, model_id, token, theta, robustness, skew, contaminated,
                   progress=gr.Progress()):  # noqa: B008
    try:
        adapter = _subject(mode, model_id, token, theta, robustness, skew, contaminated)
    except Exception as e:
        raise gr.Error(str(e)) from e
    live = not mode.startswith("Demo")
    budgets = {
        "adaptive_max_items": 25 if live else 40,
        "robustness_items": 5 if live else 10,
        "calibration_items": 15 if live else 30,
        "contamination_items": 6 if live else 12,
    }
    stages = ["adaptive ability", "robustness", "calibration", "contamination"]
    stage_iter = iter(np.linspace(0.1, 0.9, len(stages)))

    def report_progress(msg: str):
        try:
            progress(next(stage_iter), desc=msg)
        except StopIteration:
            pass

    try:
        fp = run_fingerprint(adapter, BANK, seed=0, progress=report_progress, **budgets)
    except Exception as e:
        raise gr.Error(f"Fingerprint run failed: {e}") from e

    radar = _wrap_svg(_radar_svg(fp.dimensions()))
    html_report = render_html(fp)
    tmp_dir = Path(tempfile.mkdtemp(prefix="caliper-"))
    json_path = tmp_dir / "fingerprint.json"
    html_path = tmp_dir / "fingerprint.html"
    json_path.write_text(fp.to_json(), encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")

    est = fp.ability.estimate
    summary = (
        f"**{fp.model_name}** — θ = {est.theta:+.2f} "
        f"[{est.ci95[0]:+.2f}, {est.ci95[1]:+.2f}] on {est.n_items} adaptive items · "
        f"robustness {fp.robustness.overall_consistency:.2f} · "
        f"ECE {fp.calibration.ece:.2f} · contamination risk {fp.contamination.risk:.2f}"
    )
    iframe = (
        '<iframe style="width:100%;height:900px;border:none;border-radius:8px" '
        f'srcdoc="{html_report.replace("&", "&amp;").replace(chr(34), "&quot;")}"></iframe>'
    )
    return radar, summary, iframe, [str(json_path), str(html_path)]


# ---------------------------------------------------------------- rag

def ui_rag(mode, model_id, token, halluc, ans_rel, ctx_prec, ctx_reliance,
           ver_se, ver_sp, n_samples, progress=gr.Progress()):  # noqa: B008
    if mode.startswith("Demo"):
        adapter = SimulatedRAGSubject(
            hallucination_rate=halluc, answer_relevance=ans_rel,
            context_precision=ctx_prec, context_reliance=ctx_reliance,
            verifier_sensitivity=ver_se, verifier_specificity=ver_sp,
            bank=RAG_BANK, seed=0,
        )
    else:
        if not model_id.strip():
            raise gr.Error("Enter a model id for live mode.")
        from caliper.adapters.hf_inference import HFInferenceAdapter

        adapter = HFInferenceAdapter(model=model_id.strip(), token=token.strip() or None)

    n = int(n_samples)
    steps = iter(np.linspace(0.05, 0.95, max(n, 1)))

    def report_progress(_msg):
        try:
            progress(next(steps), desc=_msg)
        except StopIteration:
            pass

    try:
        report = evaluate_rag(adapter, RAG_BANK, n_samples=n, seed=0,
                              n_boot=300, with_audit=True, with_attribution=True,
                              progress=report_progress)
    except Exception as e:
        raise gr.Error(f"RAG run failed: {e}") from e

    f_lo, f_hi = report.faithfulness_ci95
    summary = (
        f"### Faithfulness: **{report.faithfulness:.2f}** "
        f"(95% CI {f_lo:.2f} – {f_hi:.2f}, {report.n_claims} claims)\n\n"
        f"- answer relevance **{report.answer_relevance:.2f}** · "
        f"context precision **{report.context_precision:.2f}**\n"
        f"- **{report.n_unsupported_claims}** hallucinated claims localized · "
        f"verifier self-agreement {report.mean_verifier_agreement:.2f}\n"
    )
    if report.verifier is not None:
        v = report.verifier
        if report.faithfulness_corrected is not None:
            shift = report.faithfulness_corrected - report.faithfulness
            summary += (
                f"- **verifier-corrected faithfulness "
                f"{report.faithfulness_corrected:.2f}** ({shift:+.2f}) — the grader "
                f"itself scores se {v.sensitivity:.2f} / sp {v.specificity:.2f}\n"
            )
        else:
            summary += (
                f"- ⚠️ the verifier carries **no usable signal** "
                f"(se {v.sensitivity:.2f} + sp {v.specificity:.2f} ≤ 1): "
                "faithfulness here cannot be trusted or corrected\n"
            )
    if report.attribution is not None:
        a = report.attribution
        summary += (
            f"- **earned by retrieval {a.earned_by_retrieval:.2f}** — parametric "
            f"leakage {a.parametric_leakage:.2f}, context sensitivity "
            f"{a.context_sensitivity:.2f}\n"
        )
    summary += (
        "\n*Unlike a single Ragas/TruLens score, every number carries its "
        "uncertainty, each unsupported claim is pinned to its sample, the "
        "grader is itself audited, and retrieval has to prove it did the work.*"
    )
    if report.unsupported_examples:
        rows = ["", "**Localized hallucinations** (claims not entailed by the context):",
                "", "| sample | unsupported claim |", "|---|---|"]
        for e in report.unsupported_examples[:8]:
            rows.append(f"| {e['sample_id']} | {str(e['claim'])[:90]} |")
        summary += "\n" + "\n".join(rows)

    html_report = render_rag_html(report, model_name=adapter.name,
                                  bank_name=RAG_BANK.name)
    iframe = (
        '<iframe style="width:100%;height:720px;border:none;border-radius:8px" '
        f'srcdoc="{html_report.replace("&", "&amp;").replace(chr(34), "&quot;")}"></iframe>'
    )
    return summary, iframe
# ---------------------------------------------------------------- duel

def ui_duel(mode, model_id, token, theta, robustness, skew, contaminated,
            theta_b, alpha, max_items):
    """Anytime-valid head-to-head, streaming after every item."""
    if mode.startswith("Demo"):
        adapter_a = SimulatedSubject(theta=theta, bank=BANK, robustness=robustness,
                                     calibration_skew=skew, seed=0,
                                     name=f"A (θ={theta:+.1f})")
        adapter_b = SimulatedSubject(theta=theta_b, bank=BANK, seed=1,
                                     name=f"B (θ={theta_b:+.1f})")
    else:
        ids = [m.strip() for m in model_id.split(",")]
        if len(ids) != 2:
            yield gr.skip(), (
                "**Live duel needs two model ids** — put both in the model box, "
                "comma separated (`org/model-a, org/model-b`)."
            ), gr.skip()
            return
        from caliper.adapters.hf_inference import HFInferenceAdapter

        adapter_a = HFInferenceAdapter(model=ids[0], token=token.strip() or None)
        adapter_b = HFInferenceAdapter(model=ids[1], token=token.strip() or None)

    rates, bands = [], []
    last = None
    try:
        for state in run_item_duel(
            adapter_a, adapter_b, BANK,
            alpha=float(alpha), max_items=int(max_items), seed=0,
        ):
            rates.append((state.step, state.win_rate))
            bands.append((state.step, state.ci95[0], state.ci95[1]))
            last = state
            if state.step % 3 == 0 or state.decided:
                chart = _line_chart(
                    rates, bands,
                    x_label="items", y_label="P(A wins)",
                    y_range=(0.0, 1.0), aria="Paired win rate with confidence sequence",
                )
                lo, hi = state.ci95
                verdict = (
                    f"### 🏆 Decided at item {state.step}: **{adapter_a.name if state.winner == 'A' else adapter_b.name}**\n"
                    f"Stopped early — and the {int((1 - float(alpha)) * 100)}% guarantee still holds."
                    if state.decided else
                    f"### Running — item {state.step}\nNo winner yet; keep going."
                )
                status = (
                    f"{verdict}\n\n"
                    f"- paired win rate for A: **{state.win_rate:.3f}** "
                    f"(confidence sequence [{lo:.2f}, {hi:.2f}])\n"
                    f"- evidence for A: **{state.e_value_a:,.1f}** · "
                    f"evidence for B: **{state.e_value_b:,.1f}** "
                    f"(threshold {2 / float(alpha):,.0f})\n"
                    f"- ties so far carry no evidence — only items one model got "
                    "right and the other got wrong move the needle"
                )
                yield _wrap_svg(chart), status, gr.skip()
    except Exception as e:  # noqa: BLE001
        yield gr.skip(), f"**Duel failed:** {e}", gr.skip()
        return

    if last is None:
        yield gr.skip(), "**No items were administered.**", gr.skip()
        return
    yield gr.skip(), gr.skip(), {
        "models": {"A": adapter_a.name, "B": adapter_b.name},
        "items_used": last.step,
        "win_rate_a": round(last.win_rate, 4),
        "confidence_sequence_95": [round(x, 4) for x in last.ci95],
        "e_value_a": round(last.e_value_a, 3),
        "e_value_b": round(last.e_value_b, 3),
        "decided": last.decided,
        "winner": (
            adapter_a.name if last.winner == "A"
            else adapter_b.name if last.winner == "B" else None
        ),
        "note": (
            "Peeking after every single item is safe here: the e-process "
            "controls type-I error under any stopping rule."
        ),
    }


# ---------------------------------------------------------------- diagnostics

def ui_diagnose(se_target, test_length, theta):
    test_length = int(test_length) if test_length else None
    health = bank_health(BANK, se_target=float(se_target), test_length=test_length)
    power = minimum_detectable_difference(
        BANK, n_items=test_length or 40, theta=float(theta)
    )
    efficiency = adaptive_efficiency(
        BANK, theta=float(theta), n_items=test_length or 30
    )
    curve = [(row["theta"], row["se"]) for row in health.curve]
    chart = _line_chart(
        curve, None, x_label="ability θ", y_label="standard error",
        y_range=(0.0, min(1.5, max(s for _, s in curve) * 1.1)),
        aria="Standard error across the ability scale",
    )
    text = (
        f"### {health.verdict}\n\n"
        f"| | |\n|---|---|\n"
        f"| items in bank | {health.n_items} |\n"
        f"| test length assumed | {test_length or 'whole bank'} |\n"
        f"| best achievable SE | {health.best_se:.3f} at θ {health.peak_theta:+.2f} |\n"
        f"| usable θ range (SE ≤ {health.se_target}) | "
        f"{health.usable_range if health.usable_range else 'none'} |\n"
        f"| smallest detectable θ gap | {power.minimum_detectable_difference:.2f} logits |\n"
        f"| adaptive efficiency | {efficiency['efficiency_ratio']:.2f}× per item |\n\n"
        f"{power.interpretation}\n\n{efficiency['interpretation']}"
    )
    return _wrap_svg(chart), text


# ---------------------------------------------------------------- DIF

def ui_dif(n_models, n_items, n_dif, dif_strength, focal_advantage, seed):
    """Simulate two model families, inject item bias, and try to catch it."""
    rng = np.random.default_rng(int(seed))
    n_models, n_items, n_dif = int(n_models), int(n_items), int(n_dif)
    half = n_models // 2
    groups = np.array([0] * half + [1] * (n_models - half))
    theta = np.where(
        groups == 1,
        rng.normal(float(focal_advantage), 1, n_models),
        rng.normal(0.0, 1, n_models),
    )
    a = rng.lognormal(0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    c = np.full(n_items, 0.25)
    P = p_correct(theta[:, None], a[None, :], b[None, :], c[None, :])
    injected = {int(j) for j in rng.choice(n_items, n_dif, replace=False)}
    for j in injected:
        P[groups == 1, j] = p_correct(
            theta[groups == 1], a[j], b[j] - float(dif_strength), c[j]
        )
    X = (rng.random(P.shape) < P).astype(float)

    report = detect_dif(X, groups, BANK, n_strata=5,
                        reference_name="family X", focal_name="family Y")
    flagged = {it.index for it in report.flagged}
    hits = len(flagged & injected)
    false_alarms = len(flagged - injected)

    lines = [
        (
            f"### Injected {len(injected)} biased items · caught **{hits}** · "
            f"false alarms **{false_alarms}** (of {n_items - len(injected)} clean items)"
        ),
        "",
        (
            f"Family Y is genuinely stronger (mean θ {float(focal_advantage):+.1f}). "
            "That ability gap alone must **not** register as bias — only items where "
            "the gap survives ability matching are flagged."
        ),
        "",
        "| item | ETS Δ | class | favors | p | acc X | acc Y | truly biased |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for it in report.worst(12):
        lines.append(
            f"| `{it.item_id[:26]}` | {it.delta:+.2f} | {it.classification} | "
            f"{it.favors} | {it.p_value:.4f} | {it.p_reference:.2f} | "
            f"{it.p_focal:.2f} | {'✅ yes' if it.index in injected else '—'} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Caliper — LLM measurement lab") as demo:
    gr.Markdown(
        "# 🔬 Caliper — measurement-science evaluation for LLMs\n"
        "Point estimates lie. Caliper measures models the way psychometrics measures "
        "people: **adaptive IRT ability estimation** with confidence intervals, "
        "**anytime-valid model duels** that stop the moment a winner is clear, "
        "**bias-audited LLM-as-judge**, **metamorphic robustness**, **confidence "
        "calibration**, **contamination probes** — and diagnostics for the "
        "*benchmark itself*: saturation ceilings and per-item fairness between "
        "model families. Every number with its uncertainty. "
        "[Source & methodology](https://github.com/aabhimittal/LLM-evaluation)"
    )
    with gr.Accordion("Model under test", open=True):
        mode = gr.Radio(
            ["Demo (simulated subject, no token needed)",
             "Live (HF Inference Providers)"],
            value="Demo (simulated subject, no token needed)", label="Mode",
        )
        with gr.Row():
            model_id = gr.Textbox(label="HF model id (live mode)", value=DEFAULT_MODEL)
            token = gr.Textbox(label="Your HF token (live mode, session-only)",
                               type="password", value="")
        with gr.Row():
            theta = gr.Slider(-2.5, 2.5, value=0.6, step=0.1,
                              label="Demo: true ability θ")
            robustness_knob = gr.Slider(0.4, 1.0, value=0.92, step=0.02,
                                        label="Demo: true robustness")
            skew = gr.Slider(0.2, 2.0, value=1.0, step=0.1,
                             label="Demo: calibration skew (<1 = overconfident)")
            contaminated = gr.Checkbox(label="Demo: benchmark-contaminated", value=False)
        gr.Markdown(
            "In **demo mode** you set the ground truth, then watch the instruments "
            "recover it. In **live mode** the same instruments run against a real "
            "model via the `chat-completion` task (your token, your choice of model).",
            elem_classes=["small-note"],
        )

    subject_inputs = [mode, model_id, token, theta, robustness_knob, skew, contaminated]

    with gr.Tab("📈 Adaptive ability (IRT)"):
        gr.Markdown(
            "Items are chosen one at a time to **maximize Fisher information** at the "
            "current ability estimate — the θ interval shrinks with a fraction of the "
            "items a fixed benchmark needs. Item bank: 250 real ARC-Challenge "
            "questions (demo-calibrated parameters; recalibrate with `caliper calibrate`)."
        )
        max_items = gr.Slider(10, 50, value=35, step=5, label="Item budget")
        adaptive_button = gr.Button("Run adaptive evaluation", variant="primary")
        adaptive_chart = gr.HTML(label="θ convergence")
        adaptive_status = gr.Markdown()
        adaptive_json = gr.JSON(label="Final estimate")
        adaptive_button.click(
            ui_adaptive, inputs=subject_inputs + [max_items],
            outputs=[adaptive_chart, adaptive_status, adaptive_json],
        )

    with gr.Tab("⚖️ Judge lab"):
        gr.Markdown(
            "Every comparison runs in **both presentation orders × 3 samples**. "
            "Position bias cancels in the average and shows up explicitly as a flip "
            "flag. In demo mode, inject judge pathologies and watch them get caught."
        )
        with gr.Row():
            judge_model = gr.Textbox(label="Judge model (live mode)", value=DEFAULT_JUDGE)
        with gr.Row():
            accuracy = gr.Slider(0.5, 1.0, value=0.9, step=0.05,
                                 label="Demo judge: accuracy")
            position_bias = gr.Slider(0.0, 0.8, value=0.0, step=0.05,
                                      label="Demo judge: position bias")
            verbosity_bias = gr.Slider(0.0, 0.8, value=0.0, step=0.05,
                                       label="Demo judge: verbosity bias")
        judge_prompt = gr.Textbox(
            label="Prompt", value="Explain why the sky is blue in two sentences.")
        with gr.Row():
            resp_a = gr.Textbox(
                label="Response A", lines=4,
                value="Sunlight scatters off air molecules, and blue light scatters "
                      "the most because of its short wavelength (Rayleigh scattering). "
                      "So the sky we see is dominated by scattered blue light.",
            )
            resp_b = gr.Textbox(
                label="Response B", lines=4,
                value="The sky is blue because it reflects the ocean.",
            )
        judge_button = gr.Button("Judge with debiasing", variant="primary")
        judge_summary = gr.Markdown()
        judge_votes = gr.JSON(label="Individual votes")
        judge_button.click(
            ui_judge,
            inputs=[mode, judge_model, token, judge_prompt, resp_a, resp_b,
                    accuracy, position_bias, verbosity_bias],
            outputs=[judge_summary, judge_votes],
        )

    with gr.Tab("🌀 Robustness"):
        gr.Markdown(
            "**Metamorphic testing**: the same question under meaning-preserving "
            "perturbations — paraphrase, typos, casing, homoglyphs, distractor "
            "sentences, option shuffling. A trustworthy model answers identically."
        )
        preview_q = gr.Textbox(
            label="Preview a perturbation set",
            value="Which of the following is most likely the reason the planet "
                  "rotates faster after the impact?",
        )
        preview_button = gr.Button("Preview perturbations")
        preview_out = gr.Markdown()
        preview_button.click(ui_perturb_preview, inputs=[preview_q], outputs=[preview_out])
        n_items = gr.Slider(4, 20, value=8, step=1, label="Items to test")
        robustness_button = gr.Button("Run robustness suite", variant="primary")
        robustness_out = gr.Markdown()
        robustness_button.click(
            ui_robustness, inputs=subject_inputs + [n_items], outputs=[robustness_out]
        )

    with gr.Tab("🫆 Full fingerprint"):
        gr.Markdown(
            "Runs everything — adaptive ability, robustness, calibration "
            "(ECE + risk-coverage), contamination probes — and assembles the "
            "**model fingerprint**: five dimensions, each with uncertainty. "
            "Live mode makes ~100–150 model calls; expect a few minutes."
        )
        fingerprint_button = gr.Button("Run full fingerprint", variant="primary")
        fingerprint_summary = gr.Markdown()
        fingerprint_radar = gr.HTML()
        fingerprint_files = gr.File(label="Download report (JSON + HTML)",
                                    file_count="multiple")
        fingerprint_iframe = gr.HTML()
        fingerprint_button.click(
            ui_fingerprint, inputs=subject_inputs,
            outputs=[fingerprint_radar, fingerprint_summary, fingerprint_iframe,
                     fingerprint_files],
        )

    with gr.Tab("🔗 RAG grounding"):
        gr.Markdown(
            "**Faithfulness & relevance the measurement-science way.** Each answer is "
            "split into atomic claims; every claim is verified against the retrieved "
            "context by a sampled NLI judge. You get faithfulness, answer relevance and "
            "context precision — each with a **95% confidence interval** — plus the exact "
            "list of **unsupported (hallucinated) claims**.\n\n"
            "Two things no other RAG evaluator does:\n"
            "1. **The grader gets graded.** The verifier is measured against known-truth "
            "controls, and faithfulness is *bias-corrected* for its sensitivity and "
            "specificity (Rogan–Gladen). Drag the verifier sliders below and watch the raw "
            "score go wrong while the corrected one holds.\n"
            "2. **Retrieval has to prove it did the work.** The question is re-asked with "
            "the context removed, swapped and polluted. Set *context reliance* to 0 and "
            "the model answers from memory — faithfulness stays high, but *earned by "
            "retrieval* collapses to 0."
        )
        with gr.Row():
            rag_halluc = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                   label="Demo: true hallucination rate")
            rag_ans_rel = gr.Slider(0.0, 1.0, value=0.85, step=0.05,
                                    label="Demo: answer relevance")
            rag_ctx_prec = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                     label="Demo: context precision")
        with gr.Row():
            rag_ctx_reliance = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                         label="Demo: context reliance (0 = answers from memory)")
            rag_ver_se = gr.Slider(0.3, 1.0, value=1.0, step=0.05,
                                   label="Demo verifier: sensitivity")
            rag_ver_sp = gr.Slider(0.3, 1.0, value=1.0, step=0.05,
                                   label="Demo verifier: specificity")
        rag_n = gr.Slider(4, len(RAG_BANK), value=min(10, len(RAG_BANK)), step=1,
                          label="RAG samples to score")
        rag_button = gr.Button("Run RAG grounding suite", variant="primary")
        rag_summary = gr.Markdown()
        rag_iframe = gr.HTML()
        rag_button.click(
            ui_rag,
            inputs=[mode, model_id, token, rag_halluc, rag_ans_rel, rag_ctx_prec,
                    rag_ctx_reliance, rag_ver_se, rag_ver_sp, rag_n],
            outputs=[rag_summary, rag_iframe],
        )

    with gr.Tab("⚔️ Sequential duel"):
        gr.Markdown(
            "Two models, same items, **stop the moment the winner is clear**.\n\n"
            "Fixed-N evaluation wastes calls when the answer arrives early — and "
            "peeking at a p-value as results stream in inflates false positives to "
            "~40%. This uses an **e-process**: wealth starts at 1 and is bet on "
            "each item; by Ville's inequality the type-I error stays ≤ α *under "
            "any stopping rule*. Watch after every item, stop whenever you like, "
            "the guarantee holds."
        )
        with gr.Row():
            theta_b = gr.Slider(-2.5, 2.5, value=-1.2, step=0.1,
                                label="Demo: ability θ of model B")
            duel_alpha = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="α")
            duel_max = gr.Slider(20, 250, value=200, step=10, label="Item budget")
        gr.Markdown(
            "*Live mode: put **two** comma-separated model ids in the model box "
            "above.*", elem_classes=["small-note"],
        )
        duel_button = gr.Button("Run sequential duel", variant="primary")
        duel_chart = gr.HTML()
        duel_status = gr.Markdown()
        duel_json = gr.JSON(label="Final verdict")
        duel_button.click(
            ui_duel,
            inputs=subject_inputs + [theta_b, duel_alpha, duel_max],
            outputs=[duel_chart, duel_status, duel_json],
        )

    with gr.Tab("🩺 Benchmark diagnostics"):
        gr.Markdown(
            "Every other tab measures a *model*. This one measures the "
            "**benchmark**: where on the ability scale can it still tell models "
            "apart, how small a difference can it detect, and is adaptive "
            "selection actually paying off? Above the saturation ceiling, "
            "reported differences between frontier models are noise."
        )
        with gr.Row():
            diag_se = gr.Slider(0.1, 0.6, value=0.3, step=0.05,
                                label="Target standard error")
            diag_len = gr.Slider(0, 250, value=40, step=5,
                                 label="Test length (0 = whole bank)")
            diag_theta = gr.Slider(-2.0, 2.5, value=0.0, step=0.1,
                                   label="Ability of interest θ")
        diag_button = gr.Button("Diagnose the benchmark", variant="primary")
        diag_chart = gr.HTML()
        diag_text = gr.Markdown()
        diag_button.click(ui_diagnose, inputs=[diag_se, diag_len, diag_theta],
                          outputs=[diag_chart, diag_text])

    with gr.Tab("🔎 Item bias (DIF)"):
        gr.Markdown(
            "**Differential Item Functioning** asks whether an item is unfair "
            "between two model families *at matched ability* — the standard "
            "psychometric tool for finding culturally biased exam questions, "
            "pointed at LLM benchmarks. Items that reward a training recipe, a "
            "tokenizer quirk or plain contamination show up here.\n\n"
            "Below, family Y is made genuinely stronger **and** a few items are "
            "secretly rigged in its favour. A correct auditor catches the rigged "
            "items without flagging the honest ability gap."
        )
        with gr.Row():
            dif_models = gr.Slider(40, 200, value=100, step=10, label="Models")
            dif_items = gr.Slider(20, 150, value=80, step=10, label="Items")
            dif_count = gr.Slider(0, 20, value=8, step=1, label="Rigged items")
        with gr.Row():
            dif_strength = gr.Slider(0.5, 3.0, value=1.5, step=0.1,
                                     label="Bias strength (logits)")
            dif_advantage = gr.Slider(0.0, 2.0, value=0.8, step=0.1,
                                      label="Family Y's real ability edge")
            dif_seed = gr.Slider(0, 50, value=4, step=1, label="Seed")
        dif_button = gr.Button("Audit the item bank", variant="primary")
        dif_out = gr.Markdown()
        dif_button.click(
            ui_dif,
            inputs=[dif_models, dif_items, dif_count, dif_strength,
                    dif_advantage, dif_seed],
            outputs=[dif_out],
        )

    gr.Markdown(
        "---\nBuilt with [`llm-caliper`](https://github.com/aabhimittal/LLM-evaluation) · "
        "HF tasks used: `chat-completion` (subject & judge), `feature-extraction` "
        "(semantic consistency) · item bank: ARC-Challenge (CC BY-SA 4.0). "
        "Contamination probes are heuristics — elevated risk means *investigate*, "
        "not *guilty*.",
        elem_classes=["small-note"],
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
