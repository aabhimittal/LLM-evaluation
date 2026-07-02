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

from caliper.adapters import SimulatedJudge, SimulatedSubject
from caliper.irt import run_adaptive
from caliper.judge import PairwiseJudge
from caliper.report import render_html, run_fingerprint
from caliper.report.html import _line_chart, _radar_svg
from caliper.robustness import evaluate_robustness
from caliper.robustness.perturb import PERTURBATIONS
from caliper.types import ItemBank

BANK = ItemBank.bundled()

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
                max_items, progress=gr.Progress()):
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Robustness run failed: {e}") from e
    lines = [
        f"### Consistency: **{report.overall_consistency:.2f}** "
        f"(95% CI {report.ci95[0]:.2f} – {report.ci95[1]:.2f}, {report.n_items} items)",
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
                   progress=gr.Progress()):
    try:
        adapter = _subject(mode, model_id, token, theta, robustness, skew, contaminated)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(str(e)) from e
    live = not mode.startswith("Demo")
    budgets = dict(
        adaptive_max_items=25 if live else 40,
        robustness_items=5 if live else 10,
        calibration_items=15 if live else 30,
        contamination_items=6 if live else 12,
    )
    stages = ["adaptive ability", "robustness", "calibration", "contamination"]
    stage_iter = iter(np.linspace(0.1, 0.9, len(stages)))

    def report_progress(msg: str):
        try:
            progress(next(stage_iter), desc=msg)
        except StopIteration:
            pass

    try:
        fp = run_fingerprint(adapter, BANK, seed=0, progress=report_progress, **budgets)
    except Exception as e:  # noqa: BLE001
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


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Caliper — LLM measurement lab") as demo:
    gr.Markdown(
        "# 🔬 Caliper — measurement-science evaluation for LLMs\n"
        "Point estimates lie. Caliper measures models the way psychometrics measures "
        "people: **adaptive IRT ability estimation** with confidence intervals, "
        "**bias-audited LLM-as-judge**, **metamorphic robustness**, **confidence "
        "calibration** and **contamination probes** — every number with its "
        "uncertainty. [Source & methodology](https://github.com/aabhimittal/LLM-evaluation)"
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
