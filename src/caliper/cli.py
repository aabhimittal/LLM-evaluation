"""Caliper command line.

Examples
--------
Fingerprint a model on HF Inference Providers::

    caliper run --adapter hf --model Qwen/Qwen2.5-7B-Instruct --suite fingerprint

Offline demo with a simulated subject of known ability::

    caliper run --adapter simulated --theta 0.8 --suite fingerprint --out reports/

Judge two models pairwise and rank with bootstrap CIs::

    caliper compare --judge-model meta-llama/Llama-3.3-70B-Instruct \\
        --models Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct \\
        --prompts prompts.txt

Recalibrate the item bank from a real correctness matrix::

    caliper calibrate --matrix matrix.csv --bank src/caliper/data/item_bank.json

Run two models head-to-head and stop the moment the winner is decided
(anytime-valid, so peeking after every item is safe)::

    caliper duel --adapter hf --model-a Qwen/Qwen2.5-7B-Instruct \\
        --model-b microsoft/Phi-3.5-mini-instruct

Grade a judge against human preferences before trusting it to rank anything::

    caliper judge-card --adapter hf --judge-model meta-llama/Llama-3.3-70B-Instruct \\
        --data examples/gold_preferences.jsonl

Audit the benchmark itself — which items are unfair between model families::

    caliper dif --matrix matrix.csv --groups groups.csv

Ask what the benchmark can and cannot measure::

    caliper diagnose --test-length 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from caliper.types import ItemBank


def _say(msg: str) -> None:
    print(f"[caliper] {msg}", file=sys.stderr)


def _build_adapter(args: argparse.Namespace, model: str | None = None):
    from caliper.adapters import make_adapter

    kind = args.adapter
    model = model or getattr(args, "model", "") or ""
    kwargs: dict = {}
    if kind == "hf":
        kwargs["token"] = args.token or os.environ.get("HF_TOKEN")
    elif kind == "openai":
        kwargs["api_key"] = args.token or os.environ.get("OPENAI_API_KEY")
        if args.base_url:
            kwargs["base_url"] = args.base_url
    elif kind == "simulated":
        kwargs.update(
            theta=args.theta,
            calibration_skew=args.calibration_skew,
            robustness=args.robustness,
            contaminated=args.contaminated,
            seed=args.seed,
        )
    return make_adapter(kind, model=model, **kwargs)


def cmd_run(args: argparse.Namespace) -> int:
    bank = ItemBank.load(args.bank) if args.bank else ItemBank.bundled()
    adapter = _build_adapter(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = adapter.name.replace("/", "_").replace(" ", "_")

    if args.suite == "fingerprint":
        from caliper.report import render_html, run_fingerprint

        fp = run_fingerprint(
            adapter, bank, adaptive_max_items=args.max_items, seed=args.seed, progress=_say
        )
        json_path = out_dir / f"{stem}.fingerprint.json"
        html_path = out_dir / f"{stem}.fingerprint.html"
        json_path.write_text(fp.to_json(), encoding="utf-8")
        html_path.write_text(render_html(fp), encoding="utf-8")
        _say(f"wrote {json_path} and {html_path}")
        print(json.dumps(fp.dimensions(), indent=2))
        return 0

    if args.suite == "adaptive":
        from caliper.irt import run_adaptive

        last = None
        for state in run_adaptive(adapter, bank, max_items=args.max_items, seed=args.seed):
            e = state.estimate
            mark = "✓" if state.result.correct else "✗"
            _say(f"step {state.step:>3} {mark} {state.item.id}  "
                 f"θ={e.theta:+.2f} ±{1.96 * e.se:.2f}")
            last = state
        if last:
            e = last.estimate
            print(json.dumps({"theta": e.theta, "se": e.se, "ci95": list(e.ci95),
                              "n_items": e.n_items}, indent=2))
        return 0

    if args.suite == "robustness":
        from caliper.robustness import evaluate_robustness

        report = evaluate_robustness(adapter, bank, n_items=args.max_items // 2 or 10,
                                     seed=args.seed)
        print(json.dumps({"overall_consistency": report.overall_consistency,
                          "ci95": list(report.ci95),
                          "by_perturbation": report.by_perturbation}, indent=2))
        return 0

    if args.suite == "calibration":
        from dataclasses import asdict

        from caliper.calibration import evaluate_calibration

        report = evaluate_calibration(adapter, bank, n_items=args.max_items, seed=args.seed)
        payload = asdict(report)
        payload.pop("risk_coverage")
        print(json.dumps(payload, indent=2))
        return 0

    if args.suite == "contamination":
        from caliper.contamination import evaluate_contamination

        report = evaluate_contamination(adapter, bank, n_items=args.max_items // 2 or 12,
                                        seed=args.seed)
        print(json.dumps({"risk": report.risk,
                          "continuation_gap": report.continuation_gap,
                          "exact_continuation_rate": report.exact_continuation_rate,
                          "option_recall_rate": report.option_recall_rate}, indent=2))
        return 0

    _say(f"unknown suite {args.suite!r}")
    return 2


def _build_rag_adapter(args: argparse.Namespace, bank=None):
    if args.adapter == "simulated":
        from caliper.adapters import SimulatedRAGSubject

        return SimulatedRAGSubject(
            hallucination_rate=args.hallucination_rate,
            answer_relevance=args.answer_relevance,
            context_precision=args.context_precision,
            context_reliance=args.context_reliance,
            verifier_sensitivity=args.verifier_sensitivity,
            verifier_specificity=args.verifier_specificity,
            bank=bank,
            seed=args.seed,
        )
    return _build_adapter(args)


def cmd_rag(args: argparse.Namespace) -> int:
    from caliper.rag import RagBank, evaluate_rag

    bank = RagBank.load(args.rag_bank) if args.rag_bank else RagBank.bundled()
    adapter = _build_rag_adapter(args, bank)
    report = evaluate_rag(
        adapter, bank, n_samples=args.n_samples, seed=args.seed,
        with_audit=args.audit_verifier, with_attribution=args.probe_attribution,
        progress=_say,
    )
    payload = report.to_dict()
    payload.pop("per_sample", None)
    payload["unsupported_examples"] = payload["unsupported_examples"][:8]
    if payload.get("attribution"):
        payload["attribution"].pop("per_sample", None)

    if args.out:
        from caliper.report import render_rag_html

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = adapter.name.replace("/", "_").replace(" ", "_")
        json_path = out_dir / f"{stem}.rag.json"
        html_path = out_dir / f"{stem}.rag.html"
        json_path.write_text(report.to_json(), encoding="utf-8")
        html_path.write_text(
            render_rag_html(report, model_name=adapter.name, bank_name=bank.name),
            encoding="utf-8",
        )
        _say(f"wrote {json_path} and {html_path}")

    print(json.dumps(payload, indent=2))
    if report.verifier is not None and not report.verifier.usable:
        _say("WARNING: the verifier carries no usable signal (sensitivity + "
             "specificity <= 1) — faithfulness cannot be bias-corrected.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from caliper.judge import Match, PairwiseJudge, bootstrap_ratings

    prompts = [
        line.strip()
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contestants = [(m, _build_adapter(args, model=m)) for m in args.models]
    judge_adapter = _build_adapter(args, model=args.judge_model)
    judge = PairwiseJudge(judge_adapter, n_samples=args.judge_samples)

    matches: list[Match] = []
    for prompt in prompts:
        responses = {name: ad.ask(prompt, temperature=0.0, max_tokens=400)
                     for name, ad in contestants}
        names = [name for name, _ in contestants]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                verdict = judge.compare(prompt, responses[names[i]], responses[names[j]])
                matches.append(Match(names[i], names[j], verdict.p_a_wins))
                _say(f"{names[i]} vs {names[j]}: p={verdict.p_a_wins:.2f} "
                     f"conf={verdict.confidence:.2f} flip={verdict.position_flip}")

    table = bootstrap_ratings(matches, n_boot=args.n_boot, seed=args.seed)
    audit = judge.audit()
    result = {
        "ratings": {
            m: {"rating": table.rating[m], "ci95": list(table.ci95[m])}
            for m in table.sorted_models()
        },
        "n_matches": table.n_matches,
        "judge_audit": {
            "position_flip_rate": audit.position_flip_rate,
            "verbosity_bias": audit.verbosity_bias,
            "mean_confidence": audit.mean_confidence,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


def cmd_judge_card(args: argparse.Namespace) -> int:
    """Grade a judge against human labels before trusting it to rank models."""
    from caliper.judge import PairwiseJudge, judge_report_card

    rows = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        _say(f"no comparisons found in {args.data}")
        return 2
    missing = [
        key for key in ("prompt", "response_a", "response_b", "gold")
        if key not in rows[0]
    ]
    if missing:
        _say(f"each JSONL line needs keys prompt/response_a/response_b/gold; "
             f"missing {missing}")
        return 2

    if args.adapter == "simulated":
        # Demo mode: a judge with injectable pathologies, so the grading itself
        # can be sanity-checked against known ground truth.
        from caliper.adapters import SimulatedJudge

        judge_adapter = SimulatedJudge(
            accuracy=args.judge_accuracy,
            position_bias=args.judge_position_bias,
            verbosity_bias=args.judge_verbosity_bias,
            seed=args.seed,
        )
    else:
        judge_adapter = _build_adapter(args, model=args.judge_model)
    judge = PairwiseJudge(judge_adapter, n_samples=args.judge_samples)
    verdicts = []
    for i, row in enumerate(rows, 1):
        verdicts.append(
            judge.compare(row["prompt"], row["response_a"], row["response_b"])
        )
        if i % 10 == 0 or i == len(rows):
            _say(f"judged {i}/{len(rows)} comparisons")

    card = judge_report_card(verdicts, [row["gold"] for row in rows],
                             tie_band=args.tie_band)
    payload = card.to_dict()
    payload["judge_model"] = args.judge_model or judge.adapter.name
    payload["disagreements"] = card.disagreements
    print(json.dumps(payload, indent=2))
    return 0


def cmd_duel(args: argparse.Namespace) -> int:
    from caliper.sequential import run_item_duel

    bank = ItemBank.load(args.bank) if args.bank else ItemBank.bundled()
    adapter_a = _build_adapter(args, model=args.model_a)
    adapter_b = _build_adapter(args, model=args.model_b)
    if args.adapter == "simulated":
        # Give the two simulated contestants distinct abilities to duel with.
        from caliper.adapters import SimulatedSubject

        adapter_a = SimulatedSubject(theta=args.theta, bank=bank, seed=args.seed,
                                     name=args.model_a or "A")
        adapter_b = SimulatedSubject(theta=args.theta_b, bank=bank, seed=args.seed + 1,
                                     name=args.model_b or "B")

    last = None
    for state in run_item_duel(
        adapter_a, adapter_b, bank,
        alpha=args.alpha, max_items=args.max_items, seed=args.seed,
    ):
        lo, hi = state.ci95
        _say(
            f"item {state.step:>3}  outcome={state.outcome:.1f}  "
            f"win_rate={state.win_rate:.3f} CS=[{lo:.2f},{hi:.2f}]  "
            f"E_A={state.e_value_a:8.2f} E_B={state.e_value_b:8.2f}"
        )
        last = state
    if last is None:
        _say("no items administered")
        return 1
    duel_summary = {
        "models": {"A": adapter_a.name, "B": adapter_b.name},
        "n_observations": last.step,
        "win_rate_a": last.win_rate,
        "confidence_sequence_95": list(last.ci95),
        "e_value_a": last.e_value_a,
        "e_value_b": last.e_value_b,
        "decided": last.decided,
        "winner": (
            adapter_a.name if last.winner == "A"
            else adapter_b.name if last.winner == "B" else None
        ),
        "alpha": args.alpha,
        "note": (
            "Stopped as soon as the evidence crossed the threshold. The type-I "
            "error guarantee holds despite peeking after every item."
            if last.decided else
            "Inconclusive within the item budget — the models are close enough "
            "that this bank cannot separate them."
        ),
    }
    print(json.dumps(duel_summary, indent=2))
    return 0


def cmd_dif(args: argparse.Namespace) -> int:
    from caliper.dif import detect_dif

    bank = ItemBank.load(args.bank) if args.bank else ItemBank.bundled()
    responses = np.genfromtxt(args.matrix, delimiter=",", skip_header=args.skip_header)
    if responses.ndim == 1:
        responses = responses[None, :]
    groups = np.genfromtxt(args.groups, delimiter=",").astype(int).ravel()
    report = detect_dif(
        responses, groups, bank,
        n_strata=args.n_strata, alpha=args.alpha,
        reference_name=args.reference_name, focal_name=args.focal_name,
    )
    _say(
        f"{len(report.flagged)}/{report.n_items} items flagged "
        f"({report.flag_rate:.1%}) across {report.n_strata} ability strata"
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from caliper.diagnostics import (
        adaptive_efficiency,
        bank_health,
        items_needed,
        minimum_detectable_difference,
    )

    bank = ItemBank.load(args.bank) if args.bank else ItemBank.bundled()
    health = bank_health(bank, se_target=args.se_target, test_length=args.test_length)
    power = minimum_detectable_difference(
        bank, n_items=args.test_length or 40, theta=args.theta
    )
    efficiency = adaptive_efficiency(
        bank, theta=args.theta, n_items=args.test_length or 30
    )
    payload = {
        "bank": bank.name,
        "calibration": bank.calibration,
        "health": health.to_dict(),
        "power": power.to_dict(),
        "adaptive_efficiency": efficiency,
        "items_needed": {
            str(gap): items_needed(gap, bank, theta=args.theta)
            for gap in (1.5, 1.0, 0.5)
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from caliper.irt import fit_items

    bank = ItemBank.load(args.bank)
    raw = np.genfromtxt(args.matrix, delimiter=",", skip_header=args.skip_header)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != len(bank.items):
        _say(f"matrix has {raw.shape[1]} columns but bank has {len(bank.items)} items")
        return 2
    result = fit_items(raw, n_choices=np.array([len(it.choices) for it in bank.items]))
    for item, a, b in zip(bank.items, result.a, result.b):
        item.a, item.b = round(float(a), 4), round(float(b), 4)
    bank.calibration = args.label
    out = args.out or args.bank
    bank.save(out)
    _say(f"calibrated {len(bank.items)} items from {raw.shape[0]} respondents -> {out} "
         f"(converged={result.converged})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="caliper", description="Measurement-science evaluation for LLMs"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_adapter_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--adapter", default="simulated",
                       choices=["hf", "openai", "replay", "simulated"])
        p.add_argument("--model", default="")
        p.add_argument("--token", default=None, help="API token (or HF_TOKEN env)")
        p.add_argument("--base-url", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--bank", default=None, help="path to an item bank JSON")
        # simulated-subject knobs (demo mode)
        p.add_argument("--theta", type=float, default=0.5)
        p.add_argument("--calibration-skew", type=float, default=1.0)
        p.add_argument("--robustness", type=float, default=0.92)
        p.add_argument("--contaminated", action="store_true")

    run_parser = sub.add_parser("run", help="run an evaluation suite against one model")
    add_adapter_args(run_parser)
    run_parser.add_argument("--suite", default="fingerprint",
                            choices=["fingerprint", "adaptive", "robustness",
                                     "calibration", "contamination"])
    run_parser.add_argument("--max-items", type=int, default=40)
    run_parser.add_argument("--out", default="reports")
    run_parser.set_defaults(func=cmd_run)

    compare_parser = sub.add_parser("compare", help="pairwise-judge several models")
    add_adapter_args(compare_parser)
    compare_parser.add_argument("--models", nargs="+", required=True)
    compare_parser.add_argument("--judge-model", required=True)
    compare_parser.add_argument("--prompts", required=True)
    compare_parser.add_argument("--judge-samples", type=int, default=3)
    compare_parser.add_argument("--n-boot", type=int, default=200)
    compare_parser.set_defaults(func=cmd_compare)

    rag_parser = sub.add_parser(
        "rag", help="RAG grounding: faithfulness & relevance, with confidence intervals"
    )
    add_adapter_args(rag_parser)
    rag_parser.add_argument("--rag-bank", default=None, help="path to a RAG bank JSON")
    rag_parser.add_argument("--n-samples", type=int, default=20)
    rag_parser.add_argument("--hallucination-rate", type=float, default=0.2,
                            help="simulated: fraction of fabricated claims")
    rag_parser.add_argument("--answer-relevance", type=float, default=0.85,
                            help="simulated: how on-topic generated questions are")
    rag_parser.add_argument("--context-precision", type=float, default=0.75,
                            help="simulated: fraction of passages judged relevant")
    rag_parser.add_argument("--context-reliance", type=float, default=1.0,
                            help="simulated: share of claims actually read from context")
    rag_parser.add_argument("--verifier-sensitivity", type=float, default=1.0,
                            help="simulated: P(verifier says SUPPORTED | truly supported)")
    rag_parser.add_argument("--verifier-specificity", type=float, default=1.0,
                            help="simulated: P(verifier says NOT_SUPPORTED | unsupported)")
    rag_parser.add_argument("--audit-verifier", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="audit the verifier and bias-correct faithfulness")
    rag_parser.add_argument("--probe-attribution", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="test whether answers are earned by retrieval")
    rag_parser.add_argument("--out", default=None,
                            help="directory for JSON + HTML report")
    rag_parser.set_defaults(func=cmd_rag)
    card_parser = sub.add_parser(
        "judge-card", help="grade a judge against human labels before trusting it"
    )
    add_adapter_args(card_parser)
    card_parser.add_argument(
        "--data", required=True,
        help="JSONL with keys prompt, response_a, response_b, gold (A|B|tie)",
    )
    card_parser.add_argument("--judge-model", default="")
    card_parser.add_argument("--judge-samples", type=int, default=3)
    card_parser.add_argument("--tie-band", type=float, default=0.1)
    # Demo-mode judge pathologies (--adapter simulated).
    card_parser.add_argument("--judge-accuracy", type=float, default=0.9)
    card_parser.add_argument("--judge-position-bias", type=float, default=0.0)
    card_parser.add_argument("--judge-verbosity-bias", type=float, default=0.0)
    card_parser.set_defaults(func=cmd_judge_card)

    duel_parser = sub.add_parser(
        "duel", help="anytime-valid head-to-head: stop as soon as a winner is clear"
    )
    add_adapter_args(duel_parser)
    duel_parser.add_argument("--model-a", default="")
    duel_parser.add_argument("--model-b", default="")
    duel_parser.add_argument("--theta-b", type=float, default=-0.3,
                             help="ability of simulated model B (demo mode)")
    duel_parser.add_argument("--alpha", type=float, default=0.05)
    duel_parser.add_argument("--max-items", type=int, default=120)
    duel_parser.set_defaults(func=cmd_duel)

    dif_parser = sub.add_parser(
        "dif", help="audit the benchmark for items biased between model families"
    )
    dif_parser.add_argument("--matrix", required=True,
                            help="CSV, one row per model, one 0/1 column per item")
    dif_parser.add_argument("--groups", required=True,
                            help="CSV of 0 (reference) / 1 (focal), one per model row")
    dif_parser.add_argument("--bank", default=None)
    dif_parser.add_argument("--n-strata", type=int, default=5)
    dif_parser.add_argument("--alpha", type=float, default=0.05)
    dif_parser.add_argument("--reference-name", default="reference")
    dif_parser.add_argument("--focal-name", default="focal")
    dif_parser.add_argument("--skip-header", type=int, default=0)
    dif_parser.set_defaults(func=cmd_dif)

    diagnose_parser = sub.add_parser(
        "diagnose", help="what can this benchmark measure? saturation and power"
    )
    diagnose_parser.add_argument("--bank", default=None)
    diagnose_parser.add_argument("--se-target", type=float, default=0.3)
    diagnose_parser.add_argument("--test-length", type=int, default=None)
    diagnose_parser.add_argument("--theta", type=float, default=0.0)
    diagnose_parser.set_defaults(func=cmd_diagnose)

    calibrate_parser = sub.add_parser(
        "calibrate", help="fit IRT item parameters from a correctness matrix CSV"
    )
    calibrate_parser.add_argument("--matrix", required=True,
                                  help="CSV, one row per model, one 0/1 column per item")
    calibrate_parser.add_argument("--bank", required=True)
    calibrate_parser.add_argument("--out", default=None)
    calibrate_parser.add_argument("--label", default="user-calibrated")
    calibrate_parser.add_argument("--skip-header", type=int, default=0)
    calibrate_parser.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
