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


def _build_rag_adapter(args: argparse.Namespace):
    if args.adapter == "simulated":
        from caliper.adapters import SimulatedRAGSubject

        return SimulatedRAGSubject(
            hallucination_rate=args.hallucination_rate,
            answer_relevance=args.answer_relevance,
            context_precision=args.context_precision,
            seed=args.seed,
        )
    return _build_adapter(args)


def cmd_rag(args: argparse.Namespace) -> int:
    from caliper.rag import RagBank, evaluate_rag

    bank = RagBank.load(args.rag_bank) if args.rag_bank else RagBank.bundled()
    adapter = _build_rag_adapter(args)
    report = evaluate_rag(
        adapter, bank, n_samples=args.n_samples, seed=args.seed, progress=_say
    )
    payload = report.to_dict()
    payload.pop("per_sample", None)
    payload["unsupported_examples"] = payload["unsupported_examples"][:8]
    print(json.dumps(payload, indent=2))
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
    rag_parser.set_defaults(func=cmd_rag)

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
