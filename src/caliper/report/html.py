"""Self-contained HTML report with inline-SVG charts.

No external assets; colors are defined once as CSS custom properties (light
and dark) and every SVG mark references them by role.
"""

from __future__ import annotations

import html
import math

from caliper.report.fingerprint import Fingerprint

_CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --series: #2a78d6; --series-soft: rgba(42,120,214,0.16);
  --band: rgba(42,120,214,0.14); --ref: #898781;
  --good: #0ca30c; --critical: #d03b3b;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --series: #3987e5; --series-soft: rgba(57,135,229,0.22);
    --band: rgba(57,135,229,0.20); --ref: #898781;
    --good: #0ca30c; --critical: #d03b3b;
    --border: rgba(255,255,255,0.10);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 10px; color: var(--ink); }
.sub { color: var(--ink-2); font-size: 13px; margin-bottom: 24px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 18px 0; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; }
.tile .k { font-size: 12px; color: var(--ink-2); }
.tile .v { font-size: 22px; font-weight: 650; margin-top: 2px; }
.tile .d { font-size: 12px; color: var(--muted); margin-top: 2px; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px; overflow-x: auto; }
.card h3 { margin: 0 0 2px; font-size: 14px; }
.card .note { color: var(--ink-2); font-size: 12px; margin-bottom: 8px; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif;
  fill: var(--ink-2); }
svg .val { fill: var(--ink); font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.foot { color: var(--muted); font-size: 12px; margin-top: 28px; }
@media (max-width: 760px) { .cards { grid-template-columns: 1fr; } }
"""


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _radar_svg(dims: dict[str, float], size: int = 300) -> str:
    width = size + 120
    cx, cy = width / 2, size / 2 + 10
    radius = size / 2 - 46
    n = len(dims)
    labels = list(dims.keys())
    values = list(dims.values())

    def point(i: int, r: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    rings = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, radius * frac) for i in range(n)))
        rings.append(
            f'<polygon points="{pts}" fill="none" stroke="var(--grid)" stroke-width="1"/>'
        )
    spokes = [
        f'<line x1="{cx}" y1="{cy}" x2="{point(i, radius)[0]:.1f}" '
        f'y2="{point(i, radius)[1]:.1f}" stroke="var(--grid)" stroke-width="1"/>'
        for i in range(n)
    ]
    poly = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (point(i, radius * max(v, 0.02)) for i, v in enumerate(values))
    )
    dots = "".join(
        f'<circle cx="{point(i, radius * max(v, 0.02))[0]:.1f}" '
        f'cy="{point(i, radius * max(v, 0.02))[1]:.1f}" r="3.5" fill="var(--series)"/>'
        for i, v in enumerate(values)
    )
    texts = []
    for i, (label, v) in enumerate(zip(labels, values)):
        x, y = point(i, radius + 26)
        anchor = "middle"
        if x < cx - 12:
            anchor = "end"
        elif x > cx + 12:
            anchor = "start"
        texts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{html.escape(label)}</text>'
            f'<text class="val" x="{x:.1f}" y="{y + 13:.1f}" text-anchor="{anchor}">'
            f"{_fmt(v)}</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {size + 30}" width="100%" role="img" '
        f'aria-label="Fingerprint radar">'
        + "".join(rings) + "".join(spokes)
        + f'<polygon points="{poly}" fill="var(--series-soft)" '
          f'stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>'
        + dots + "".join(texts) + "</svg>"
    )


def _line_chart(
    points: list[tuple[float, float]],
    band: list[tuple[float, float, float]] | None = None,
    *,
    width: int = 420,
    height: int = 220,
    x_label: str = "",
    y_label: str = "",
    y_range: tuple[float, float] | None = None,
    reference_diagonal: bool = False,
    aria: str = "chart",
) -> str:
    if not points:
        return ""
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 34
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    if y_range:
        y0, y1 = y_range
    else:
        lo = min(ys + ([b[1] for b in band] if band else []))
        hi = max(ys + ([b[2] for b in band] if band else []))
        margin = (hi - lo) * 0.15 or 0.5
        y0, y1 = lo - margin, hi + margin
    if x1 == x0:
        x1 = x0 + 1

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - y0) / (y1 - y0) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="{html.escape(aria)}">']
    for frac in (0.0, 0.5, 1.0):
        gy = y0 + frac * (y1 - y0)
        parts.append(
            f'<line x1="{pad_l}" y1="{sy(gy):.1f}" x2="{width - pad_r}" y2="{sy(gy):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{sy(gy) + 4:.1f}" text-anchor="end">{_fmt(gy)}</text>'
        )
    if reference_diagonal:
        parts.append(
            f'<line x1="{sx(max(x0, y0)):.1f}" y1="{sy(max(x0, y0)):.1f}" '
            f'x2="{sx(min(x1, y1)):.1f}" y2="{sy(min(x1, y1)):.1f}" '
            f'stroke="var(--ref)" stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
    if band:
        top = " ".join(f"{sx(b[0]):.1f},{sy(b[2]):.1f}" for b in band)
        bot = " ".join(f"{sx(b[0]):.1f},{sy(b[1]):.1f}" for b in reversed(band))
        parts.append(f'<polygon points="{top} {bot}" fill="var(--band)"/>')
    path = " ".join(
        f"{'M' if i == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}" for i, (x, y) in enumerate(points)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="var(--series)" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
    lx, ly = points[-1]
    parts.append(f'<circle cx="{sx(lx):.1f}" cy="{sy(ly):.1f}" r="3.5" fill="var(--series)"/>')
    parts.append(
        f'<text class="val" x="{min(sx(lx) + 6, width - 40):.1f}" y="{sy(ly) - 8:.1f}">'
        f"{_fmt(ly)}</text>"
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
        f'y2="{height - pad_b}" stroke="var(--axis)" stroke-width="1"/>'
    )
    for frac in (0.0, 0.5, 1.0):
        gx = x0 + frac * (x1 - x0)
        parts.append(
            f'<text x="{sx(gx):.1f}" y="{height - pad_b + 16}" text-anchor="middle">'
            f"{_fmt(gx, 1 if (x1 - x0) < 10 else 0)}</text>"
        )
    if x_label:
        parts.append(f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 4}" '
                     f'text-anchor="middle">{html.escape(x_label)}</text>')
    if y_label:
        parts.append(f'<text x="{pad_l}" y="{pad_t - 1}">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _reliability_svg(bins: list[dict], width: int = 420, height: int = 220) -> str:
    if not bins:
        return '<p class="note">No parseable confidence data.</p>'
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 34

    def sx(x: float) -> float:
        return pad_l + x * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - y * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="Reliability diagram">']
    for frac in (0.0, 0.5, 1.0):
        parts.append(
            f'<line x1="{pad_l}" y1="{sy(frac):.1f}" x2="{width - pad_r}" y2="{sy(frac):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{sy(frac) + 4:.1f}" text-anchor="end">{_fmt(frac, 1)}</text>'
            f'<text x="{sx(frac):.1f}" y="{height - pad_b + 16}" text-anchor="middle">'
            f"{_fmt(frac, 1)}</text>"
        )
    parts.append(
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" '
        f'stroke="var(--ref)" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    for b in bins:
        x_left = sx(b["lo"]) + 1
        x_right = sx(b["hi"]) - 1
        bar_width = max(x_right - x_left, 2)
        y_top = sy(b["accuracy"])
        parts.append(
            f'<rect x="{x_left:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" '
            f'height="{max(sy(0) - y_top, 1):.1f}" rx="3" fill="var(--series-soft)" '
            f'stroke="var(--series)" stroke-width="1.5"/>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" '
        f'y2="{height - pad_b}" stroke="var(--axis)" stroke-width="1"/>'
        f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 4}" text-anchor="middle">'
        f"stated confidence</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def render_html(fp: Fingerprint) -> str:
    dims = fp.dimensions()
    est = fp.ability.estimate
    lo, hi = est.ci95
    trajectory = [(t["step"], t["theta"]) for t in fp.ability.trajectory]
    band = [
        (t["step"], t["theta"] - 1.96 * t["se"], t["theta"] + 1.96 * t["se"])
        for t in fp.ability.trajectory
    ]
    rc_points = [(p["coverage"], p["risk"]) for p in fp.calibration.risk_coverage]

    pert_rows = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        f'<td class="num">{_fmt(v)}</td></tr>'
        for k, v in sorted(fp.robustness.by_perturbation.items(), key=lambda kv: kv[1])
    )
    flip_rows = "".join(
        f'<tr><td>{html.escape(f["item_id"])}</td><td>{html.escape(f["perturbation"])}</td>'
        f'<td>{html.escape(str(f["baseline"]))[:40]}</td>'
        f'<td>{html.escape(str(f["perturbed"]))[:40]}</td></tr>'
        for f in fp.robustness.flips[:8]
    )

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="k">Ability θ</div>
        <div class="v">{est.theta:+.2f}</div>
        <div class="d">95% CI {lo:+.2f} … {hi:+.2f} · {est.n_items} items</div></div>
      <div class="tile"><div class="k">Accuracy on administered items</div>
        <div class="v">{_fmt(fp.ability.accuracy)}</div>
        <div class="d">adaptive selection targets ~50%</div></div>
      <div class="tile"><div class="k">Robustness</div>
        <div class="v">{_fmt(fp.robustness.overall_consistency)}</div>
        <div class="d">95% CI {_fmt(fp.robustness.ci95[0])} … {_fmt(fp.robustness.ci95[1])}</div></div>
      <div class="tile"><div class="k">Calibration error (ECE)</div>
        <div class="v">{_fmt(fp.calibration.ece)}</div>
        <div class="d">overconfidence {fp.calibration.overconfidence:+.2f}</div></div>
      <div class="tile"><div class="k">Contamination risk</div>
        <div class="v">{_fmt(fp.contamination.risk)}</div>
        <div class="d">continuation gap {fp.contamination.continuation_gap:+.2f}</div></div>
    </div>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caliper fingerprint · {html.escape(fp.model_name)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>Caliper fingerprint — {html.escape(fp.model_name)}</h1>
  <div class="sub">item bank <b>{html.escape(fp.bank_name)}</b>
    (calibration: {html.escape(fp.bank_calibration)}) · generated {html.escape(fp.created_at)}</div>
  {tiles}
  <div class="cards">
    <div class="card"><h3>Fingerprint</h3>
      <div class="note">all dimensions normalized to 0–1, higher is better</div>
      {_radar_svg(dims)}</div>
    <div class="card"><h3>Ability convergence</h3>
      <div class="note">θ estimate ±95% CI per adaptive step</div>
      {_line_chart(trajectory, band, x_label="items administered", y_label="θ",
                   aria="Ability convergence")}</div>
    <div class="card"><h3>Reliability diagram</h3>
      <div class="note">bar = observed accuracy per stated-confidence bin;
        dashed = perfect calibration</div>
      {_reliability_svg(fp.calibration.bins)}</div>
    <div class="card"><h3>Risk-coverage</h3>
      <div class="note">error rate if the model answers only above a confidence
        threshold · AURC {_fmt(fp.calibration.aurc)}</div>
      {_line_chart(rc_points, x_label="coverage", y_label="risk",
                   y_range=(0, max(0.5, max((p[1] for p in rc_points), default=0.5) * 1.2)),
                   aria="Risk-coverage curve")}</div>
  </div>
  <h2>Robustness by perturbation</h2>
  <div class="card"><table>
    <tr><th>perturbation</th><th class="num">consistency</th></tr>{pert_rows}
  </table></div>
  {"<h2>Example answer flips</h2><div class='card'><table><tr><th>item</th><th>perturbation</th><th>baseline</th><th>perturbed</th></tr>" + flip_rows + "</table></div>" if flip_rows else ""}
  <div class="foot">Generated by <b>llm-caliper</b> — measurement-science evaluation
    for LLMs. Ability is estimated with a 3PL IRT model and adaptive item selection;
    all intervals are 95%. Contamination probes are heuristics: treat elevated risk
    as cause for investigation, not proof.</div>
</div></body></html>"""
