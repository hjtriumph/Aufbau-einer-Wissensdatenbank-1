"""Figures + single-file dashboard. Every chart reads the frozen summary schema."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from arms.schema import ARM_ORDER
from core.console import enable_utf8


def _setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


_setup_font()

CITE_ARMS = ("C", "C-P1", "C-P2", "C-4", "C-q", "C-c", "D")
TYPE_ORDER = ("numeric", "single_hop", "multi_hop", "method", "unanswerable")
COLORS = {
    "A": "#8c8c8c",
    "A'": "#b0b0b0",
    "B": "#4c78a8",
    "C": "#2e7d32",
    "B-P1": "#6baed6",
    "B-P2": "#9ecae1",
    "C-P1": "#74c476",
    "C-P2": "#a1d99b",
    "D": "#f58518",
    "D'": "#ffbf79",
    "C-1": "#9467bd",
    "C-2": "#c5b0d5",
    "C-3": "#f7b6d2",
    "C-4": "#d62728",
    "C-q": "#bcbd22",
    "C-c": "#17becf",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cell(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def ladder_chart(summary: dict[str, dict], path: Path) -> Path:
    ids = [a for a in ["A", "A'", "B", "C"] if a in summary]
    values = [_f(summary.get(i, {}).get("completeness")) for i in ids]
    lows = [values[n] - _f(summary.get(i, {}).get("completeness_ci_low")) for n, i in enumerate(ids)]
    highs = [_f(summary.get(i, {}).get("completeness_ci_high")) - values[n] for n, i in enumerate(ids)]

    fig, ax = plt.subplots(figsize=(8.4, 5))
    bars = ax.bar(ids, values, color=[COLORS[i] for i in ids], width=0.62)
    ax.errorbar(ids, values, yerr=[lows, highs], fmt="none", ecolor="#333", capsize=5, lw=1.2)
    for bar, arm_id in zip(bars, ids):
        row = summary.get(arm_id, {})
        ax.annotate(
            f"{values[ids.index(arm_id)]:.3f}\n{_f(row.get('latency_p50')):.1f}s  ¥{_f(row.get('cost_cny_per_q')):.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("完整性（22 道可答题，3 次重复均值）")
    ax.set_title("图 1　检索优化阶梯：A → A' → B → C")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def prompt_chart(summary: dict[str, dict], path: Path) -> Path:
    groups = [
        ("B", [a for a in ("B", "B-P1", "B-P2") if a in summary]),
        ("C", [a for a in ("C", "C-P1", "C-P2") if a in summary]),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.25
    for offset, (label, ids) in enumerate(groups):
        for n, arm_id in enumerate(ids):
            x = offset * 2.2 + n * width
            row = summary.get(arm_id, {})
            value = _f(row.get("completeness"))
            ax.bar(x, value, width=width * 0.92, color=COLORS[arm_id], label=arm_id)
            ax.text(x, value + 0.01, f"{value:.3f}", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks([0.25, 2.45])
    ax.set_xticklabels(["B 系（基础检索）", "C 系（优化检索）"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("完整性")
    ax.set_title("图 2　提示词策略：P0 / P1(zero-shot CoT) / P2(3-shot)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def paradigm_chart(by_type: list[dict], path: Path, summary_ids: set[str]) -> Path:
    ids = [a for a in ("C", "D", "D'") if a in summary_ids]
    types = ["numeric", "single_hop", "multi_hop", "method"]
    lookup = {(r["arm_id"], r["type"]): _f(r.get("completeness")) for r in by_type}
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.24
    for n, arm_id in enumerate(ids):
        values = [lookup.get((arm_id, t), 0.0) for t in types]
        positions = [i + (n - 1) * width for i in range(len(types))]
        ax.bar(positions, values, width=width * 0.92, color=COLORS[arm_id], label=arm_id)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("完整性")
    ax.set_title("图 3　架构范式：C vs D(ReAct) vs D'(等上下文)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def ablation_chart(summary: dict[str, dict], paired: list[dict], path: Path) -> Path:
    items = [
        (arm_id, baseline, label)
        for arm_id, baseline, label in [
            ("C-1", "B", "分块"), ("C-2", "B", "混合检索"), ("C-3", "B", "重排序"),
            ("C-4", "B", "提示词纪律"), ("C-q", "C", "Step-back"), ("C-c", "C", "父子分块"),
        ]
        if arm_id in summary and baseline in summary
    ]
    diffs, lows, highs, labels = [], [], [], []
    paired_lookup = {
        (r["baseline"], r["compare"], r["metric"]): r
        for r in paired
    }
    for arm_id, baseline, label in items:
        row = paired_lookup.get((baseline, arm_id, "completeness"))
        if row:
            diffs.append(_f(row.get("mean_diff")))
            lows.append(_f(row.get("mean_diff")) - _f(row.get("ci_low")))
            highs.append(_f(row.get("ci_high")) - _f(row.get("mean_diff")))
        else:
            diffs.append(
                _f(summary.get(arm_id, {}).get("completeness"))
                - _f(summary.get(baseline, {}).get("completeness"))
            )
            lows.append(0.0)
            highs.append(0.0)
        labels.append(f"{arm_id}\n{label}")

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = ["#2e7d32" if d >= 0 else "#c62828" for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors, width=0.6)
    ax.errorbar(range(len(diffs)), diffs, yerr=[lows, highs], fmt="none", ecolor="#333", capsize=4)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("完整性增量（配对 bootstrap 均值差）")
    ax.set_title("图 4　消融归因：相对基线的净贡献（95% CI）")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def heatmap_chart(summary: dict[str, dict], by_type: list[dict], path: Path) -> Path:
    arms = [a for a in ARM_ORDER if a in summary]
    # Completeness is not defined for unanswerable cases; that column would
    # read as a false zero, so the heatmap covers the four answerable types.
    types = [t for t in TYPE_ORDER if t != "unanswerable"]
    lookup = {(r["arm_id"], r["type"]): _f(r.get("completeness")) for r in by_type}
    matrix = [[lookup.get((a, t), 0.0) for t in types] for a in arms]

    fig, ax = plt.subplots(figsize=(7.8, 8.2))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=25, ha="right")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    for i in range(len(arms)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("图 5　16 臂 × 题型完整性热力图")
    fig.colorbar(im, ax=ax, shrink=0.7)
    return _save(fig, path)


def scatter_chart(summary: dict[str, dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for arm_id, row in summary.items():
        x = _f(row.get("latency_p50"))
        y = _f(row.get("completeness"))
        cost = _f(row.get("cost_cny_per_q"))
        ax.scatter(x, y, s=40 + cost * 900, color=COLORS.get(arm_id, "#555"))
        ax.annotate(arm_id, (x, y), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("延迟中位数（秒）")
    ax.set_ylabel("完整性")
    ax.set_title("图 6　质量—延迟—成本（点面积 ∝ 单题成本）")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def refusal_chart(summary: dict[str, dict], path: Path) -> Path:
    arms = [a for a in ARM_ORDER if a in summary]
    correct = [_f(summary[a].get("refusal_correct")) for a in arms]
    false = [_f(summary[a].get("refusal_false")) for a in arms]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = range(len(arms))
    ax.bar([i - 0.2 for i in x], correct, width=0.4, label="正确拒答（8 道不可答）", color="#2e7d32")
    ax.bar([i + 0.2 for i in x], false, width=0.4, label="错误拒答（22 道可答）", color="#c62828")
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("图 7　拒答双指标")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def citation_chart(summary: dict[str, dict], path: Path) -> Path:
    arms = [a for a in CITE_ARMS if a in summary]
    values = [_f(summary[a].get("citation_precision")) for a in arms]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(arms, values, color=[COLORS.get(a, "#555") for a in arms])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("引用精度")
    ax.set_title("图 8　有引用义务的 7 个臂")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def forest_chart(paired: list[dict], path: Path) -> Path:
    rows = [r for r in paired if r.get("metric") == "completeness"]
    if not rows:
        rows = [r for r in paired if r.get("metric") == "faithfulness"][:12]
    rows = rows[:24]
    labels = [f"{r['compare']} vs {r['baseline']}" for r in rows]
    diffs = [_f(r.get("mean_diff")) for r in rows]
    errors = [
        [_f(r.get("mean_diff")) - _f(r.get("ci_low")) for r in rows],
        [_f(r.get("ci_high")) - _f(r.get("mean_diff")) for r in rows],
    ]

    fig, ax = plt.subplots(figsize=(7.8, max(3.6, 0.32 * len(rows) + 1)))
    y = list(range(len(rows)))
    ax.errorbar(diffs, y, xerr=errors, fmt="o", color="#333", ecolor="#555", capsize=3)
    ax.axvline(0, color="#c62828", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("完整性均值差（95% bootstrap CI）")
    ax.set_title("图 9　各臂相对基线的配对置信区间")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(path.stem)}">'


def _examples(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    wanted_types = ("numeric", "multi_hop", "unanswerable")
    best: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("repeat") != 0 or row.get("language") != "zh":
                continue
            key = f"{row['case_id']}|{row['arm']}"
            if row["arm"] not in ("A", "B", "C"):
                continue
            best[key] = row
    picks = []
    for case_type in wanted_types:
        cases = sorted(
            {
                row["case_id"]: row
                for row in best.values()
                if row["type"] == case_type
            }.items()
        )
        if not cases:
            continue
        case_id = cases[0][0]
        for arm in ("A", "B", "C"):
            row = best.get(f"{case_id}|{arm}")
            if row:
                picks.append(row)
    return picks


def build_dashboard(run_dir: Path) -> Path:
    summary_rows = _read_csv(run_dir / "summary.csv")
    by_type = _read_csv(run_dir / "by_type.csv")
    paired = _read_csv(run_dir / "paired.csv")
    summary = {r["arm_id"]: r for r in summary_rows}

    headers = ["Arm", "名称", "完整性", "相关性", "忠实度", "引用精度", "正确拒答", "错误拒答", "延迟p50", "延迟p95", "成本/题", "错误E1-E6"]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                row["arm_id"],
                row.get("arm_name", ""),
                _cell(row.get("completeness")),
                _cell(row.get("relevance")),
                _cell(row.get("faithfulness")),
                _cell(row.get("citation_precision")),
                f"{_f(row.get('refusal_correct')):.2f}",
                f"{_f(row.get('refusal_false')):.2f}",
                f"{_f(row.get('latency_p50')):.1f}s",
                f"{_f(row.get('latency_p95')):.1f}s",
                f"¥{_f(row.get('cost_cny_per_q')):.4f}",
                f"{row.get('error_E1','0')}/{row.get('error_E2','0')}/{row.get('error_E3','0')}/{row.get('error_E4','0')}/{row.get('error_E5','0')}/{row.get('error_E6','0')}",
            ]
        )

    examples_html = []
    for row in _examples(run_dir / "raw.jsonl"):
        examples_html.append(
            f"""
            <div class="answer-card">
              <div class="meta"><span class="badge">{html.escape(row['arm'])}</span>
                完整性 {html.escape(str(row.get('completeness')))} ·
                引用精度 {html.escape(str(row.get('citation_precision')))} ·
                {'拒答' if row.get('refused') else '作答'}</div>
              <pre>{html.escape(str(row.get('answer', ''))[:1400])}</pre>
            </div>"""
        )
    grouped_examples = ""
    if examples_html:
        grouped_examples = (
            '<div class="columns">'
            + "".join(examples_html)
            + "</div>"
        )

    figures = {
        "图 1 阶梯主图": "fig1_ladder.png",
        "图 2 提示词策略": "fig2_prompt.png",
        "图 3 架构范式": "fig3_paradigm.png",
        "图 4 消融归因": "fig4_ablation.png",
        "图 5 题型热力图": "fig5_heatmap.png",
        "图 6 质量-延迟-成本": "fig6_scatter.png",
        "图 7 拒答双指标": "fig7_refusal.png",
        "图 8 引用精度": "fig8_citation.png",
        "图 9 配对置信区间": "fig9_forest.png",
    }
    figure_html = "".join(
        f'<section class="figure"><h3>{html.escape(title)}</h3>{_img(run_dir / name)}</section>'
        for title, name in figures.items()
        if (run_dir / name).exists()
    )

    meta = {}
    if (run_dir / "run_meta.json").exists():
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 对比实验 Dashboard · {html.escape(run_dir.name)}</title>
<style>
:root{{--ink:#1d232b;--muted:#6b7684;--line:#e3e7ed;--bg:#f6f8fb;--accent:#2e7d32}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--bg)}}
header{{padding:28px 36px;background:#12202f;color:#fff}}
header h1{{margin:0 0 6px;font-size:22px;font-weight:600}}
header p{{margin:0;color:#a9b6c5;font-size:13px}}
main{{max-width:1280px;margin:0 auto;padding:24px 28px 60px}}
section.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:18px 0}}
h2{{font-size:16px;margin:0 0 14px;border-left:4px solid var(--accent);padding-left:10px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:#eef2f6;font-weight:600;position:sticky;top:0}}
tr:hover td{{background:#f9fbfd}}
.figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}
.figure h3{{font-size:14px;margin:8px 0}}
.columns{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.answer-card{{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fcfdff}}
.answer-card pre{{white-space:pre-wrap;font-family:inherit;font-size:12px;line-height:1.6;margin:8px 0 0;max-height:340px;overflow:auto}}
.badge{{display:inline-block;background:#12202f;color:#fff;border-radius:4px;padding:1px 6px;margin-right:6px;font-size:11px}}
.meta{{color:var(--muted);font-size:11.5px}}
.note{{color:var(--muted);font-size:12px}}
</style></head><body>
<header><h1>RAG 对比实验 · 16 臂 Dashboard</h1>
<p>Run {html.escape(run_dir.name)} · 模型 {html.escape(str(meta.get('model','')))} · 温度 {html.escape(str(meta.get('temperature','')))} · 题库 {html.escape(str(meta.get('questions_sha256',''))[:12])}</p></header>
<main>
<section class="card"><h2>按臂汇总表（22 道可答题）</h2>
{_table(headers, table_rows)}
<p class="note">完整性 = 每题 3 次重复先取均值，再对 22 题求均值；延迟为中位数与 p95。错误列依次为 E1 检索未召回 / E2 排序截断 / E3 生成失误 / E4 引用失误 / E5 拒答失误 / E6 工程错误。</p>
</section>
<section class="card"><h2>主图</h2>{figure_html}</section>
<section class="card"><h2>三臂同题对比（真实结果，非实时调用）</h2>{grouped_examples or '<p class="note">暂无 raw.jsonl</p>'}</section>
<section class="card"><h2>运行元数据</h2><pre class="note">{html.escape(json.dumps(meta, ensure_ascii=False, indent=2)[:4000])}</pre></section>
</main></body></html>"""

    target = run_dir / "dashboard.html"
    target.write_text(document, encoding="utf-8")
    return target


def generate(run_dir: Path) -> list[Path]:
    # Remove figures from a previous partial run: a stale stage chart would
    # otherwise reappear in the dashboard after the arms are gone.
    for stale in run_dir.glob("fig*.png"):
        stale.unlink()
    summary_rows = _read_csv(run_dir / "summary.csv")
    if not summary_rows:
        raise SystemExit(f"no summary.csv in {run_dir}")
    summary = {r["arm_id"]: r for r in summary_rows}
    by_type = _read_csv(run_dir / "by_type.csv")
    paired = _read_csv(run_dir / "paired.csv")

    figs = [
        ladder_chart(summary, run_dir / "fig1_ladder.png"),
        heatmap_chart(summary, by_type, run_dir / "fig5_heatmap.png"),
        scatter_chart(summary, run_dir / "fig6_scatter.png"),
        refusal_chart(summary, run_dir / "fig7_refusal.png"),
        citation_chart(summary, run_dir / "fig8_citation.png"),
        forest_chart(paired, run_dir / "fig9_forest.png"),
    ]
    # Stage-dependent charts are only drawn once their arms exist; an empty
    # panel would read as "zero" rather than "not run yet".
    if any(a in summary for a in ("B-P1", "B-P2", "C-P1", "C-P2")):
        figs.append(prompt_chart(summary, run_dir / "fig2_prompt.png"))
    if any(a in summary for a in ("D",)):
        figs.append(paradigm_chart(by_type, run_dir / "fig3_paradigm.png", set(summary)))
    if any(a in summary for a in ("C-1", "C-2", "C-3", "C-4", "C-q", "C-c")):
        figs.append(ablation_chart(summary, paired, run_dir / "fig4_ablation.png"))
    dashboard = build_dashboard(run_dir)
    print(f"[visualize] {len(figs)} figures + {dashboard}")
    return figs + [dashboard]


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="results/YYYYMMDD-HHMMSS; defaults to newest")
    args = ap.parse_args()
    if args.run:
        run_dir = Path(args.run)
    else:
        runs = [p for p in Path("results").iterdir() if (p / "summary.csv").exists()]
        if not runs:
            raise SystemExit("no runs with summary.csv")
        run_dir = max(runs, key=lambda p: (p / "summary.csv").stat().st_mtime)
    generate(run_dir)


if __name__ == "__main__":
    main()
