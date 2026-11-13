"""Generate bilingual paper tables and figures from saved evidence only."""
import csv
import argparse
import gzip
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_multidim import METHODS, LABELS, dump

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Evidence/multidim"


def number(x):
    return f"{x:g}"


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.data
    rows = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((DATA / "problems.json").read_text(encoding="utf-8"))
    summaries, radius_rows, comparisons = [], [], []
    baseline = {(r["problem"], r["start"]): r for r in rows if r["method"] == "gap_log"}
    for method, label in zip(METHODS, LABELS):
        subset = [r for r in rows if r["method"] == method]
        summaries.append(label + " & " + str(sum(r["status"] == "converged" for r in subset)) + "/54 & "
                         + " & ".join(number(median(r[k] for r in subset)) for k in
                                      ("updates", "f_calls", "g_calls", "lmo_calls"))
                         + f" & {1000 * median(r['seconds'] for r in subset):.2f}" + r" \\")
        radius_rows.append(label + " & " + " & ".join(number(median(r["updates"] for r in subset
                                                                                   if r["radius"] == R))
                                                          for R in (0.5, 2, 4)) + r" \\")
        if method != "gap_log":
            comparisons.append({"method": method, **{
                k + "_less_than_gap_log": sum(r[k] < baseline[(r["problem"], r["start"])][k] for r in subset)
                for k in ("updates", "f_calls", "g_calls", "lmo_calls", "seconds")}})
    for folder in (ROOT / "Paper", ROOT / "Paper/ru"):
        (folder / "multidim_table.tex").write_text("\n".join(summaries) + "\n", encoding="utf-8")
        (folder / "radius_table.tex").write_text("\n".join(radius_rows) + "\n", encoding="utf-8")
    dump(DATA / "paired_comparison.json", comparisons)

    budgets = np.unique(np.rint(np.geomspace(1, 20001, 100)).astype(int))
    costs = ("f_calls", "g_calls", "lmo_calls")
    curves = {}
    with gzip.open(DATA / "traces.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["problem"], row["start"], row["method"])
            if key not in curves:
                curves[key] = np.full((3, len(budgets)), np.inf)
            cert = max(float(row["certificate"]), 1e-16)
            for j, cost in enumerate(costs):
                first = np.searchsorted(budgets, int(row[cost]))
                curves[key][j, first:] = np.minimum(curves[key][j, first:], cert)
    aggregates = {m: np.median([v for k, v in curves.items() if k[2] == m], axis=0) for m in METHODS}
    dump(DATA / "budget_curves.json", {"budgets": budgets.tolist(),
         "note": "Median best recorded certificate over 54 starts; capped and completed trajectories are held at their final best value.",
         "costs": list(costs), "values": {m: a.tolist() for m, a in aggregates.items()}})
    figures = ROOT / "Paper/figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for lang in ("en", "ru"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
        for method, label in zip(METHODS, LABELS):
            for j, axis in enumerate(axes):
                axis.loglog(budgets, aggregates[method][j], label=label, linewidth=1.5)
        titles = (["Function calls", "Gradient calls", "LMO calls"] if lang == "en"
                  else ["Вычисления функции", "Вычисления градиента", "Линейные минимизации"])
        for axis, title in zip(axes, titles):
            axis.axhline(1e-4, color="black", linestyle=":", linewidth=1)
            axis.set_xlabel(title); axis.grid(alpha=0.2, which="both")
            axis.set_ylim(1e-7, 20)
        axes[0].set_ylabel("Median best certificate" if lang == "en" else "Медиана лучшего сертификата")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        fig.savefig(figures / f"multidim_{lang}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(figures / f"multidim_{lang}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 3))
        with (ROOT / 'Evidence/v19/gap_summary.csv').open(encoding='utf-8', newline='') as stream:
            controls = list(csv.DictReader(stream))
        controls = [r for r in controls if r['family'] == 'exponential' and float(r['A']) == 1]
        Rs = sorted({float(r['R']) for r in controls})
        values_by_rule = {rule: [next(int(r['accepted_updates']) for r in controls
                                     if r['rule'] == rule and float(r['R']) == R) for R in Rs]
                          for rule in ('gap_logarithmic', 'ordinary_best_global_L')}
        for values, label, marker in ((values_by_rule['gap_logarithmic'], "Gap-log", "o"),
                                      (values_by_rule['ordinary_best_global_L'], "Global-L", "s"),
                                      ([1] * 5, "Open-loop / exact line search", "^")):
            ax.semilogy(Rs, values, marker=marker, label=label)
        ax.set_xlabel("R")
        ax.set_ylabel("Updates" if lang == "en" else "Обновления")
        ax.legend(frameon=False); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(figures / f"gap_benchmark_{lang}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(figures / f"gap_benchmark_{lang}.png", dpi=180)
        plt.close(fig)
    print("Generated tables and bilingual figures from", len(rows), "runs;", len(metadata), "problems.")
    print("\n".join(radius_rows))


if __name__ == "__main__":
    main()
