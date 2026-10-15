"""Generate manuscript tables/figures from the recorded experiment, without rerunning it."""

import csv
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "Work" / "benchmark_results"


def main():
    with (RESULTS / "aggregate.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    names = {"H_memory": "Hyperbolic, memory", "quadratic_memory": "Quadratic, memory",
             "global_L": "Global $L$", "diminishing": "$2/(k+2)$", "Armijo": "Armijo",
             "adaptive_L": "Adaptive $L$", "adaptive_L0L1": "Adaptive $(L_0,L_1)$",
             "secant": "Safeguarded secant", "line_search": "Golden-section search"}
    methods = tuple(names)
    lines = [r"\begin{table}[p]", r"\centering\small",
             r"\caption{Recorded comparison. Each cell is the median over ten seeds; "
             r"successes count runs with $U\le10^{-5}$ within 1000 updates. "
             r"Capped runs remain in all medians. Error is a reference-based upper bound on suboptimality. "
             r"Function and gradient calls include line-search trials.}",
             r"\label{tab:results}",
             r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Method & Successes & Updates & $U$ & Error $\le$ & $f$ calls & $\nabla f$ calls \\",
             r"\hline"]
    for scenario in ("interior", "active", "active_scaled", "breast_cancer"):
        lines += [r"\multicolumn{7}{l}{\textbf{" + scenario.replace("_", r"\_") + r"}} \\"]
        for method in methods:
            row = next(r for r in rows if r["scenario"] == scenario and r["method"] == method)
            lines.append(f"{names[method]} & {row['certified_runs']}/10 & "
                         f"{float(row['median_iterations']):.1f} & "
                         f"{float(row['median_certificate']):.2e} & "
                         f"{float(row['median_error_upper']):.2e} & "
                         f"{float(row['median_function_evals']):.1f} & "
                         f"{float(row['median_gradient_evals']):.1f} " + r"\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (ROOT / "results_table.tex").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "figures").mkdir(exist_ok=True)
    shutil.copy2(RESULTS / "certificates.png", ROOT / "figures" / "certificates.png")
    adaptive = ROOT.parent / "Work" / "adaptive_results"
    if (adaptive / "aggregate.csv").exists():
        with (adaptive / "aggregate.csv").open(encoding="utf-8") as file:
            adaptive_rows = list(csv.DictReader(file))
        adaptive_names = {"H_memory": "Hyperbolic, known $H_0,H_1$",
                          "adaptive_L": "Adaptive $L$",
                          "adaptive_certificate": "Certificate-scaled backtracking",
                          "Armijo": "Armijo"}
        lines = [r"\begin{table}[htbp]", r"\centering\small",
                 r"\caption{Additional 840-run study: selected comparisons. Medians over ten seeds "
                 r"include capped runs; tolerance $10^{-5}$ and cap 1000 updates. On Bike Sharing, "
                 r"seeds vary the starting vertex of one fixed training objective.}",
                 r"\label{tab:adaptive}", r"\begin{tabular}{lrrrrr}", r"\hline",
                 r"Method & Successes & Updates & $U$ & $f$ calls & $\nabla f$ calls \\", r"\hline"]
        for scenario in ("interior", "active_scaled", "bike_poisson"):
            lines += [r"\multicolumn{6}{l}{\textbf{" + scenario.replace("_", r"\_") + r"}} \\"]
            for method, label in adaptive_names.items():
                row = next(r for r in adaptive_rows if r["scenario"] == scenario and r["method"] == method)
                lines.append(f"{label} & {row['certified_runs']}/10 & "
                             f"{float(row['median_iterations']):.1f} & "
                             f"{float(row['median_certificate']):.2e} & "
                             f"{float(row['median_function_evals']):.1f} & "
                             f"{float(row['median_gradient_evals']):.1f} " + r"\\")
            lines.append(r"\hline")
        lines += [r"\end{tabular}", r"\end{table}", ""]
        (ROOT / "adaptive_table.tex").write_text("\n".join(lines), encoding="utf-8")
    backtracking = ROOT.parent / "Work" / "backtracking_results"
    if (backtracking / "aggregate.csv").exists():
        with (backtracking / "aggregate.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        labels = {"adaptive_L_initial": "Ordinary, $M=U_0/D^2$",
                  "adaptive_L": "Ordinary, $M=1$",
                  "adaptive_certificate": "Certificate scaling",
                  "AC_FW": "AC-FW, $\\delta=1$", "Armijo": "Armijo"}
        lines = [r"\begin{table}[htbp]", r"\centering\small",
                 r"\caption{350-run extension, selected families. Medians include capped runs. "
                 r"The cap is 1000 outer iterations, including AC-FW rejections; "
                 r"the compulsory AC-FW initialization is included in all oracle costs. "
                 r"Success means $U\le10^{-5}$.}",
                 r"\label{tab:backtracking}", r"\begin{tabular}{lrrrrr}", r"\hline",
                 r"Method & Successes & Outer iterations & $U$ & $f$ calls & $\nabla f$ calls \\",
                 r"\hline"]
        for scenario in ("active", "active_scaled", "bike_poisson"):
            lines += [r"\multicolumn{6}{l}{\textbf{" + scenario.replace("_", r"\_") + r"}} \\"]
            for method, label in labels.items():
                row = next(r for r in rows if r["scenario"] == scenario and r["method"] == method)
                lines.append(f"{label} & {row['certified_runs']}/10 & "
                             f"{float(row['median_iterations']):.1f} & "
                             f"{float(row['median_certificate']):.2e} & "
                             f"{float(row['median_function_evals']):.1f} & "
                             f"{float(row['median_gradient_evals']):.1f} " + r"\\")
            lines.append(r"\hline")
        lines += [r"\end{tabular}", r"\end{table}", ""]
        (ROOT / "backtracking_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
