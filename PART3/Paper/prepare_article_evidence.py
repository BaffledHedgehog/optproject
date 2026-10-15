"""Generate focused-article tables and a claim ledger from immutable archives."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent/"Work"
INPUTS = {}


def read(study, name):
    root = WORK/study
    INPUTS[f"{study}/{name}"] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if name in manifest.get("result_sha256", {}):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == manifest["result_sha256"][name]
    with (root/name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    evidence = ROOT/"evidence"
    evidence.mkdir(exist_ok=True)
    ledger = []
    def record(claim, source, selection, value):
        ledger.append({"claim": claim, "source": source, "selection": selection, "value": str(value)})
    def table(name, caption, label, columns, headings, rows):
        lines = [r"\begin{table}[htbp]", r"\centering\small", r"\caption{"+caption+"}",
                 r"\label{"+label+"}", r"\begin{tabular}{"+columns+"}", r"\toprule",
                 " & ".join(headings)+r" \\", r"\midrule"]
        lines.extend(" & ".join(row)+r" \\" for row in rows)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        (evidence/name).write_text("\n".join(lines), encoding="utf-8")
    manifest = json.loads((WORK/"multidata_results"/"manifest.json").read_text(encoding="utf-8"))
    metadata = []
    for dataset in manifest["datasets"]:
        ps = [p for p in manifest["problems"] if p["dataset"] == dataset]
        p = ps[0]
        width = max(q["reference_upper"]-q["reference_lower"] for q in ps)
        metadata.append([dataset.capitalize(), str(p["rows"]), str(p["dimension"]),
                         f"{p['critical_radius']:.6f}", f"{p['H1']:.3f}", f"{width:.2g}"])
        record("training_rows", "multidata_results/manifest.json", dataset, p["rows"])
        record("reference_width_max", "multidata_results/manifest.json", dataset, width)
    table("data_table.tex", r"Count-data protocols. $R_*$ is the offline unconstrained reference "
          r"$\ell_1$ norm; tested radii are $R_*/2$ and $2R_*$. Widths are numerical reference brackets, "
          r"without directed rounding. All $2d$ signed coordinate vertices are used at each radius.",
          "tab:data", "lrrrrr", ["Data", "Training rows", "$d$", "$R_*$", "$H_1$", "Max. width"], metadata)
    summary = read("multidata_results", "summary.csv")+read("count_comparator_results", "summary.csv")
    count_budgets = read("multidata_results", "budget.csv")+read("count_comparator_results", "budget.csv")
    rows = []
    methods = manifest["methods"]+["AC_FW"]
    for dataset in manifest["datasets"]:
        for ratio in manifest["radius_ratios"]:
            cells = []
            for method in methods:
                rs = [r for r in summary if r["dataset"] == dataset
                      and float(r["radius_ratio"]) == ratio and r["method"] == method]
                count, calls = sum(r["certified"] == "True" for r in rs), median(int(r["function_evals"]) for r in rs)
                cells.append(f"{count}; {calls:g}")
                record("certified_count;median_function_calls", "multidata_results/summary.csv",
                       f"{dataset};R/R*={ratio:g};{method}", f"{count}/{len(rs)};{calls:g}")
            rows.append([dataset.capitalize(), f"{ratio:g}", *cells])
    table("counts_table.tex", r"Certification at $10^{-5}$ within 1000 updates. Each cell gives the "
          r"certified count and median function calls to stop or cap, including failed runs. Seoul has "
          r"30 starts and Metro 24 per radius/method. $F/A$: FW/AFW; subscripts $1,U$: BT initialized "
          r"at $1,U_0/D^2$; $Q$: known-$H$ short step. AC counts 1000 outer trials, including "
          r"rejections, and pays two preprocessing source evaluations. These medians are not times to success.",
          "tab:counts", "llrrrrrrr", ["Data", "$R/R_*$", "$F_1$", "$F_U$", "$A_1$", "$A_U$", "$F_Q$", "$A_Q$", "AC"], rows)
    rows = []
    for dataset in manifest["datasets"]:
        for ratio in manifest["radius_ratios"]:
            cells = []
            for method in methods:
                rs = [r for r in count_budgets if r["dataset"] == dataset and float(r["radius_ratio"]) == ratio
                      and r["method"] == method and r["oracle"] == "function_evals" and r["budget"] == "1000"]
                assert rs and all(r["budget_reached_or_certified"] == "True" for r in rs)
                certified, mid = sum(r["certified"] == "True" for r in rs), median(float(r["certificate"]) for r in rs)
                cells.append(str(certified))
                record("certified_at_1000f;median_certificate", "multidata_results/budget.csv + count_comparator_results/budget.csv",
                       f"{dataset};R/R*={ratio:g};{method}", f"{certified}/{len(rs)};{mid:.12g}")
            rows.append([dataset.capitalize(), f"{ratio:g}", *cells])
    table("budget_table.tex", r"Certified starts at the common budget of 1000 function calls. "
          r"Every nonterminal run reaches this budget before its cap; completed runs are carried forward. "
          r"Seoul has 30 starts per cell, Metro 24. $F/A$: FW/AFW; subscripts $1,U$: initial "
          r"$M=1,U_0/D^2$; $Q$: known-curvature short step; AC: auto-conditioned FW with charged preprocessing.",
          "tab:budget", "llrrrrrrr", ["Data", "$R/R_*$", "$F_1$", "$F_U$", "$A_1$", "$A_U$", "$F_Q$", "$A_Q$", "AC"], rows)
    budgets = read("initialization_factorial", "budget.csv")
    comparisons = read("initialization_factorial", "paired_comparisons.csv")
    rows = []
    for scenario in ("active", "active_scaled", "active_highdim"):
        cells = []
        for cell in ("AC_original", "AC_vertex", "BT_original", "BT_vertex"):
            rs = [r for r in budgets if r["scenario"] == scenario and r["cell"] == cell
                  and r["oracle"] == "function_evals" and int(r["budget"]) == 1000]
            assert len(rs) == 10
            count = sum(r["certified"] == "True" for r in rs)
            cells.append(str(count))
            record("initialization_success_count", "initialization_factorial/budget.csv", f"{scenario};{cell};1000f", f"{count}/10")
        paired = next(r for r in comparisons if r["scenario"] == scenario and r["numerator"] == "BT_vertex"
                      and r["denominator"] == "AC_vertex" and r["oracle"] == "function_evals" and r["budget"] == "1000")
        interval = f"{float(paired['paired_median_certificate_ratio']):.3f} [{float(paired['bootstrap_025']):.3f}, {float(paired['bootstrap_975']):.3f}]"
        rows.append([scenario.replace("_", r"\_"), *cells, interval])
        record("paired_certificate_ratio_interval", "initialization_factorial/paired_comparisons.csv", scenario, interval)
    table("initialization_control.tex", r"Starting-point control on three planted active families, "
          r"ten matched instances each. Counts are certified starts at 1000 function calls. O/V: original "
          r"point/compulsory initial LMO vertex. The last column is the descriptive paired median "
          r"BT-V/AC-V certificate ratio and 95\% percentile-bootstrap interval, not a speedup estimate.",
          "tab:init", "lrrrrl", ["Family", "AC-O", "AC-V", "BT-O", "BT-V", "Ratio [interval]"], rows)
    summaries = read("certificate_results", "summary.csv")
    paths = read("certificate_results", "audit.csv")
    rows = []
    for profile in ("scalar", "block_8", "block_32", "block_128", "known_mu"):
        rs = [r for r in summaries if r["profile"] == profile]
        baseline = {(r["dataset"], r["radius_ratio"], r["start"], r["method"]): r
                    for r in summaries if r["profile"] == "scalar"}
        early = sum(int(r["iteration"]) < int(baseline[(r["dataset"], r["radius_ratio"], r["start"], r["method"])]["iteration"]) for r in rs)
        extra_cost = sum(int(r["lmo_evals"]) > int(baseline[(r["dataset"], r["radius_ratio"], r["start"], r["method"])]["lmo_evals"]) for r in rs)
        successes = sum(r["certified"] == "True" for r in rs)
        rows.append([profile.replace("_", r"\_"), str(successes), str(early), str(extra_cost)])
        record("profile_successes;early;more_total_LMOs", "certificate_results/summary.csv", profile, f"{successes};{early};{extra_cost}")
    assert len(paths) == 432 and len(summaries) == 2160
    table("monitor_table.tex", r"Isolated causal controls on 432 exactly repeated BT paths, "
          r"giving 2160 dependent profiles. Counts use each profile's first stop or the original cap. "
          r"More LMOs compares total direction plus certificate LMOs with scalar. "
          r"Known $\mu$ receives the analytical ridge coefficient; blocks do not.",
          "tab:monitors", "lrrr", ["Profile", "Certified", "Earlier prefix", "More total LMOs"], rows)
    with (evidence/"claim_ledger.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    artifacts = ("budget_table.tex", "claim_ledger.csv", "counts_table.tex",
                 "data_table.tex", "initialization_control.tex", "monitor_table.tex")
    studies = ("multidata_results", "count_comparator_results", "initialization_factorial", "certificate_results")
    report = {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_manifest_sha256": {s: hashlib.sha256((WORK/s/"manifest.json").read_bytes()).hexdigest() for s in studies},
              "source_csv_sha256": INPUTS,
              "artifact_sha256": {n: hashlib.sha256((evidence/n).read_bytes()).hexdigest() for n in artifacts}}
    # Vector figure with the same seven methods and the uncensored function budget.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
    shown = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    styles = (("#d55e00", "-"), ("#e69f00", "--"), ("#0072b2", "-"), ("#009e73", "--"),
              ("#cc79a7", ":"), ("#666666", "-."), ("#111111", (0, (5, 1, 1, 1))))
    labels = (r"FW $M=1$", r"FW $M=U_0/D^2$", r"AFW $M=1$", r"AFW $M=U_0/D^2$", "FW Q", "AFW Q", "AC-FW")
    for i, dataset in enumerate(manifest["datasets"]):
        for j, ratio in enumerate(manifest["radius_ratios"]):
            ax = axes[i, j]
            for method, label, (color, style) in zip(methods, labels, styles):
                values = [[float(r["certificate"]) for r in count_budgets if r["dataset"] == dataset
                           and float(r["radius_ratio"]) == ratio and r["method"] == method
                           and r["oracle"] == "function_evals" and int(r["budget"]) == b] for b in shown]
                mid = [median(v) for v in values]
                ax.plot(shown, np.maximum(mid, 1e-16), label=label, color=color, linestyle=style, linewidth=1.1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(manifest["tolerance"], color=".5", linestyle=":", linewidth=.7)
            ax.set_title(f"{dataset.capitalize()}, R/R*={ratio:g}", fontsize=8)
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel("Median certificate")
            if i == 1:
                ax.set_xlabel("Function-call budget")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=4)
    fig.tight_layout(rect=(0, .12, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(ROOT/"figures"/f"count_comparators.{extension}", dpi=220)
    plt.close(fig)
    report["figure_sha256"] = {f"figures/count_comparators.{ext}": hashlib.sha256(
        (ROOT/"figures"/f"count_comparators.{ext}").read_bytes()).hexdigest() for ext in ("pdf", "png")}
    (evidence/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Focused-article evidence: {len(ledger)} claim records, five generated tables and vector figure")


if __name__ == "__main__":
    main()
