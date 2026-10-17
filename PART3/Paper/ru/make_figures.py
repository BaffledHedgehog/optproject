"""Russian labels for the article's two figures, using the original saved data."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
WORK = PAPER.parent / "Work"
OUTPUT = PAPER / "ru" / "figures"


def read(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    geometry_path = PAPER / "evidence" / "geometry_grid.csv"
    geometry = read(geometry_path)
    assert len(geometry) == 322
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for ax, panel in zip(axes, ("positive_base", "zero_base")):
        rows = [r for r in geometry if r["panel"] == panel]
        ax.loglog([float(r["x"]) for r in rows], [float(r["quadratic"]) for r in rows],
                  "--", color="#C67400", label="Квадратичная граница")
        ax.loglog([float(r["x"]) for r in rows], [float(r["exact"]) for r in rows],
                  "-", color="#15658C", label="Дифференциальная граница")
        ax.grid(True, which="major", alpha=.2)
    axes[0].set(xlabel=r"Истинный зазор $\delta$", ylabel=r"Верхняя граница $G$",
                title=r"$H_0=H_1=1$, $D=2$")
    axes[1].set(xlabel=r"$b=H_1D^2$", ylabel=r"Коэффициент в $G\leq c\delta$",
                title=r"Нулевая базовая кривизна, $H_0=0$")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.savefig(OUTPUT / "gap_geometry.pdf")
    plt.close(fig)

    input_paths = [WORK / study / "budget.csv"
                   for study in ("multidata_results", "count_comparator_results")]
    budgets = [r for path in input_paths for r in read(path)]
    manifest_path = WORK / "multidata_results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = manifest["methods"] + ["AC_FW"]
    assert methods == ["FW_BT_fixed", "FW_BT_certificate", "AFW_BT_fixed",
                       "AFW_BT_certificate", "FW_Q", "AFW_Q", "AC_FW"]
    labels = (r"FW $M=1$", r"FW $M=U_0/D^2$", r"AFW $M=1$", r"AFW $M=U_0/D^2$",
              "FW Q", "AFW Q", "AC-FW")
    styles = (("#d55e00", "-"), ("#e69f00", "--"), ("#0072b2", "-"), ("#009e73", "--"),
              ("#cc79a7", ":"), ("#666666", "-."), ("#111111", (0, (5, 1, 1, 1))))
    shown = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    series = []
    for i, dataset in enumerate(manifest["datasets"]):
        for j, ratio in enumerate(manifest["radius_ratios"]):
            ax = axes[i, j]
            for method, label, (color, style) in zip(methods, labels, styles):
                medians = []
                for budget in shown:
                    values = [float(r["certificate"]) for r in budgets if r["dataset"] == dataset
                              and float(r["radius_ratio"]) == ratio and r["method"] == method
                              and r["oracle"] == "function_evals" and int(r["budget"]) == budget]
                    assert len(values) == {"seoul": 30, "metro": 24}[dataset]
                    medians.append(median(values))
                ax.plot(shown, np.maximum(medians, 1e-16), label=label, color=color,
                        linestyle=style, linewidth=1.1)
                series.append({"dataset": dataset, "ratio": ratio, "method": method,
                               "budgets": shown, "medians": medians})
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(manifest["tolerance"], color=".5", linestyle=":", linewidth=.7)
            ax.set_title(f"{dataset.capitalize()}, R/R*={ratio:g}", fontsize=8)
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel("Медиана сертификата")
            if i == 1:
                ax.set_xlabel("Бюджет вызовов функции")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", frameon=False, ncol=4)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(OUTPUT / "count_comparators.pdf")
    plt.close(fig)
    inputs = [geometry_path, manifest_path, *input_paths]
    report = {"scope": "Russian labels; original saved grid and median calculation; no new optimization runs",
              "inputs_sha256": {str(p.relative_to(PAPER.parent)).replace("\\", "/"):
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "series": series}
    (OUTPUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8", newline="\n")
    print("Russian figures generated from the original 322 grid rows and 28 budget curves")


if __name__ == "__main__":
    main()
