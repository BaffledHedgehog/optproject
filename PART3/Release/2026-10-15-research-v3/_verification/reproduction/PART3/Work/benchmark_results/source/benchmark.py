"""Run the expanded PART3 experiment; the legacy experiment is preserved.

python PART3/Work/benchmark.py --seeds 10 --max-iter 1000 --tolerance 1e-5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from fw import METHODS, solve
from problems import SCENARIOS, make_problem


ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, scenarios, seeds):
    result = []
    for scenario in scenarios:
        for method in METHODS:
            selected = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            out = {"scenario": scenario, "method": method, "runs": len(selected),
                   "certified_runs": sum(r["certified"] for r in selected)}
            for key in ("iterations", "certificate", "error_upper", "error_lower",
                        "function_evals", "gradient_evals", "lmo_evals", "elapsed_seconds", "support",
                        "certificate_over_current_gap"):
                values = [r[key] for r in selected]
                out[f"median_{key}"] = float(np.median(values))
                if key == "elapsed_seconds":
                    out["q25_seconds"], out["q75_seconds"] = map(float, np.quantile(values, [0.25, 0.75]))
            # Paired bootstrap across matched seed instances, not across iterations.
            control = {r["seed"]: r for r in rows if r["scenario"] == scenario and r["method"] == "H_memory"}
            ratios = np.array([r["error_upper"] / max(control[r["seed"]]["error_upper"], 1e-15)
                               for r in selected])
            rng = np.random.default_rng(8301)
            bootstrap = np.median(rng.choice(ratios, size=(2000, seeds), replace=True), axis=1)
            out["median_error_ratio_to_H_memory"] = float(np.median(ratios))
            out["ratio_bootstrap_025"], out["ratio_bootstrap_975"] = map(float, np.quantile(bootstrap, [0.025, 0.975]))
            result.append(out)
    return result


def make_report(output, aggregated, args):
    lines = ["# Расширенный эксперимент PART3", "",
             f"Параметры: seed 0–{args.seeds-1}, максимум {args.max_iter} обновлений, "
             f"остановка по накопленному сертификату U ≤ {args.tolerance:g}.", "",
             "Все задачи совпадают для методов одного seed. H_nonnegative дополнительно знает "
             "нижнюю границу 0; остальные варианты H её не используют. H₀,H₁ известны аналитически. "
             "Эталонные решения не передаются оптимизаторам. Гиперпараметры baseline фиксированы, "
             "подбор на тестовых задачах не выполнялся.", "",
             "Медианы включают запуски, исчерпавшие бюджет. Столбец успехов обязателен: "
             "1000 итераций не означает достижение точности. Ошибка — верхняя граница f(x)−f*, "
             "вычисленная по независимой нижней оценке эталона. Для cosh эталон точный; "
             "для логистической регрессии ширина эталонного интервала записана в manifest.json.", "",
             "Время — один запуск каждого метода на задаче, медиана и IQR по seed, "
             "включая ведение журнала и сертификата; это предварительное измерение, "
             "не микробенчмарк. Вызовы f и ∇f считаются раздельно; принятые значения "
             "при поиске шага повторно используются. Производные внутри поиска секущими "
             "считаются градиентами. Построение задачи и эталона во время метода не входит.", ""]
    for scenario in args.scenarios:
        lines += [f"## {scenario}", "",
                  "| Метод | Успехи | Итерации | U | Ошибка ≤ | f | ∇f | LMO | Время, мс [IQR] |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in aggregated:
            if row["scenario"] != scenario:
                continue
            lines.append(f"| {row['method']} | {row['certified_runs']}/{row['runs']} | "
                         f"{row['median_iterations']:.0f} | {row['median_certificate']:.2e} | "
                         f"{row['median_error_upper']:.2e} | {row['median_function_evals']:.0f} | "
                         f"{row['median_gradient_evals']:.0f} | {row['median_lmo_evals']:.0f} | "
                         f"{1000*row['median_elapsed_seconds']:.1f} "
                         f"[{1000*row['q25_seconds']:.1f}, {1000*row['q75_seconds']:.1f}] |")
        lines.append("")
    lines += ["## Интерпретация", "",
              "Сравнивать качество следует одновременно с сертификатом и бюджетом оракулов. "
              "Успехов может быть мало даже при маленькой фактической ошибке: сертификат консервативен. "
              "breast_cancer — контроль H₁=0 на реальных данных; он не проверяет выгоду обобщённой "
              "гладкости. Активные cosh-задачи специально сконструированы с известным ККТ-решением "
              "и не заменяют внешнюю проверку на естественных задачах с H₁>0.", "",
              "В aggregate.csv также есть парный bootstrap-интервал для медианы отношения конечных "
              "ошибок к H_memory. Это описательная статистика при данном бюджете и правиле остановки; "
              "она не является поправленным тестом множественных гипотез. Нулевые ошибки в "
              "знаменателе ограничены снизу 1e-15. Десять seed дают лишь предварительную оценку "
              "устойчивости; для статьи нужны дополнительные семейства задач и повторные замеры времени.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(output, traces, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = ("H_memory", "quadratic_memory", "global_L", "diminishing", "Armijo",
               "adaptive_L", "adaptive_L0L1", "secant", "line_search")
    colors = dict(zip(plotted, plt.get_cmap("tab10").colors))
    ncols = 3
    nrows = (len(args.scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows), squeeze=False)
    grid = np.arange(args.max_iter+1)
    for ax, scenario in zip(axes.flat, args.scenarios):
        for method in plotted:
            curves = []
            for seed in range(args.seeds):
                rows = traces[(scenario, seed, method)]
                curves.append(np.interp(grid, [r["iteration"] for r in rows], [r["certificate"] for r in rows]))
            values = np.maximum(curves, 1e-16)
            med = np.median(values, axis=0)
            lo, hi = np.quantile(values, [0.25, 0.75], axis=0)
            ax.semilogy(grid, med, color=colors[method], label=method, lw=1.5)
            ax.fill_between(grid, lo, hi, color=colors[method], alpha=0.08)
        ax.axhline(args.tolerance, color="gray", linestyle=":", lw=1)
        ax.set(title=scenario, xlabel="FW updates", ylabel="Certificate U (median, IQR)")
        ax.set_ylim(bottom=args.tolerance * 0.2)
        ax.grid(alpha=0.25, which="both")
    for ax in list(axes.flat)[len(args.scenarios):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "certificates.png", dpi=160)
    plt.close(fig)

    # Separate oracle units instead of an arbitrary f+gradient cost scalar.
    fig, axes = plt.subplots(len(args.scenarios), 2, figsize=(15, 3*len(args.scenarios)), squeeze=False)
    for pair, scenario in zip(axes, args.scenarios):
        for ax, key in zip(pair, ("function_evals", "gradient_evals")):
            for method in plotted:
                for seed in range(args.seeds):
                    rows = traces[(scenario, seed, method)]
                    ax.loglog([r[key] for r in rows], np.maximum([r["certificate"] for r in rows], 1e-16),
                              color=colors[method], alpha=0.2, lw=0.8)
            ax.set(title=scenario, xlabel=key, ylabel="Certificate U; individual seeds")
            ax.grid(alpha=0.25, which="both")
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "oracle_costs.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1 or args.tolerance <= 0:
        parser.error("seeds, max-iter and tolerance must be positive")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("scenarios must not be repeated")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, sampled, traces, problem_manifest = [], [], {}, []
    for scenario in args.scenarios:
        for seed in range(args.seeds):
            problem = make_problem(seed, scenario)
            problem_manifest.append({"scenario": scenario, "seed": seed,
                                     "H0": problem.h0, "H1": problem.h1, "L_global": problem.global_l,
                                     "reference_lower": problem.reference_lower,
                                     "reference_upper": problem.reference_upper, **problem.metadata})
            for method in METHODS:
                result = solve(problem, method, args.max_iter, args.tolerance)
                row = result.trace[-1]
                identity = {"scenario": scenario, "seed": seed, "method": method}
                summary.append({**identity, "iterations": row["iteration"],
                                "certified": row["certificate"] <= args.tolerance,
                                "certificate": row["certificate"], "fw_gap": row["fw_gap"],
                                "error_lower": max(0, row["objective"] - problem.reference_upper),
                                "error_upper": max(0, row["objective"] - problem.reference_lower),
                                "objective": row["objective"],
                                "function_evals": row["function_evals"],
                                "gradient_evals": row["gradient_evals"], "lmo_evals": row["lmo_evals"],
                                "elapsed_seconds": result.elapsed_seconds, "support": row["support"],
                                "certificate_over_current_gap": row["certificate"] / max(row["fw_gap"], 1e-300),
                                "line_search_fallbacks": result.line_search_fallbacks})
                selected = [r for r in result.trace if r["iteration"] % 10 == 0 or r is result.trace[-1]]
                traces[(scenario, seed, method)] = selected
                sampled.extend({**identity, **r} for r in selected)
            print(f"{scenario}: seed {seed} complete", flush=True)
    aggregated = aggregate(summary, args.scenarios, args.seeds)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "trace.csv", sampled)
    write_csv(args.output / "aggregate.csv", aggregated)
    import scipy, sklearn, matplotlib
    manifest = {
        "research_revision_date": "2026-10-15", "command": [sys.executable, *sys.argv],
        "seeds": list(range(args.seeds)), "max_iterations": args.max_iter,
        "tolerance": args.tolerance, "trace_stride": 10,
        "scenarios": args.scenarios, "methods": METHODS,
        "python": sys.version, "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ("benchmark.py", "fw.py", "problems.py")},
        "problems": problem_manifest,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(args.output, aggregated, args)
    make_plots(args.output, traces, args)
    print(f"Saved {len(summary)} runs to {args.output}")


if __name__ == "__main__":
    main()
