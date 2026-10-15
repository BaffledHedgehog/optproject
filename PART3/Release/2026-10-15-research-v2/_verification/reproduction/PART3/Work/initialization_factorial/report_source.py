"""Generate the factorial report and manuscript table from immutable CSVs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median


def publication_figure(root, budgets, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    shown = [s for s in ("active", "active_scaled", "bike_poisson") if s in manifest["scenarios"]]
    if not shown:
        shown = manifest["scenarios"][:3]
    grid = [b for b in manifest["budgets"] if b <= 1000]
    lookup = {(r["scenario"], r["seed"], r["cell"], r["budget"]): float(r["certificate"])
              for r in budgets if r["oracle"] == "function_evals"}
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig, axes = plt.subplots(1, len(shown), figsize=(6.5, 2.7), squeeze=False)
        for ax, scenario in zip(axes[0], shown):
            for cell in manifest["cells"]:
                values = np.maximum([[lookup[(scenario, str(seed), cell, str(b))] for b in grid]
                                     for seed in manifest["seeds"]], 1e-16)
                center = np.median(values, axis=0)
                lo, hi = np.quantile(values, [0.25, 0.75], axis=0)
                line, = ax.loglog(grid, center, label=cell.replace("_", ", "), lw=1.2)
                ax.fill_between(grid, lo, hi, color=line.get_color(), alpha=0.1)
            ax.axhline(manifest["tolerance"], color="gray", linestyle=":", lw=0.8)
            ax.set(title=scenario, xlabel="Function calls", ylabel="Certificate U")
            ax.grid(alpha=0.2, which="both")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.5, w_pad=0.7)
        fig.savefig(root/"initialization_publication.png", dpi=300)
        fig.savefig(root/"initialization_publication.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"initialization_factorial")
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    summary, budgets = read("summary.csv"), read("budget.csv")
    lines = ["# Контроль начальной точки и правила адаптации", "",
             f"{len(summary)} запусков: четыре условия, семейств {len(manifest['scenarios'])}, seed {len(manifest['seeds'])}. "
             "Использованы прежние семейства задач. Ячейка AC_vertex воспроизводит прежний AC-FW; "
             "на полном протоколе все 70 её запусков совпали с прежними результатами. "
             "Три остальных условия — контролируемые варианты, а не заявленные алгоритмы авторов AC-FW.", "",
             "В каждом условии оплачены f, градиент и LMO в исходной точке x и в новой вершине "
             "z=LMO(grad f(x)): по два вызова каждого оракула. Оба линейных нижних ограничения и "
             "секущая оценка кривизны доступны всем. AC использует эту оценку первой; у BT начальное "
             "хранимое значение вдвое больше, чтобы после деления пополам первая проба совпала. "
             "BT затем делит оценку пополам и удваивает при отказе. AC использует затухание по (6) "
             "arXiv:2605.15512v2 с delta=1. Эти настройки фиксированы без подбора по задачам.", "",
             f"Допуск U≤{manifest['tolerance']:g}, ограничение {manifest['max_iterations']} внешних проб. "
             "Отказ AC оплачивается и расходует внешнюю итерацию; BT проверяет несколько проб внутри неё. "
             "Отдельный budget.csv содержит последний полностью доступный сертификат при заданном бюджете "
             "каждого оракула, остальные типы вызовов не ограничиваются. Это точные префиксы полного журнала, "
             "а не интерполяция по сохранённому trace.csv. После достижения допуска значение сохраняется. "
             "На рисунке при больших бюджетах сохраняется и последнее значение ограниченных числом итераций "
             "запусков; основной срез по f использует бюджет 1000, который достигается до внешнего ограничения "
             "в любом незавершённом запуске.", ""]
    for scenario in manifest["scenarios"]:
        lines += [f"## {scenario}", "",
                  "| Условие | Успехи к концу | Медиана f | Медиана ∇f | U при ≤1000 f | Успехи при ≤1000 f | U при ≤1000 ∇f | Успехи при ≤1000 ∇f |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for cell in manifest["cells"]:
            rows = [r for r in summary if r["scenario"] == scenario and r["cell"] == cell]
            fixed = {o: [r for r in budgets if r["scenario"] == scenario and r["cell"] == cell
                         and r["oracle"] == o and r["budget"] == "1000"]
                     for o in ("function_evals", "gradient_evals")}
            f, g = fixed["function_evals"], fixed["gradient_evals"]
            lines.append(f"| {cell} | {sum(r['certified']=='True' for r in rows)}/{len(rows)} | "
                         f"{median(int(r['function_evals']) for r in rows):.1f} | "
                         f"{median(int(r['gradient_evals']) for r in rows):.1f} | "
                         f"{median(float(r['certificate']) for r in f):.3e} | "
                         f"{sum(r['certified']=='True' for r in f)}/{len(f)} | "
                         f"{median(float(r['certificate']) for r in g):.3e} | "
                         f"{sum(r['certified']=='True' for r in g)}/{len(g)} |")
        lines.append("")
    lines += ["## Интерпретация полного протокола (семь семейств, десять seed)", "",
              "На active переход к новой вершине меняет успехи AC с 1/10 на 10/10 и BT с 0/10 на 10/10 "
              "при бюджете 1000 значений функции. Поэтому прежние 10/10 AC-FW против 0–1/10 обычных "
              "вариантов не выделяли вклад последующего правила адаптации. Это конкретный результат для "
              "этих экземпляров, а не общий закон о выборе стартовой вершины.", "",
              "На bike_poisson успехи и распределение затрат меняются неодинаково. При бюджете 1000 f "
              "у AC исходная/новая точка дают 6/10 и 7/10, у BT — 2/10 и 5/10. При бюджете 1000 ∇f "
              "BT из исходной точки даёт 6/10, как AC. Порядок методов зависит от стоимости оракула; "
              "фиксированная сумма затрат с произвольными весами не используется.", "",
              "paired_comparisons.csv содержит 84 описательных сравнения при бюджете 1000: медиану "
              "индивидуальных парных отношений сертификатов и percentile-bootstrap интервалы "
              "(seed 20261015, 10000 повторов). Поправки на множественные сравнения нет. "
              "На уже завершённых запусках сертификат находится около допуска; отношение таких "
              "сертификатов не является коэффициентом ускорения. На Bike Sharing seed меняет старт "
              "одного обучающего функционала, а не независимый набор данных.", "",
              "Геометрия трёх активных семейств проверяется отдельно: face_summary.csv и face_trace.csv "
              "используют эталонную опору только после завершения оптимизации. Линейное тождество "
              "σ(x+α(s−x))=(1−α)σ(x)+ασ(s), σ(x)=R−sign(x*)ᵀx, объясняет сохранение массы вне "
              "оптимальной грани при неполных шагах. Эта геометрия известна в анализе FW; новым "
              "механизмом ускорения она не объявляется.", ""]
    lines += ["## Рисунок для статьи", "",
              "![Сертификаты при отдельных бюджетах значений функции](initialization_publication.png)", "",
              "[Векторный PDF](initialization_publication.pdf). Линии между указанными бюджетами "
              "служат только для визуализации; значения в budget.csv взяты из действительных префиксов.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    project = next(p for p in root.resolve().parents if p.name.upper() == "PART3")
    paper = project/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Controlled starting-point/update-rule ablation. Each cell has the same "
           r"preprocessing cost and first trial curvature. Successes and median certificates "
           r"at separate budgets of 1000 function or gradient calls, over ten seeds. "
           r"Other oracle types are unrestricted; the outer cap remains 1000.}",
           r"\label{tab:factorial}", r"\begin{tabular}{llrrrr}", r"\hline",
           r" & & \multicolumn{2}{c}{1000 function calls} & \multicolumn{2}{c}{1000 gradient calls} \\",
           r"Rule & Start & Successes & Median $U$ & Successes & Median $U$ \\", r"\hline"]
    for scenario in ("active", "active_scaled", "bike_poisson"):
        if scenario not in manifest["scenarios"]:
            continue
        tex += [r"\multicolumn{6}{l}{\textbf{"+scenario.replace("_", r"\_")+r"}} \\"]
        for cell in manifest["cells"]:
            rule, point = cell.split("_")
            fixed = [[r for r in budgets if r["scenario"] == scenario and r["cell"] == cell
                      and r["oracle"] == oracle and r["budget"] == "1000"]
                     for oracle in ("function_evals", "gradient_evals")]
            values = [f"{sum(r['certified']=='True' for r in rows)}/{len(rows)} & "
                      f"{median(float(r['certificate']) for r in rows):.2e}" for rows in fixed]
            tex.append(f"{rule} & {point} & "+" & ".join(values)+r" \\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    table = paper/"initialization_table.tex"
    table.write_text("\n".join(tex), encoding="utf-8")
    publication_figure(root, budgets, manifest)
    shutil.copy2(root/"initialization_publication.pdf", paper/"figures"/"initialization_effect.pdf")
    destination = root/"report_source.py"
    if destination.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, destination)
    provenance = {"source": destination.name, "source_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "experiment_manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                  "report_sha256": hashlib.sha256((root/"results.md").read_bytes()).hexdigest(),
                  "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                  "figure_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in ("initialization_effect.png", "initialization_effect.pdf",
                                                 "initialization_publication.png", "initialization_publication.pdf")}}
    (root/"report_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Generated {root/'results.md'} and {table}")


if __name__ == "__main__":
    main()
