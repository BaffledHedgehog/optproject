"""Build the AFW report, publication figure and table from archived CSVs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"away_results")
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    summary, budgets = read("summary.csv"), read("budget.csv")
    lines = ["# Away-step FW при кривизне, зависящей от значения функции", "",
             f"{len(summary)} запусков, семейств {len(manifest['scenarios'])}, seed {len(manifest['seeds'])}. "
             "Все методы начинают из одной исходной вершины, без обязательного начального LMO-перехода. "
             "FW_BT/AFW_BT используют обычное деление оценки пополам и удвоение при отказе, начальная оценка 1. "
             "FW_Q/AFW_Q получают аналитические H0,H1 и используют короткий квадратичный шаг по текущему U. "
             "Это существующие схемы FW/AFW; новым алгоритмом AFW не объявляется.", "",
             f"Допуск U≤{manifest['tolerance']:g}, ограничение {manifest['max_iterations']} обновлений. "
             "В U входит полный FW-gap, независимо от выбранного направления. Все пробы f оплачены, "
             "принятое значение повторно используется. Число градиентов и полных LMO равно N+1. "
             "У AFW отдельно записаны away_scans и active_atom_scores. Обновление и восстановление "
             "активных весов также требуют работы, поэтому одни счётчики LMO не равны полному времени. "
             "Время — по одному запуску экземпляра, без доказанного преимущества по wall-clock.", "",
             "Префиксы при отдельных бюджетах функции/градиента/LMO вычислены из полного журнала до "
             "его прореживания. Другие типы затрат при таком срезе не ограничиваются. В полном протоколе "
             "бюджет 1000 достигается до внешнего ограничения у незавершённых запусков. Медианы к концу "
             "включают ограниченные бюджетом запуски, поэтому даны и числа успехов.", ""]
    for scenario in manifest["scenarios"]:
        lines += [f"## {scenario}", "",
                  "| Метод | Успехи к концу | f | ∇f / LMO | U при ≤1000 f | Успехи при ≤1000 f | Удаления | Оценки активных атомов |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in manifest["methods"]:
            rows = [r for r in summary if r["scenario"] == scenario and r["method"] == method]
            fixed = [r for r in budgets if r["scenario"] == scenario and r["method"] == method
                     and r["oracle"] == "function_evals" and r["budget"] == "1000"]
            drops = f"{median(int(r['drop_steps']) for r in rows):.1f}" if method.startswith("AFW") else "—"
            scores = f"{median(int(r['active_atom_scores']) for r in rows):.1f}" if method.startswith("AFW") else "—"
            lines.append(f"| {method} | {sum(r['certified']=='True' for r in rows)}/{len(rows)} | "
                         f"{median(int(r['function_evals']) for r in rows):.1f} | "
                         f"{median(int(r['gradient_evals']) for r in rows):.1f} | "
                         f"{median(float(r['certificate']) for r in fixed):.3e} | "
                         f"{sum(r['certified']=='True' for r in fixed)}/{len(fixed)} | {drops} | {scores} |")
        lines.append("")
    lines += ["## Интерпретация полного протокола", "",
              "AFW_BT достигает допуска на всех 70 экземплярах. На active_scaled медиана 56.5 значений "
              "функции и 28.5 градиентов; обычный FW_BT не сертифицирует ни один из десяти запусков. "
              "На breast_cancer — 10/10 успехов при 103.5 значениях функции у AFW_BT против 0/10 у FW_BT. "
              "На bike_poisson к концу AFW_BT даёт 10/10, но при 1000 значений функции — только 6/10; "
              "это различие нельзя скрывать. FW_BT при том же фиксированном бюджете даёт 1/10.", "",
              "AFW_Q всё ещё не достигает допуска на обеих реальных регрессиях за 1000 обновлений: "
              "аналитические константы консервативны. Поэтому переход к away-направлениям сам по себе "
              "не делает любой шаг быстрым. На interior разница умеренная. Результаты относятся к "
              "этим экземплярам и не устанавливают превосходство над всеми FW-вариантами.", "",
              "Два FW-baseline повторяют 140 прежних траекторий; 140 AFW-запусков дополняют сравнение "
              "на тех же 70 задачах. Независимых новых наборов данных нет. Все 70 полных траекторий AFW_BT "
              "проверены по принятым кривизнам, фазовым границам, счёту удалений, точному телескопическому "
              "равенству проб и переходу от ошибки к сертификату. Неизвестная пирамидальная ширина "
              "не подгоняется по траекториям; универсальная проверка геометрической теоремы тестами "
              "не заявляется. Численные проверки не заменяют доказательство и аудит приоритета.", ""]
    project = next(parent for parent in root.resolve().parents if parent.name.upper() == "PART3")
    paper = project/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{FW versus AFW from the same original point, ten matched seeds. "
           r"Final medians include capped runs; the cap is 1000 updates. "
           r"The last columns use a separate budget of 1000 function calls.}",
           r"\label{tab:away}", r"\begin{tabular}{lrrrrr}", r"\hline",
           r"Method & Final successes & $f$ calls & $\nabla f$ calls & Successes at 1000 $f$ & Median $U$ \\", r"\hline"]
    shown = [s for s in ("active_scaled", "breast_cancer", "bike_poisson") if s in manifest["scenarios"]]
    for scenario in shown:
        tex.append(r"\multicolumn{6}{l}{\textbf{"+scenario.replace("_", r"\_")+r"}} \\")
        for method in manifest["methods"]:
            rows = [r for r in summary if r["scenario"] == scenario and r["method"] == method]
            fixed = [r for r in budgets if r["scenario"] == scenario and r["method"] == method
                     and r["oracle"] == "function_evals" and r["budget"] == "1000"]
            tex.append(method.replace("_", r"\_")+f" & {sum(r['certified']=='True' for r in rows)}/{len(rows)} & "
                       f"{median(int(r['function_evals']) for r in rows):.1f} & "
                       f"{median(int(r['gradient_evals']) for r in rows):.1f} & "
                       f"{sum(r['certified']=='True' for r in fixed)}/{len(fixed)} & "
                       f"{median(float(r['certificate']) for r in fixed):.2e}"+r" \\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    table = paper/"away_table.tex"
    table.write_text("\n".join(tex), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    shown = shown or manifest["scenarios"][:3]
    grid = [b for b in manifest["budgets"] if b <= 1000]
    lookup = {(r["scenario"], r["seed"], r["method"], r["budget"]): float(r["certificate"])
              for r in budgets if r["oracle"] == "function_evals"}
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig, axes = plt.subplots(1, len(shown), figsize=(6.5, 2.7), squeeze=False)
        for ax, scenario in zip(axes[0], shown):
            for method in manifest["methods"]:
                curves = np.maximum([[lookup[(scenario, str(seed), method, str(b))] for b in grid]
                                     for seed in manifest["seeds"]], 1e-16)
                lo, hi = np.quantile(curves, [0.25, 0.75], axis=0)
                line, = ax.loglog(grid, np.median(curves, axis=0), label=method.replace("_", " "), lw=1.2)
                ax.fill_between(grid, lo, hi, color=line.get_color(), alpha=0.1)
            ax.axhline(manifest["tolerance"], color="gray", ls=":", lw=0.8)
            ax.set(title=scenario, xlabel="Function calls", ylabel="Certificate U")
            ax.grid(alpha=0.2, which="both")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.5, w_pad=0.7)
        fig.savefig(root/"away_comparison.pdf")
        fig.savefig(root/"away_comparison.png", dpi=300)
        plt.close(fig)
    shutil.copy2(root/"away_comparison.pdf", paper/"figures"/"away_comparison.pdf")
    lines += ["## Рисунок", "", "![Сертификаты при одинаковом бюджете f](away_comparison.png)", "",
              "[Векторный PDF](away_comparison.pdf). Медиана и IQR; значения взяты из действительных префиксов. "
              "Между отмеченными бюджетами линии служат визуализации; завершённые значения сохраняются.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    destination = root/"report_source.py"
    if destination.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, destination)
    report_manifest = {"source": destination.name, "source_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                       "experiment_manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                       "report_sha256": hashlib.sha256((root/"results.md").read_bytes()).hexdigest(),
                       "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                       "matplotlib_version": matplotlib.__version__,
                       "figure_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                         for name in ("away_comparison.pdf", "away_comparison.png")}}
    (root/"report_manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    print(f"Generated {root/'results.md'} and {table}")


if __name__ == "__main__":
    main()
