"""Build additional-dataset report, table and vector figure from archived CSVs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"multidata_results")
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    summary, budgets = read("summary.csv"), read("budget.csv")
    methods, datasets, ratios = [manifest[k] for k in ("methods", "datasets", "radius_ratios")]
    def select(rows, dataset, ratio, method):
        return [r for r in rows if r["dataset"] == dataset and float(r["radius_ratio"]) == ratio and r["method"] == method]
    fixed = [r for r in budgets if r["oracle"] == "function_evals" and r["budget"] == "1000"]
    def successes(rows):
        return sum(r["certified"] == "True" for r in rows)
    def median_f(rows):
        return median(int(r["function_evals"]) for r in rows)
    def count_text(dataset, ratio, method, reference=False):
        rows = select(summary, dataset, ratio, method)
        n = sum(float(r["error_upper"]) <= manifest["tolerance"] for r in rows) if reference else successes(rows)
        return f"{n}/{len(rows)}"
    diagnostics = [{**{key: r[key] for key in ("dataset", "radius_ratio", "start", "method")},
                    "online_certified": r["certified"], "reference_error_upper": r["error_upper"],
                    "reference_within_target": float(r["error_upper"]) <= manifest["tolerance"],
                    "certificate": r["certificate"]} for r in summary]
    with (root/"reference_diagnostics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    lines = ["# Два дополнительных реальных набора счётных данных", "",
             f"{len(summary)} запусков существующих FW/AFW, {len(datasets)} новых набора данных, "
             f"{len(ratios)} режима радиуса, все 2d знаковые координатные вершины без случайных повторов. "
             "Это расширяет прежний опыт на одном наборе Bike Sharing; радиусы и старты одного набора "
             "не считаются независимыми выборками. Прогнозное качество не оценивается.", "",
             "## Данные и функция", "",
             "- [Seoul Bike Sharing Demand, UCI](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), "
             "2020, [doi:10.24432/C5F62R](https://doi.org/10.24432/C5F62R), CC BY 4.0.",
             "- [Metro Interstate Traffic Volume, John Hogue, UCI](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), "
             "2019, [doi:10.24432/C5X60B](https://doi.org/10.24432/C5X60B), CC BY 4.0.", "",
             "Оригинальные файлы сохранены побайтно, SHA-256 и атрибуция — Data/*/source.json. "
             "Хронологически первые 80% строк определяют цель; повторы исходных временных меток "
             "сохранены с исходными весами. Все неинтерсептные признаки преобразуются как tanh((v−mean)/std), "
             "статистики только обучающей части. Это новый фиксированный протокол, а не воспроизведение "
             "линейного масштабирования прежнего Bike Sharing. Оставлены заранее заданные погодные "
             "и циклические временные признаки, holiday и для Сеула functioning; прочие текстовые категории опущены.", "",
             "Цель — средний нормированный пуассоновский deviance плюс 0.01||x||²/2; "
             "счётчики делятся на среднее обучающей части. Интерсепт также ограничен и регуляризуется. "
             "Поскольку ||a_i||∞=1 и среднее нормированной цели равно 1, корректны "
             "H1=max||a_i||², H0=µ+H1(1+R), L_Q=µ+exp(R)λ_max(AᵀA/m). "
             "Эти границы консервативны. H1>0 — допустимый коэффициент модели, а не доказанная "
             "необходимость неглобальной гладкости на компактном шаре.", "",
             "Независимый стационарный минимум определяет R*=||x_unc||1. При R=R*/2 эталон активный, "
             "при R=2R* — внутренний. Для каждого шара пересчитан ограниченный эталон и линейный "
             "интервал f*. Численные остатки вычислены в float64, без строгой интервальной арифметики. "
             "Эталон и R* не передаются оптимизационным алгоритмам.", "",
             "| Набор | Строки | Размерность | R* | R/R* | Режим | Ширина интервала f* | H0 | H1 | L_Q |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for p in manifest["problems"]:
        lines.append(f"| {p['dataset']} | {p['rows']} | {p['dimension']} | {p['critical_radius']:.6f} | {p['radius_ratio']:g} | "
                     f"{p['regime']} | {p['reference_upper']-p['reference_lower']:.3e} | {p['H0']:.3f} | {p['H1']:.3f} | {p['global_L']:.3f} |")
    lines += ["", "## Протокол сравнения", "",
              f"Допуск U≤{manifest['tolerance']:g}, предел {manifest['max_iterations']} обновлений. "
              "FW_BT_fixed/AFW_BT_fixed используют M=1; certificate — M=U0/D² из первых оплаченных "
              "оракулов. FW_Q/AFW_Q получают аналитические H0,H1 и шаг по H0+H1*U. "
              "Принятые f кешируются. Во всех вариантах N+1 градиентов и полных LMO, Q также "
              "использует N+1 значений f, а BT оплачивает все отклонения. AFW дополнительно "
              "сканирует активные атомы; эти затраты не равны одному полному LMO. Рейтинга полного времени нет.", "",
              "Оракульные префиксы получены из полных журналов до прореживания. Бюджет одного "
              "оракула не ограничивает остальные типы работы. Медианы к концу включают неуспехи "
              "на внешнем пределе; поэтому их нельзя трактовать как время до решения без долей успеха.", ""]
    for dataset in datasets:
        for ratio in ratios:
            lines += [f"## {dataset}, R/R*={ratio:g}", "",
                      "| Метод | Остановка U к концу | e_upper≤допуск к концу | Медиана e_upper | Медиана f | Медиана ∇f/LMO | Успехи при ≤1000 f | Медиана U при ≤1000 f |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for method in methods:
                rows, fixed_rows = [select(data, dataset, ratio, method) for data in (summary, fixed)]
                lines.append(f"| {method} | {successes(rows)}/{len(rows)} | "
                             f"{sum(float(r['error_upper'])<=manifest['tolerance'] for r in rows)}/{len(rows)} | "
                             f"{median(float(r['error_upper']) for r in rows):.3e} | {median_f(rows):g} | "
                             f"{median(int(r['gradient_evals']) for r in rows):g} | "
                             f"{successes(fixed_rows)}/{len(fixed_rows)} | "
                             f"{median(float(r['certificate']) for r in fixed_rows):.3e} |")
            lines.append("")
    lines += ["## Что ограничивает завершение", "",
              f"Активный Seoul: остановки AFW-BT с M=1 и M=U0/D² — "
              f"{count_text('seoul',0.5,'AFW_BT_fixed')} и {count_text('seoul',0.5,'AFW_BT_certificate')}; "
              f"активный Metro — {count_text('metro',0.5,'AFW_BT_fixed')} и {count_text('metro',0.5,'AFW_BT_certificate')}. "
              f"Внутренний Seoul — {count_text('seoul',2,'AFW_BT_fixed')} и {count_text('seoul',2,'AFW_BT_certificate')}; "
              f"внутренний Metro — {count_text('metro',2,'AFW_BT_fixed')} и {count_text('metro',2,'AFW_BT_certificate')}. "
              "Эти числа ограничивают перенос прежних положительных результатов на новые режимы. "
              "Аналитические Q-шаги необходимо оценивать по собственным долям завершения и затратам.", "",
              "Для анализа используется e_upper=max(0,f(x)−reference_lower), вычисленное после "
              "запуска из независимого эталонного интервала. Это не вход алгоритма и не критерий "
              f"его остановки. На внутреннем Metro e_upper≤{manifest['tolerance']:g} имеют "
              f"{count_text('metro',2,'FW_BT_fixed',True)} и {count_text('metro',2,'FW_BT_certificate',True)} FW-стартов, "
              f"{count_text('metro',2,'AFW_BT_fixed',True)} и {count_text('metro',2,'AFW_BT_certificate',True)} AFW-стартов. "
              f"На внутреннем Seoul соответствующие AFW-числа — {count_text('seoul',2,'AFW_BT_fixed',True)} "
              f"и {count_text('seoul',2,'AFW_BT_certificate',True)}. "
              "Значит, незавершение может отражать и консервативность доступного сертификата, "
              "и оставшуюся ошибку функции. Подменять реальные онлайн-остановки этим "
              "послепусковым анализом нельзя. Полные признаки сохранены в reference_diagnostics.csv.", "",
              "## Граница вывода", "",
              "Полный перебор вершин устраняет случайные повторы стартов, но не покрывает все "
              "внутренние точки и не создаёт независимые наборы данных. Два новых источника дают "
              "более широкую проверку, чем одно семейство. Не установлена необходимость "
              "конкретной нормировки шага, преимущество над всеми projection-free методами или "
              "универсальная практическая польза аналитической пары H0,H1. Эти ограничения "
              "читаются вместе с результатами всех шести baseline, включая незавершённые запуски.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    project = next(p for p in root.resolve().parents if p.name.upper() == "PART3")
    paper = project/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Additional count datasets, every signed coordinate vertex. Entries: "
           f"successes; median final function calls including failures capped at {manifest['max_iterations']} updates. "
           r"Seoul has 30 starts per radius, Metro 24. $F/A$: FW/AFW; "
           r"subscripts $1,U$: initial $M=1,U_0/D^2$; $Q$: known-$H$ short step.}",
           r"\label{tab:multidata}", r"\begin{tabular}{lrrrrrr}", r"\hline",
           r"$R/R_*$ & $F_1$ & $F_U$ & $A_1$ & $A_U$ & $F_Q$ & $A_Q$ \\", r"\hline"]
    for dataset in datasets:
        tex.append(r"\multicolumn{7}{l}{\textbf{"+dataset+r"}} \\")
        for ratio in ratios:
            cells = []
            for method in methods:
                rows = select(summary, dataset, ratio, method)
                cells.append(f"{successes(rows)}; {median_f(rows):g}")
            tex.append(f"{ratio:g} & "+" & ".join(cells)+r" \\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (paper/"multidata_table.tex").write_text("\n".join(tex), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
    shown_budgets = [b for b in manifest["budgets"] if b <= 1000]
    fig, axes = plt.subplots(len(datasets), len(ratios), figsize=(6.5, 4.0), sharex=True, sharey=True, squeeze=False)
    labels = (r"FW $M=1$", r"FW $M=U_0/D^2$", r"AFW $M=1$", r"AFW $M=U_0/D^2$", "FW Q", "AFW Q")
    for i, dataset in enumerate(datasets):
        for j, ratio in enumerate(ratios):
            ax = axes[i, j]
            for method, label, color, style in zip(methods, labels,
                                                  ("#d55e00", "#e69f00", "#0072b2", "#009e73", "#cc79a7", "#555555"),
                                                  ("-", "--", "-", "--", ":", "-.")):
                values = [[float(r["certificate"]) for r in select(budgets, dataset, ratio, method)
                           if r["oracle"] == "function_evals" and int(r["budget"]) == b] for b in shown_budgets]
                mid, low, high = [np.array([np.quantile(v, q) for v in values]) for q in (0.5, 0.25, 0.75)]
                ax.plot(shown_budgets, mid, label=label, color=color, linestyle=style, linewidth=1.1)
                ax.fill_between(shown_budgets, low, high, color=color, alpha=0.06)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(manifest["tolerance"], color="0.5", linestyle=":", linewidth=0.8)
            ax.set_title(f"{dataset.capitalize()}, R/R*={ratio:g}", fontsize=8)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("Certificate U")
            if i == len(datasets)-1:
                ax.set_xlabel("Function-call budget")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=0.8)
    for extension in ("png", "pdf"):
        fig.savefig(root/f"additional_counts.{extension}", dpi=220)
    plt.close(fig)
    shutil.copy2(root/"additional_counts.pdf", paper/"figures"/"additional_counts.pdf")
    shutil.copy2(Path(__file__), root/"report_source.py")
    artifacts = ("manifest.json", "report_source.py", "results.md", "reference_diagnostics.csv", "additional_counts.pdf", "additional_counts.png")
    (root/"report_manifest.json").write_text(json.dumps({
        "artifact_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in artifacts},
        "paper_table_sha256": hashlib.sha256((paper/"multidata_table.tex").read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"Report from {len(summary)} runs, table and vector figure saved")


if __name__ == "__main__":
    main()
