"""Summarize isolated causal monitors, including their additional LMO costs."""

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
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"certificate_results")
    root = parser.parse_args().directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    summary, final, budgets = [read(n) for n in ("summary.csv", "final.csv", "budget.csv")]
    methods, profiles = manifest["methods"], manifest["profiles"]
    def key(r):
        return r["dataset"], r["radius_ratio"], r["start"], r["method"]
    scalar = {key(r): r for r in summary if r["profile"] == "scalar"}
    scalar_end = {key(r): r for r in final if r["profile"] == "scalar"}
    def select(rows, dataset, ratio, method, profile):
        return [r for r in rows if r["dataset"] == dataset and float(r["radius_ratio"]) == ratio
                and r["method"] == method and r["profile"] == profile]
    def successes(rows):
        return sum(r["certified"] == "True" for r in rows)
    def cost(rows, field="function_evals"):
        return median(int(r[field]) for r in rows)
    paired = []
    ends = {(*key(r), r["profile"]): r for r in final}
    for r in summary:
        base, end = scalar[key(r)], ends[(*key(r), r["profile"])]
        paired.append({**{k: r[k] for k in ("dataset", "radius_ratio", "start", "method", "profile")},
                       "certified": r["certified"], "scalar_certified": base["certified"],
                       "first_stop_iteration": r["iteration"], "scalar_stop_iteration": base["iteration"],
                       "earlier_prefix": int(r["iteration"]) < int(base["iteration"]),
                       "function_calls_saved": int(base["function_evals"])-int(r["function_evals"]),
                       "total_lmo_change": int(r["lmo_evals"])-int(base["lmo_evals"]),
                       "full_path_certificate": end["certificate"],
                       "scalar_full_path_certificate": scalar_end[key(r)]["certificate"]})
    with (root/"paired.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    lines = ["# Причинные сертификаты на неизменённых траекториях", "",
             "Повторены 432 исходные BT-траектории из опыта Seoul/Metro: два набора, два радиуса, "
             "все 30/24 различные вершины и четыре FW/AFW-варианта. Их исходные конечные значения, "
             "сертификаты и счётчики точно совпали с архивом. Пять мониторов дают 2160 доступных "
             "префиксов остановки, а не 2160 независимых оптимизаций. Новых данных и стартов нет.", "",
             "`scalar` — прежний максимум отдельных касательных; `block_8/32/128` — максимум с нижними "
             "оценками средних касательных по завершённым непересекающимся блокам. Один фактический "
             "дополнительный LMO оплачивается на блок; незавершённые блоки не вычисляются. Размеры "
             "8, 32, 128 заданы до получения результатов, без подбора по эталону. `known_mu` получает "
             "известный ridge-коэффициент 0.01 и использует стандартную оценку f−||g||²/(2μ). "
             "Последний контроль использует больше информации, его нельзя считать адаптацией неизвестной μ.", "",
             "Все мониторы видят только уже оплаченные f, g и геометрию Q. Они не меняют начальную "
             "кривизну, направления или line search. Записан первый префикс с сертификатом ≤1e−5 "
             "либо исходный конец при 1000 обновлениях. Причинность построения проверена 20 нативными "
             "запусками с callback на синтетическом внутреннем экземпляре: все сочетания четырёх "
             "методов и пяти профилей; доступ к будущим точкам/эталону исключён. "
             "Траектория после остановки используется только в `final.csv` для диагностики общего конца.", "",
             "Полная стоимость LMO отдельного профиля = LMO направлений + LMO сертификата. "
             "Эксперимент вычисляет все мониторы вместе, но для каждого показывает стоимость его "
             "отдельного исполнения. При бюджетном срезе нужна оплата всей запланированной проверки "
             "в точке. Бюджеты f, g и LMO рассматриваются отдельно; активные сканирования AFW остаются "
             "дополнительной работой. Сравнения полного времени нет.", "",
             "## Остановка к исходному пределу траектории", "",
             "Ячейка: число сертифицированных / число стартов; медиана вызовов f до новой остановки "
             "либо предела. Медиана включает незавершённые случаи и не является оценкой времени до успеха.", "",
             "| Данные, R/R* | Метод | scalar | block 8 | block 32 | block 128 | known μ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in manifest["datasets"]:
        for ratio in manifest["radius_ratios"]:
            for method in methods:
                cells = []
                for profile in profiles:
                    rows = select(summary, dataset, ratio, method, profile)
                    cells.append(f"{successes(rows)}/{len(rows)}; {cost(rows):g}")
                lines.append(f"| {dataset}, {ratio:g} | {method} | "+" | ".join(cells)+" |")
    lines += ["", "## Достижение цели при равном бюджете", "",
              "Число сертифицированных к 1000 вызовам указанного оракула; учитывается дополнительный LMO. "
              "Исходные траектории не продолжаются за прежний предел.", "",
              "| Данные, R/R* | Метод | Бюджет | scalar | block 8 | block 32 | block 128 | known μ |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in manifest["datasets"]:
        for ratio in manifest["radius_ratios"]:
            for method in methods:
                for oracle in ("function_evals", "lmo_evals"):
                    cells = []
                    for profile in profiles:
                        rows = [r for r in select(budgets, dataset, ratio, method, profile)
                                if r["oracle"] == oracle and r["budget"] == "1000"]
                        cells.append(f"{successes(rows)}/{len(rows)}")
                    lines.append(f"| {dataset}, {ratio:g} | {method} | {oracle} | "+" | ".join(cells)+" |")
    lines += ["", "## Парное сравнение с scalar", "",
              "| Профиль | Успехи / 432 | Раньше исходного конца | Больше полных LMO до остановки/предела | "
              "Медиана отношения сертификатов на общем конце |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for profile in profiles:
        rows = [r for r in paired if r["profile"] == profile]
        ratios = [float(r["full_path_certificate"])/float(r["scalar_full_path_certificate"]) for r in rows
                  if float(r["scalar_full_path_certificate"]) > 0]
        lines.append(f"| {profile} | {sum(r['certified']=='True' for r in rows)}/432 | "
                     f"{sum(r['earlier_prefix'] for r in rows)} | {sum(r['total_lmo_change']>0 for r in rows)} | "
                     f"{median(ratios):.4g} |")
    lines += ["", "## Что можно заключить", "",
              "**Отрицательный результат блоков:** ни один из трёх размеров не даёт более ранней "
              "остановки на этих 432 траекториях: у всех остаётся 129/432 успехов. Это не означает "
              "полного совпадения нижних оценок: на общем конце K=8 улучшает сертификат в "
              f"{sum(float(r['full_path_certificate']) < float(r['scalar_full_path_certificate'])*(1-1e-10) for r in paired if r['profile']=='block_8')} случаях, "
              "а K=32 — в "
              f"{sum(float(r['full_path_certificate']) < float(r['scalar_full_path_certificate'])*(1-1e-10) for r in paired if r['profile']=='block_32')}. "
              "Улучшения недостаточно для пересечения цели. Не следует обобщать этот неуспех "
              "на оптимальные веса, другие размеры блоков или точные Fenchel dual-сертификаты.", "",
              "**Контроль с известной μ:** 226/432 успехов, 110 более ранних префиксов. На внутреннем "
              "Metro каждый из четырёх BT-методов сертифицирует 24/24 старта. Для FW медианы f "
              "становятся 746.5 и 724.5 вместо ограниченных бюджетом 2000 и 2002; для AFW — "
              "481.5 и 458 вместо 2000 и 2001. При 1000 f это 18/24, 15/24 для FW и по 24/24 "
              "для AFW. На внутреннем Seoul AFW получает 4/30 и 10/30 успехов лишь к полному "
              "пределу; при 1000 f все варианты всё ещё без успехов. В активных режимах число "
              "успехов не меняется. Таким образом, контроль объясняет часть certificate-lag, "
              "но не устраняет все неуспехи и использует аналитическую μ.", "",
              "Добавление корректных нижних оценок гарантирует неухудшение сертификата на том же источнике "
              "и остановку не позже в числе обновлений/f/g. Для полной стоимости LMO такого порядка нет: "
              "дополнительные запросы могут не окупиться. Размер блока — параметр этой цены, не новый закон сходимости.", "",
              "Усреднение касательных и оценка через μ — стандартные конструкции. Наш блоковый bound "
              "может быть слабее точного Fenchel dual в усреднённом градиенте; теорема 3 "
              "[Pedregosa et al.](https://arxiv.org/html/1806.05123v5) относится к их dual и весам. "
              "Оценка через норму градиента дана, например, в приложении B "
              "[Karimi–Nutini–Schmidt](https://arxiv.org/html/1608.04636v4). "
              "Их приоритет явно учитывается. Даже положительный результат этого контроля не "
              "устанавливает новизну алгоритма или универсальное преимущество сертификации.", "",
              "Проверки до прореживания: все нижние оценки ≤ независимой верхней границы optimum, "
              "сертификат ≥ нижней границы primal-ошибки, неухудшение относительно scalar, "
              "монотонность и точное расписание платных LMO. `paired.csv` сохраняет все парные "
              "изменения; `final.csv` — сравнение сертификатов на общих концах. "
              "Старты, радиусы и профили на одном наборе зависимы; статистика не трактует их как новые независимые задачи.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    paper = next(p for p in root.resolve().parents if p.name == "PART3")/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Interior causal stopping controls. Each cell is the certified count and median "
           r"function calls up to its first stop or the original cap; censored cases are included. "
           r"Seoul has 30 starts per method, Metro 24. $F/A$: FW/AFW; subscripts $1,U$: "
           r"initial $M=1,U_0/D^2$. Blocks use one additional full LMO per completed block; "
           r"the known-$\mu$ column receives extra problem information.}",
           r"\label{tab:certificates}", r"\begin{tabular}{llrrrrr}", r"\hline",
           r"Data & Method & Scalar & $K=8$ & $K=32$ & $K=128$ & Known $\mu$ \\", r"\hline"]
    for dataset in manifest["datasets"]:
        for method, label in zip(methods, (r"$F_1$", r"$F_U$", r"$A_1$", r"$A_U$")):
            cells = []
            for profile in profiles:
                rows = select(summary, dataset, 2., method, profile)
                cells.append(f"{successes(rows)}; {cost(rows):g}")
            tex.append(dataset.capitalize()+" & "+label+" & "+" & ".join(cells)+r" \\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (root/"certificate_table.tex").write_text("\n".join(tex), encoding="utf-8")
    shutil.copy2(root/"certificate_table.tex", paper/"certificate_table.tex")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
    shown = [b for b in manifest["budgets"] if b <= 1000]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharex=True, sharey=True)
    for i, dataset in enumerate(manifest["datasets"]):
        for j, oracle in enumerate(("function_evals", "lmo_evals")):
            ax = axes[i, j]
            for profile, label, color, style in zip(profiles, ("Scalar", "Block 8", "Block 32", "Block 128", r"Known $\mu$"),
                                                   ("#222222", "#d55e00", "#0072b2", "#009e73", "#cc79a7"),
                                                   ("-", "--", "-.", ":", "-")):
                values = [[float(r["certificate"]) for r in select(budgets, dataset, 2., "AFW_BT_certificate", profile)
                           if r["oracle"] == oracle and int(r["budget"]) == b] for b in shown]
                mid = [median(v) for v in values]
                ax.plot(shown, np.maximum(mid, 1e-16), color=color, linestyle=style, label=label, linewidth=1.1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(manifest["tolerance"], color=".5", linestyle=":", linewidth=.7)
            ax.set_title(dataset.capitalize()+", interior", fontsize=8)
            ax.set_xlabel("Function-call budget" if j == 0 else "Total LMO budget")
            if j == 0:
                ax.set_ylabel("Median certificate")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(root/f"causal_certificates.{extension}", dpi=220)
    plt.close(fig)
    shutil.copy2(root/"causal_certificates.pdf", paper/"figures"/"causal_certificates.pdf")
    if Path(__file__).resolve() != (root/"report_source.py").resolve():
        shutil.copy2(Path(__file__), root/"report_source.py")
    artifacts = ("manifest.json", "report_source.py", "results.md", "paired.csv", "certificate_table.tex",
                 "causal_certificates.pdf", "causal_certificates.png")
    (root/"report_manifest.json").write_text(json.dumps({
        "artifact_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in artifacts}}, indent=2), encoding="utf-8")
    print(f"Report: {len(summary)} dependent profiles on 432 paths; table and vector figure saved")


if __name__ == "__main__":
    main()
