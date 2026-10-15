"""Derive radius/units tables and plots from immutable experiment records."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median

import numpy as np


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"robustness_results")
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    if 1.0 not in manifest["objective_scales"]:
        parser.error("Scale 1 is required for the paired units report")
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    summary, budgets = read("summary.csv"), read("budget.csv")
    scenarios, factors, methods = [manifest[k] for k in ("scenarios", "radius_factors", "methods")]
    scales = manifest["objective_scales"]
    def select(rows, scenario, factor, scale, method):
        return [r for r in rows if r["scenario"] == scenario and float(r["radius_factor"]) == factor
                and float(r["scale"]) == scale and r["method"] == method]
    fixed = [r for r in budgets if r["oracle"] == "function_evals" and r["budget"] == "1000"]
    def success(rows):
        return f"{sum(r['certified']=='True' for r in rows)}/{len(rows)}"
    def run_key(row):
        return (row["scenario"], row["radius_factor"], row["seed"], row["method"])
    base_summary = {run_key(r): r for r in summary if float(r["scale"]) == 1}
    base_budgets = {(*run_key(r), r["oracle"], r["budget"]): r for r in budgets if float(r["scale"]) == 1}
    checks = []
    for row in summary:
        if float(row["scale"]) == 1:
            continue
        baseline = base_summary[run_key(row)]
        matched = [r for r in budgets if run_key(r) == run_key(row) and r["scale"] == row["scale"]]
        differences = []
        for r in matched:
            b = base_budgets[(*run_key(r), r["oracle"], r["budget"])]
            differences.append(abs(float(r["normalized_certificate"])-float(b["normalized_certificate"]))
                               / max(manifest["base_tolerance"], abs(float(b["normalized_certificate"]))))
        checks.append({key: row[key] for key in ("scenario", "radius_factor", "seed", "scale", "method")}
                      | {"same_final_counts": all(row[k] == baseline[k] for k in
                                                  ("function_evals", "gradient_evals", "lmo_evals")),
                         "same_outcome": row["certified"] == baseline["certified"],
                         "function_count_change": int(row["function_evals"])-int(baseline["function_evals"]),
                         "max_relative_prefix_certificate_change": max(differences)})
    write_csv(root/"scale_checks.csv", checks)
    nseeds = len(manifest["seeds"])
    lines = ["# Устойчивость к радиусу ограничения и единицам функции", "",
             f"{len(summary)} запусков, {len(manifest['problems'])} условий «семейство × радиус × seed», "
             f"{len(scales)} масштаба. Масштабы — парные контроли одной задачи, не независимые наблюдения. "
             "Логистические seed меняют обучающую выборку; пуассоновские seed меняют только исходную вершину. "
             "Новых наборов данных нет.", "",
             f"Радиус умножается на {factors}; функция при смене радиуса остаётся прежней. "
             "Для каждого радиуса заново решена эталонная ограниченная задача и вычислен её FW-gap. "
             "Пуассоновские H0 и глобальная L пересчитаны на новом шаре. Старый эталон нельзя просто "
             "перенести на другое множество. Эти значения недоступны сравниваемым алгоритмам.", "",
             f"Цель a*f, a∈{scales}, допуск a*{manifest['base_tolerance']:g}. В таблицах U/a возвращает "
             "сертификат в исходные единицы. H0, глобальная L и µ умножаются на a, H1 остаётся прежним. "
             "Варианты fixed используют M=1; certificate — M=U0/(2R)^2, вычисленное из первого уже "
             "оплаченного f, градиента и LMO. Обязательного начального перехода нет.", "",
             f"Ограничение {manifest['max_iterations']} обновлений. Отдельные бюджеты f/градиента/LMO "
             "получены до прореживания журналов. Срез по одному оракулу не ограничивает остальные затраты. "
             "В таблицах медиана f к концу включает незавершённые запуски и читается вместе с успехами. "
             "Активные сканирования AFW оплачены отдельными счётчиками; преимущество по времени не утверждается.", ""]
    for scenario in scenarios:
        for factor in factors:
            radius = next(p["radius"] for p in manifest["problems"] if p["scenario"] == scenario and p["radius_factor"] == factor)
            lines += [f"## {scenario}, R={radius:g}", "",
                      "| Масштаб a | Метод | Успехи к концу | Медиана f к концу | Успехи при ≤1000 f | Медиана U/a при ≤1000 f |",
                      "| ---: | --- | ---: | ---: | ---: | ---: |"]
            for scale in scales:
                for method in methods:
                    rows = select(summary, scenario, factor, scale, method)
                    prefix = select(fixed, scenario, factor, scale, method)
                    lines.append(f"| {scale:g} | {method} | {success(rows)} | "
                                 f"{median(int(r['function_evals']) for r in rows):.1f} | {success(prefix)} | "
                                 f"{median(float(r['normalized_certificate']) for r in prefix):.3e} |")
            lines.append("")
    poisson = [p for p in manifest["problems"] if p["scenario"] == "bike_poisson"]
    interior = sum(p["reference_l1_norm"] < p["radius"]-1e-6 for p in poisson)
    lines += [f"Из {len(poisson)} пуассоновских эталонов {interior} находятся строго внутри шара. "
              "При всех трёх основных радиусах пуассоновский минимум остаётся внутренним; "
              "активное пуассоновское ограничение этим опытом не проверяется. "
              "У логистических задач эталон находится на границе. Изменение радиуса меняет "
              "и допустимое множество, и исходную вершину, поэтому не изолирует влияние одной геометрической константы.", "",
              "## Парная проверка масштаба", "",
              "Сравнение каждого a≠1 с a=1 на том же радиусе и seed; полные результаты — scale_checks.csv. "
              "Сравниваются конечные счётчики и сертификаты в 33 префиксах (11 бюджетов × 3 оракула). "
              "Это проверка записанных характеристик, а не доказательство побитового совпадения всех итератов.", "",
              "| Метод | Совпали конечные счётчики | Совпал исход остановки | Макс. относительное изменение U/a в префиксах |",
              "| --- | ---: | ---: | ---: |"]
    for method in methods:
        rows = [r for r in checks if r["method"] == method]
        lines.append(f"| {method} | {sum(r['same_final_counts'] for r in rows)}/{len(rows)} | "
                     f"{sum(r['same_outcome'] for r in rows)}/{len(rows)} | "
                     f"{max(r['max_relative_prefix_certificate_change'] for r in rows):.3e} |")
    lines += ["", "В точной арифметике M=U0/D² обеспечивает эквивариантность при a*f+b: "
              "градиенты, gaps, сертификаты, оценки кривизны и допуск умножаются на a; направления, "
              "шаги и результаты квадратичного теста сохраняются при одинаковом правиле разрешения "
              "равенств. Фиксированное M после масштабирования соответствует M/a для исходной функции. "
              "В плавающей арифметике строгие сравнения и вычитания могут изменить ветвление; малая "
              "разница сертификатов здесь не устанавливает численную инвариантность вообще.", "",
              "Смена радиуса существенно меняет трудность. AFW устойчив на этих условиях к внешнему "
              "пределу обновлений, но при фиксированном бюджете значений функции часть пуассоновских "
              "стартов остаётся незавершённой. Выбор M=U0/D² не даёт равномерно меньших затрат, чем M=1, "
              "и не устраняет граничную медленную сходимость обычного FW. Результаты не доказывают "
              "общего превосходства над другими алгоритмами.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    project = next(p for p in root.resolve().parents if p.name.upper() == "PART3")
    paper = project/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\caption{Radius controls at objective scale $a=1$, five matched seeds. "
           r"Each entry gives final successes (out of five); median function calls. Capped failures are included. "
           r"The cap is 1000 updates, not 1000 function calls. "
           r"Subscript $1$: $M=1$; subscript $U$: $M=U_0/D^2$.}",
           r"\label{tab:robustness}", r"\begin{tabular}{lrrrr}", r"\hline",
           r"Radius & FW$_1$ & FW$_U$ & AFW$_1$ & AFW$_U$ \\", r"\hline"]
    for scenario in scenarios:
        tex.append(r"\multicolumn{5}{l}{\textbf{"+scenario.replace("_", r"\_")+r"}} \\")
        for factor in factors:
            radius = next(p["radius"] for p in manifest["problems"] if p["scenario"] == scenario and p["radius_factor"] == factor)
            cells = []
            for method in methods:
                rows = select(summary, scenario, factor, 1.0, method)
                cells.append(f"{sum(r['certified']=='True' for r in rows)}; {median(int(r['function_evals']) for r in rows):g}")
            tex.append(f"$R={radius:g}$ & "+" & ".join(cells)+r" \\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (paper/"robustness_table.tex").write_text("\n".join(tex), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(scenarios), len(factors), figsize=(6.5, 3.65), sharex=True, sharey=True, squeeze=False)
    labels = (r"FW, $M=1$", r"FW, $M=U_0/D^2$", r"AFW, $M=1$", r"AFW, $M=U_0/D^2$")
    for i, scenario in enumerate(scenarios):
        for j, factor in enumerate(factors):
            ax = axes[i, j]
            for method, label, color, marker in zip(methods, labels, ("#d55e00", "#e69f00", "#0072b2", "#009e73"), ("o", "s", "^", "D")):
                values = [[float(r["normalized_certificate"]) for r in select(fixed, scenario, factor, scale, method)] for scale in scales]
                middle, low, high = [np.array([np.quantile(v, q) for v in values]) for q in (0.5, 0.25, 0.75)]
                ax.plot(scales, middle, label=label, color=color, marker=marker, markersize=3, linewidth=1.1)
                ax.fill_between(scales, low, high, color=color, alpha=0.08)
            radius = next(p["radius"] for p in manifest["problems"] if p["scenario"] == scenario and p["radius_factor"] == factor)
            ax.set_title(("Logistic" if scenario == "breast_cancer" else "Poisson")+f", R={radius:g}")
            ax.axhline(manifest["base_tolerance"], color="0.4", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(scales)
            ax.grid(alpha=0.2)
            if i == len(scenarios)-1:
                ax.set_xlabel("Objective scale a")
            if j == 0:
                ax.set_ylabel("Certificate U/a\nat 1000 f calls")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.8, w_pad=0.8)
    for extension in ("png", "pdf"):
        fig.savefig(root/f"radius_and_units.{extension}", dpi=220)
    plt.close(fig)
    shutil.copy2(root/"radius_and_units.pdf", paper/"figures"/"radius_and_units.pdf")
    shutil.copy2(Path(__file__), root/"report_source.py")
    artifacts = ("manifest.json", "report_source.py", "results.md", "scale_checks.csv",
                 "radius_and_units.png", "radius_and_units.pdf")
    (root/"report_manifest.json").write_text(json.dumps({
        "artifact_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in artifacts},
        "paper_table_sha256": hashlib.sha256((paper/"robustness_table.tex").read_bytes()).hexdigest(),
        "scale_pairs": len(checks), "seed_count": nseeds,
    }, indent=2), encoding="utf-8")
    print(f"Report from {len(summary)} runs; {len(checks)} scale pairs; vector figure and table saved")


if __name__ == "__main__":
    main()
