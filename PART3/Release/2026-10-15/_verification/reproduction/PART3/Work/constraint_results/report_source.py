"""Report the transition sweep with equal weights on distinct starting vertices."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median

import numpy as np


def seed_atom(seed, dimension):
    rng = np.random.default_rng(seed)
    # problems.py assigns x0[random_index] = radius*random_sign.
    # Python evaluates the RHS (sign) before the subscript target (index).
    sign = 1 if rng.integers(2) else -1
    return int(rng.integers(dimension)), sign


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"constraint_results")
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    records = {n: read(n) for n in ("summary.csv", "trace.csv", "budget.csv", "accuracy.csv", "audit.csv")}
    classes, starts = {}, []
    dimension = manifest["problems"][0]["dimension"]
    for seed in manifest["seeds"]:
        atom = seed_atom(seed, dimension)
        representative = classes.setdefault(atom, seed)
        starts.append({"seed": seed, "coordinate": atom[0], "sign": atom[1], "representative_seed": representative})
    write_csv(root/"distinct_starts.csv", starts)
    representatives = set(classes.values())
    duplicates = []
    grouped = {}
    for name, rows in records.items():
        groups = {}
        for row in rows:
            k = (float(row["radius_ratio"]), row["method"], int(row["seed"]))
            groups.setdefault(k, []).append({k: v for k, v in row.items() if k not in ("seed", "elapsed_seconds")})
        grouped[name] = groups
    for start in starts:
        seed, rep = start["seed"], start["representative_seed"]
        if seed == rep:
            continue
        for ratio in manifest["radius_ratios"]:
            for method in manifest["methods"]:
                for name in records:
                    assert grouped[name][(ratio, method, seed)] == grouped[name][(ratio, method, rep)], (name, ratio, method, seed)
                duplicates.append({"radius_ratio": ratio, "method": method, "seed": seed,
                                   "representative_seed": rep, "saved_records_match": True})
    if duplicates:
        write_csv(root/"duplicate_checks.csv", duplicates)
    primary = {n: [r for r in rows if int(r["seed"]) in representatives] for n, rows in records.items()}
    summary, accuracy = primary["summary.csv"], primary["accuracy.csv"]
    budget1000 = [r for r in primary["budget.csv"] if r["oracle"] == "function_evals" and r["budget"] == "1000"]
    methods, ratios = manifest["methods"], manifest["radius_ratios"]
    def select(rows, ratio, method, epsilon=None):
        return [r for r in rows if float(r["radius_ratio"]) == ratio and r["method"] == method
                and (epsilon is None or float(r["epsilon"]) == epsilon)]
    def successes(rows):
        return sum(r["certified"] == "True" for r in rows)
    n = len(representatives)
    critical = manifest["critical_radius"]
    lines = ["# Пуассоновская регрессия: переход от активного ограничения к внутреннему минимуму", "",
             f"{len(records['summary.csv'])} сырых траекторий; {len(summary)} после исключения повторных "
             f"начальных вершин. {len(manifest['seeds'])} seed дают только {n} различных вершин. "
             f"Проверены {len(duplicates)} повторов: сохранённые состояния, счётчики, аудит и все бюджетные "
             "и точностные префиксы совпадают, время исключено из сравнения. Основные таблицы и график "
             "взвешивают каждую различную вершину одинаково. Это один набор данных и одна исходная "
             "функция; радиусы и старты не являются независимыми наборами данных.", "",
             f"Порог R*≈{critical['estimate']:.12f} — норма независимо найденного стационарного минимума. "
             f"Норма градиента {critical['residual_norm']:.3e}; оценка ошибки радиуса "
             f"√d·||∇f||/µ≈{critical['residual_radius_bound']:.3e}. Она следует из глобальной сильной "
             "выпуклости ridge-цели при точном остатке. Числа вычислены в float64, без интервальной "
             "арифметики; точное положение порога этими цифрами не сертифицируется. Условие R=R* "
             "помечено переходным. Все семь отношений R/R* заданы до запуска сравниваемых методов.", "",
             f"Методы начинают из исходной вершины, M=1 либо M=U0/D². Ни эталон, ни H0,H1,µ, "
             f"порог активности не передаются солверам. Одна траектория до U≤{manifest['tolerance']:g} "
             f"или {manifest['max_iterations']} обновлений даёт префиксы для {manifest['accuracies']}. "
             "Это вложенные префиксы, не дополнительные независимые прогоны. Все пробы f, градиенты "
             "и полные LMO оплачены; активные сканирования AFW — дополнительные затраты. "
             "Медианы конечных f включают ограниченные бюджетом неуспехи и не равны медиане времени до решения.", "",
             "## Независимые эталоны и KKT-диагностика", "",
             "| R/R* | R | Режим | Норма x_ref | Размер опоры | Оценка λ | KKT-остаток ∞ | Ширина интервала f* |",
             "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in manifest["problems"]:
        lines.append(f"| {p['radius_ratio']:g} | {p['radius']:.8f} | {p['regime']} | {p['reference_l1_norm']:.8f} | "
                     f"{p['reference_support_size']} | {p['estimated_multiplier']:.3e} | {p['kkt_residual_inf']:.3e} | "
                     f"{p['reference_upper']-p['reference_lower']:.3e} |")
    lines += ["", "Опора определяется численным порогом 1e-7, λ — из эталонного градиента. "
              "reference_face_slack в summary.csv — только послепусковая диагностика по этой опоре; "
              "она не является строгой сертификацией идентификации оптимальной грани.", ""]
    for ratio in ratios:
        lines += [f"## R/R*={ratio:g}, по {n} различным стартам", "",
                  "| Метод | Успехи 1e-3 | Успехи 1e-5 | Успехи 1e-7 | Медиана f при 1e-7 / пределе | Медиана U при ≤1000 f |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for method in methods:
            rows = [select(accuracy, ratio, method, e) for e in manifest["accuracies"]]
            fixed = select(budget1000, ratio, method)
            lines.append(f"| {method} | "+" | ".join(f"{successes(r)}/{n}" for r in rows)
                         +f" | {median(int(r['function_evals']) for r in rows[-1]):g} | "
                         f"{median(float(r['certificate']) for r in fixed):.3e} |")
        lines.append("")
    lines += ["## Связь ошибки с сертификатом", "",
              "На каждом полном журнале до прореживания проверено G≤max{2δ,D√(2(H0+H1δ)δ)}. "
              "Вместо неизвестной δ использована её верхняя оценка f−reference_lower. Это проверка "
              "с консервативным эталонным интервалом, не доказательство точности всех неизвестных δ. "
              "Новая связь не использует сильную выпуклость; она следует из обычного аргумента "
              "допустимого уменьшения функции. Геометрический error bound остаётся отдельным "
              "предположением логарифмической AFW-теоремы.", "",
              f"Замена прежнего порога ошибки даёт сокращение расчётного числа фаз до округления "
              f"от {min(float(r['phase_saving_before_rounding']) for r in accuracy):.2f} до "
              f"{max(float(r['phase_saving_before_rounding']) for r in accuracy):.2f} в этих условиях. "
              "Алгоритмы и их фактические траектории от этой замены доказательства не меняются. "
              "Геометрическая константа κ не подгоняется по опытам.", "",
              "## Выводы и ограничения", ""]
    for method in methods:
        rows = [r for r in accuracy if r["method"] == method and float(r["epsilon"]) == manifest["tolerance"]]
        lines.append(f"- {method}: {successes(rows)}/{len(rows)} при точности {manifest['tolerance']:g} к пределу обновлений.")
    lines += ["", "Активный режим проверен, но близость радиуса к порогу сама по себе не предсказывает "
              "всю стоимость: сужение шара меняет начальную вершину, опору решения и обусловленность. "
              "Нормировка M не обеспечивает равномерного выигрыша. Измерения относятся к одному набору "
              f"данных и {n} выбранным вершинам, а не ко всем возможным стартам или wall-clock рейтингу. "
              "Теоретическая новизна и независимая проверка доказательства остаются отдельными вопросами.", ""]
    (root/"results.md").write_text("\n".join(lines), encoding="utf-8")
    project = next(p for p in root.resolve().parents if p.name.upper() == "PART3")
    paper = project/"Paper"
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           rf"\caption{{Poisson transition sweep, {n} distinct starting vertices per radius. "
           rf"Entries are successes at $10^{{-7}}$ (out of {n}); median final function calls, "
           r"including failures capped at 2000 updates. Subscripts give $M=1$ or $M=U_0/D^2$.}",
           r"\label{tab:transition}", r"\begin{tabular}{lrrrr}", r"\hline",
           r"$R/R_*$ & FW$_1$ & FW$_U$ & AFW$_1$ & AFW$_U$ \\", r"\hline"]
    for ratio in ratios:
        cells = []
        for method in methods:
            rows = select(accuracy, ratio, method, manifest["tolerance"])
            cells.append(f"{successes(rows)}; {median(int(r['function_evals']) for r in rows):g}")
        tex.append(f"{ratio:g} & "+" & ".join(cells)+r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (paper/"constraint_table.tex").write_text("\n".join(tex), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, (ax, heat) = plt.subplots(2, 1, figsize=(6.5, 4.1), gridspec_kw={"height_ratios": [1.5, 1]}, sharex=True)
    labels = (r"FW, $M=1$", r"FW, $M=U_0/D^2$", r"AFW, $M=1$", r"AFW, $M=U_0/D^2$")
    for method, label, color, marker in zip(methods, labels, ("#d55e00", "#e69f00", "#0072b2", "#009e73"), ("o", "s", "^", "D")):
        v = [[float(r["certificate"]) for r in select(budget1000, ratio, method)] for ratio in ratios]
        mid, low, high = [np.array([np.quantile(a, q) for a in v]) for q in (0.5, 0.25, 0.75)]
        ax.plot(range(len(ratios)), mid, label=label, color=color, marker=marker, markersize=3, linewidth=1.1)
        ax.fill_between(range(len(ratios)), low, high, color=color, alpha=0.1)
    ax.set_yscale("log")
    ax.axhline(manifest["tolerance"], color="0.5", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Certificate U\nat 1000 f calls")
    ax.grid(alpha=0.2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, loc="upper center", frameon=False)
    values = np.array([[successes(select(accuracy, ratio, method, manifest["tolerance"])) for ratio in ratios] for method in methods])
    heat.imshow(values, vmin=0, vmax=n, cmap="Blues", aspect="auto")
    for i in range(len(methods)):
        for j in range(len(ratios)):
            heat.text(j, i, f"{values[i,j]}/{n}", ha="center", va="center", fontsize=7,
                      color="white" if values[i,j] >= n*0.6 else "black")
    heat.set_yticks(range(len(methods)), labels)
    heat.set_xticks(range(len(ratios)), [f"{ratio:g}" for ratio in ratios])
    heat.set_xlabel(r"Radius ratio $R/R_*$ (categorical positions)")
    heat.set_title(r"Successes at $U\leq10^{-7}$ within 2000 updates", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8)
    for extension in ("png", "pdf"):
        fig.savefig(root/f"constraint_transition.{extension}", dpi=220)
    plt.close(fig)
    shutil.copy2(root/"constraint_transition.pdf", paper/"figures"/"constraint_transition.pdf")
    shutil.copy2(Path(__file__), root/"report_source.py")
    artifacts = ["manifest.json", "report_source.py", "results.md", "distinct_starts.csv",
                 "constraint_transition.png", "constraint_transition.pdf"]
    if duplicates:
        artifacts.append("duplicate_checks.csv")
    (root/"report_manifest.json").write_text(json.dumps({
        "artifact_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in artifacts},
        "raw_trajectories": len(records["summary.csv"]), "distinct_trajectories": len(summary),
        "distinct_starts": n, "duplicate_trajectories_checked": len(duplicates),
        "paper_table_sha256": hashlib.sha256((paper/"constraint_table.tex").read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"{len(summary)} distinct trajectories; {len(duplicates)} duplicates match; table and figure saved")


if __name__ == "__main__":
    main()
