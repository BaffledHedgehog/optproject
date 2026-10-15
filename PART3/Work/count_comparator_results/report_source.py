"""Report the additional AC-FW runs with charged equal function budgets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).parent/"count_comparator_results")
    root = parser.parse_args().directory
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (root/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    rows, budgets = read("summary.csv"), read("budget.csv")
    text = ["# AC-FW на дополнительных счётных данных", "",
            "108 дополнительных запусков метода AC-FW на тех же 108 сочетаниях набора, радиуса и начальной "
            "вершины, что в `multidata_results`. Два радиуса и все 30/24 различные вершины сохранены. "
            "Прежние 648 записей не перезаписаны; вместе это семь методов на прежних условиях, а не новые независимые данные.", "",
            "Использованы алгоритмы 1–2 [Giang-Tran et al., v2](https://arxiv.org/html/2605.15512v2), "
            "формула (6), δ=1. Это независимая реализация, а не код авторов. Обязательный начальный "
            "переход в LMO-вершину и две исходные оценки f/g/LMO оплачиваются. Отказ потребляет "
            "одну внешнюю пробу и f, но сохраняет градиент. При N пробах Nf=NLMO=N+2, Ng=2+Naccepted.", "",
            "Допуск 1e−5; предел 1000 внешних проб, включая отказы. Основной срез — 1000 вызовов f, "
            "достижимый всеми незавершёнными запусками до предела. Бюджеты g и LMO записаны отдельно "
            "с признаком доступности; конечные медианы включают незавершённые случаи. Сравнения полного времени нет.", "",
            "| Набор | R/R* | Успехи к концу | Медиана f к концу | Успехи при 1000 f | Медиана U при 1000 f | Отказы, сумма |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in manifest["datasets"]:
        for ratio in manifest["radius_ratios"]:
            rs = [r for r in rows if r["dataset"] == dataset and float(r["radius_ratio"]) == ratio]
            bs = [r for r in budgets if r["dataset"] == dataset and float(r["radius_ratio"]) == ratio
                  and r["oracle"] == "function_evals" and r["budget"] == "1000"]
            assert all(r["budget_reached_or_certified"] == "True" for r in bs)
            text.append(f"| {dataset} | {ratio:g} | {sum(r['certified']=='True' for r in rs)}/{len(rs)} | "
                        f"{median(int(r['function_evals']) for r in rs):g} | {sum(r['certified']=='True' for r in bs)}/{len(bs)} | "
                        f"{median(float(r['certificate']) for r in bs):.6g} | {sum(int(r['rejected_steps']) for r in rs)} |")
    text += ["", "На активном Metro AC-FW сертифицирует 13/24 старта при 1000 f, FW-BT — 2/24 и 3/24, "
             "AFW-BT — по 24/24. На активном Seoul AC-FW и FW-BT не завершают ни одного старта, AFW-BT — все 30/30. "
             "Во внутренних режимах при этом бюджете ни один из семи методов со scalar-сертификатом не проходит цель. "
             "Это конечное сравнение на зависимых условиях, не общий рейтинг методов.", "",
             "Условие глобально липшицева градиента на Rᵈ из опубликованной AC-FW-теории не выполняется "
             "для экспоненциальной цели. Здесь показано эмпирическое применение на компактном Q; теорема "
             "авторов не переносится дословно. Эталоны, H₀,H₁, гессиан и глобальная L в шаг AC-FW не поступают. "
             "Отдельный тест на обоих наборах скрывает эти параметры и инструментирует реальные вызовы оракулов.", "",
             "Полный аудит до прореживания проверяет допустимость, нижние оценки, монотонность после "
             "обязательной инициализации, строгую приёмку, отказы и счётчики. `validate_results.py` связывает "
             "задачи с исходным архивом по SHA-256 и проверяет бюджетные записи.", ""]
    (root/"results.md").write_text("\n".join(text), encoding="utf-8")
    if Path(__file__).resolve() != (root/"report_source.py").resolve():
        shutil.copy2(__file__, root/"report_source.py")
    names = ("manifest.json", "results.md", "report_source.py")
    (root/"report_manifest.json").write_text(json.dumps({
        "artifact_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}}, indent=2), encoding="utf-8")
    print("Additional AC-FW comparator report saved")


if __name__ == "__main__":
    main()
