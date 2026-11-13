# Проект по оптимизации

Текущая статья — **Frank-Wolfe under gap-dependent smoothness**, редакция v20
от 14 ноября 2026 года. Авторы: Yaroslav P. Lyakhov и Stepan A. Suvorov.

- [Статья на русском](PART3/Submission/2026-11-14-v20/paper_ru_v20.pdf).
- [English paper](PART3/Submission/2026-11-14-v20/paper_en_v20.pdf).
- [Краткое объяснение](PART3/Submission/2026-11-14-v20/guide_professor_v20.txt).
- [Исходники, данные и воспроизведение](PART3/Submission/2026-11-14-v20/README_RU.txt).
- [Изменения после проверки](PART3/Submission/2026-11-14-v20/REVIEW_RESPONSE_v20.txt).

В статье исследуется условие кривизны через полный зазор Франка–Вульфа:
явный логарифмический шаг, сертифицированная остановка и многомерная
пуассоновская регрессия. Сравнение восьми правил включает 432 запуска
на 54 стартовых конфигурациях. Практическое превосходство над адаптивными
методами не заявляется.

Проверить комплект и собрать архив из сохранённых материалов:

```powershell
cd PART3/Submission/2026-11-14-v20
python -m pip install -r requirements.txt
python verify_release.py
python package_release.py
```

Полный архив создаётся локально; в Git хранятся исходники, результаты и одна
копия каждого итогового PDF. [Структура PART3 и связанные исследования](PART3/README.md).

## Другие проекты

Эти материалы относятся к самостоятельным учебным работам.

| Материал | Назначение |
| --- | --- |
| [Ноутбук по функциям потерь](prototypeproject.ipynb.ipynb) | MSE и BCE при обучении через sigmoid, сравнение по 20 seed |
| [Отчёт](report.md) и [Word-версия](report.docx) | Результаты первой лабораторной работы |
| [Вторая часть](part2/report_part2.md) | Эксперименты с ослабленной гладкостью |
| [Теория второй части](part2/theory.md) | Постановки и предположения |
| [Обзор переводов](part2/articles_ru_index.md) | Источники по теме второй части |

Для первой работы: `pip install -r requirements.txt`, затем
`jupyter notebook prototypeproject.ipynb.ipynb`.
