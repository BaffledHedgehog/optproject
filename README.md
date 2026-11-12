# Проект по оптимизации

Текущий исследовательский проект находится в [PART3](PART3/README.md).
Основная статья: **Sharp Error Bounds and Adaptive Stopping for Frank-Wolfe
beyond Convexity**, Yaroslav P. Lyakhov и Stepan A. Suvorov.

Редакция v17 от 12 ноября 2026:

- [Русский PDF](PART3/Submission/2026-11-12-v17/paper_ru.pdf), 23 страницы.
- [English PDF](PART3/Submission/2026-11-12-v17/paper.pdf), 23 pages.
- [Краткий разбор](PART3/Submission/2026-11-12-v17/guide_professor.txt).
- [Исходники, код и воспроизведение](PART3/Submission/2026-11-12-v17/README_RU.txt).

Повторные доказательства объединены. Новый результат учитывает диаметр
области и усиливает совместный сертификат по зазору и норме градиента.
Полные центральные доказательства входят в текущую статью. Расширенные
материалы v16 сохранены в отдельном архиве
extended_sources_and_records_v16.zip, который предоставляется отдельно
от этого репозитория.

Ниже сохранено описание более ранней лабораторной работы.

# Loss Backpropagation Comparison

Главный файл:

- `prototypeproject.ipynb.ipynb` - основной Jupyter Notebook с теорией, кодом, графиками, таблицами и выводами.

Дополнительно:

- `figures/` - PNG-графики из ноутбука;
- `report.docx` - Word-отчет, если нужен отдельный текстовый документ;
- `report.md` - Markdown-версия отчета;
- `project_base.pdf` - исходная обзорная статья по loss functions and metrics;
- `make_project_notebook.py` и `make_report_docx.py` - служебные генераторы старой версии. Для проверки основного ноутбука они не нужны.

Запуск:

```bash
pip install -r requirements.txt
jupyter notebook prototypeproject.ipynb.ipynb
```

Тема проекта:

- задача обучения нейронной сети для бинарной классификации;
- сравнение MSE и Binary Cross-Entropy при backpropagation через sigmoid;
- анализ градиентов, обучения, насыщения sigmoid и границ классификации;
- проверка результатов на 20 seed;
- отдельная проверка честности сравнения через подбор learning rate на validation split.

Ключевой вывод:

BCE дает более прямой градиент по логиту. При одинаковом learning rate она быстрее выходит на хорошее качество. После подбора learning rate MSE может приблизиться по accuracy/F1. В этом запуске она даже дает лучшие вероятностные метрики. итог простой: BCE удобнее для оптимизации, но MSE тоже может обучить классификатор.

Авторы: Ляхов Ярослав, Суворов Степан.

Университет: Университет Иннополис.
