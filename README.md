# Проект по оптимизации

Текущий исследовательский проект находится в [PART3](PART3/README.md).
Актуальная статья — **Observable Certificates and Exact Termination in
Backtracking Frank–Wolfe**, Yaroslav P. Lyakhov и Stepan A. Suvorov.

- [PDF для экспертного чтения](PART3/Submission/2026-11-01/paper.pdf).
- [Исходники для arXiv](PART3/Submission/2026-11-01/arxiv_source_v2.zip).
- [Инструкция и воспроизведение](PART3/Submission/2026-11-01/README_RU.txt).

Большие генерируемые архивы, распакованные проверки, промежуточные сборки
и служебные отчёты ассистента исключены из Git. Исходники, проверенные
экспериментальные данные и снимки кода сохранены. Полный архив кода и данных
можно создать из каталога `PART3/Submission/2026-11-01`:

```powershell
python prepare_submission.py --build
python prepare_submission.py --package --verify --tag local
```

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
