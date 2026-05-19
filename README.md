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
