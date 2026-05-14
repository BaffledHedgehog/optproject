# Loss Backpropagation Comparison

Главный файл:

- `loss_backprop_comparison.ipynb` - выполненный Jupyter Notebook с теорией, кодом, таблицами, графиками и выводами.
- `report.docx` - реферат по работе в формате Word.
- `report.md` - Markdown-версия реферата.

Дополнительно:

- `figures/` - PNG-графики из ноутбука;
- `make_project_notebook.py` - генератор ноутбука, полезен если нужно пересобрать `.ipynb`;
- `make_report_docx.py` - генератор Word-реферата;
- `project_base.pdf` - исходная статья-обзор по loss functions and metrics.

Запуск:

```bash
pip install -r requirements.txt
jupyter notebook loss_backprop_comparison.ipynb
```

Тема проекта:

- задача обучения нейронной сети для бинарной классификации;
- сравнение квадратичной функции потерь MSE и перекрёстной энтропии BCE;
- анализ обратного распространения ошибки, градиентов, метрик и границ классификации.

Авторы: Ляхов Ярослав, Суворов Степан.  
Университет: Университет Иннополис.
