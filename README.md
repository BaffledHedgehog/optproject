# Loss Backpropagation Comparison

Проект сравнивает работу обратного распространения ошибки для разных функций потерь.

Главный файл:

- `loss_backprop_comparison.ipynb` - выполненный Jupyter Notebook с теорией, кодом, таблицами, графиками и выводами.

Дополнительно:

- `figures/` - PNG-графики из ноутбука;
- `make_project_notebook.py` - генератор ноутбука, полезен если нужно пересобрать `.ipynb`;
- `project_base.pdf` - исходная статья-обзор по loss functions and metrics.

Запуск:

```bash
pip install -r requirements.txt
jupyter notebook loss_backprop_comparison.ipynb
```

В ноутбуке сравниваются:

- классификация: Cross-Entropy/BCE, MSE(sigmoid), Hinge, Focal;
- регрессия: MSE, MAE, Huber, Log-Cosh.
