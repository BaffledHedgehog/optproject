import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

cells = []

cells.append(md(r"""
# Сравнение обучения нейронной сети с квадратичной функцией потерь и перекрестной энтропией

**Авторы:** Ляхов Ярослав, Суворов Степан  
**Университет:** Университет Иннополис

## Цель работы

Рассмотреть обучение нейронной сети для бинарной классификации. Сравнить работу метода обратного распространения ошибки при двух функциях потерь:

- квадратичная функция потерь, то есть MSE;
- перекрестная энтропия, то есть Binary Cross-Entropy.

Сравнение проводится на одной архитектуре, одном наборе данных, одинаковой инициализации и одинаковом оптимизаторе. Так видно влияние функции потерь на градиенты, скорость обучения и качество классификации.
"""))

cells.append(md(r"""
## Теоретическая часть

Пусть нейронная сеть для бинарной классификации выдает логит $z$. Вероятность класса `1` получается через sigmoid:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

При обратном распространении ошибки градиент параметров получается по цепному правилу:

$$
\frac{\partial L}{\partial w} =
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial w}.
$$

Значит, функция потерь задает величину сигнала, который идет назад по сети.

### Перекрестная энтропия

Для бинарной классификации:

$$
L_{BCE}= -y\log p -(1-y)\log(1-p).
$$

Если совместить BCE с sigmoid, то градиент по логиту имеет простой вид:

$$
\frac{\partial L_{BCE}}{\partial z}=p-y.
$$

### Квадратичная функция потерь

Если использовать MSE для вероятности:

$$
L_{MSE}=(p-y)^2,
$$

то градиент по логиту:

$$
\frac{\partial L_{MSE}}{\partial z}=2(p-y)p(1-p).
$$

Главное отличие: у MSE появляется множитель $p(1-p)$. Когда sigmoid насыщается, $p$ близко к 0 или 1. Тогда этот множитель становится малым. Поэтому MSE может давать слабый градиент даже при сильной ошибке.
"""))

cells.append(code(r"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
"""))

cells.append(md(r"""
## 1. Сравнение градиентов по логиту

Сначала сравним не полное обучение, а только сигнал $\partial L / \partial z$, который loss-функция отправляет назад в сеть.

Рассмотрим объект истинного класса $y=1$. Если логит $z$ сильно отрицательный, то модель уверенно ошибается: вероятность $p$ близка к 0.
"""))

cells.append(code(r"""
z = torch.linspace(-10, 10, 1000)
p = torch.sigmoid(z)
y = torch.ones_like(p)

grad_bce = p - y
grad_mse = 2 * (p - y) * p * (1 - p)

plt.figure(figsize=(9, 5))
plt.plot(z, grad_bce, label="Перекрестная энтропия / BCE", linewidth=2)
plt.plot(z, grad_mse, label="Квадратичная потеря / MSE", linewidth=2)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("логит z")
plt.ylabel("dL/dz для y=1")
plt.title("Градиент функции потерь по логиту")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/gradient_bce_vs_mse.png", dpi=160)
plt.show()
"""))

cells.append(md(r"""
На графике видно, что при уверенной ошибке $z \ll 0$:

- у BCE градиент близок к `-1`, поэтому сеть получает сильный сигнал для исправления ошибки;
- у MSE градиент близок к `0`, потому что sigmoid насыщена и множитель $p(1-p)$ почти нулевой.

Это объясняет, почему для классификации перекрёстная энтропия обычно обучает сеть быстрее, чем MSE.
"""))

cells.append(md(r"""
## 2. Постановка эксперимента

Используем синтетическую задачу бинарной классификации `make_moons`. Она нелинейно разделима. Поэтому для нее нужна небольшая нейронная сеть, а не просто линейная модель.

Условия сравнения:

- одинаковая обучающая и тестовая выборки;
- одинаковая архитектура MLP;
- одинаковая начальная инициализация весов;
- одинаковый оптимизатор SGD с momentum;
- различается только функция потерь.
"""))

cells.append(code(r"""
X, y = make_moons(n_samples=1200, noise=0.25, random_state=SEED)
X = StandardScaler().fit_transform(X).astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device)
X_test_t = torch.tensor(X_test, device=device)
y_test_t = torch.tensor(y_test, device=device)

plt.figure(figsize=(7, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap="RdBu_r", s=24, alpha=0.75, edgecolor="white")
plt.title("Обучающая выборка make_moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/dataset_make_moons.png", dpi=160)
plt.show()
"""))

cells.append(code(r"""
class BinaryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def loss_function(name, logits, targets):
    if name == "BCE":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if name == "MSE":
        probs = torch.sigmoid(logits)
        return F.mse_loss(probs, targets)
    raise ValueError(name)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    pred = (probs >= 0.5).astype(int)
    true = y_test.ravel().astype(int)

    return {
        "accuracy": accuracy_score(true, pred),
        "f1": f1_score(true, pred),
        "bce_metric": log_loss(true, np.clip(probs, 1e-7, 1 - 1e-7)),
        "mse_metric": mean_squared_error(true, probs),
    }


def train(loss_name, epochs=350, lr=0.05):
    torch.manual_seed(SEED)
    model = BinaryMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = loss_function(loss_name, logits, y_train_t)
        loss.backward()

        first_layer_grad = model.net[0].weight.grad.detach().norm().item()
        all_grad = torch.sqrt(sum(
            (param.grad.detach() ** 2).sum()
            for param in model.parameters()
            if param.grad is not None
        )).item()

        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "grad_norm_first_layer": first_layer_grad,
                "grad_norm_all": all_grad,
                **evaluate(model),
            })

    return model, pd.DataFrame(history)


models = {}
histories = {}
results = {}

for loss_name in ["BCE", "MSE"]:
    model, hist = train(loss_name)
    models[loss_name] = model
    histories[loss_name] = hist
    results[loss_name] = evaluate(model)

pd.DataFrame(results).T
"""))

cells.append(md(r"""
## 3. Графики обучения

Ниже показаны:

- значение оптимизируемой функции потерь на обучении;
- accuracy на тестовой выборке;
- F1-score на тестовой выборке;
- норма градиента первого слоя.

Градиент первого слоя полезен как индикатор того, насколько сильный сигнал доходит в ранние параметры сети.
"""))

cells.append(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.ravel()

for loss_name, hist in histories.items():
    axes[0].plot(hist["epoch"], hist["train_loss"], label=loss_name, linewidth=2)
    axes[1].plot(hist["epoch"], hist["accuracy"], label=loss_name, linewidth=2)
    axes[2].plot(hist["epoch"], hist["f1"], label=loss_name, linewidth=2)
    axes[3].plot(hist["epoch"], hist["grad_norm_first_layer"], label=loss_name, linewidth=2)

axes[0].set_title("Оптимизируемый loss на train")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")

axes[1].set_title("Accuracy на test")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")

axes[2].set_title("F1-score на test")
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("F1")

axes[3].set_title("Норма градиента первого слоя")
axes[3].set_xlabel("epoch")
axes[3].set_ylabel("gradient norm")
axes[3].set_yscale("log")

for ax in axes:
    ax.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/training_bce_vs_mse.png", dpi=160)
plt.show()
"""))

cells.append(md(r"""
F1-score показывает баланс между точностью положительных предсказаний и полнотой нахождения положительного класса:

$$
F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}
$$

Здесь `precision` показывает, какая доля объектов, предсказанных как класс `1`, действительно относится к классу `1`. `recall` показывает, какую долю настоящих объектов класса `1` модель смогла найти.
"""))

cells.append(md(r"""
## 4. Границы классификации

Следующий график показывает, какие области пространства признаков каждая сеть относит к классу `1`.
"""))

cells.append(code(r"""
def plot_decision_boundaries(models):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.tensor(grid, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (loss_name, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(grid_t)).cpu().numpy().reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.75)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.4)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap="RdBu_r", edgecolor="white", s=26)
        ax.set_title(f"Граница решения: {loss_name}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decision_boundaries_bce_vs_mse.png", dpi=160)
    plt.show()


plot_decision_boundaries(models)
"""))

cells.append(md(r"""
## 5. Итоговая таблица

В таблице сравниваются финальные метрики на тестовой выборке.

`bce_metric` и `mse_metric` здесь считаются уже как метрики качества вероятностей, а не как обязательно оптимизируемая функция.
"""))

cells.append(code(r"""
summary = pd.DataFrame(results).T
summary.index.name = "training_loss"
summary
"""))

cells.append(md(r"""
## Вывод

В задаче бинарной классификации перекрестная энтропия хорошо согласована с вероятностным выходом sigmoid. Ее градиент по логиту равен $p-y$. Поэтому даже при уверенной ошибке сеть получает заметный сигнал для исправления весов.

Квадратичная функция потерь на вероятности имеет градиент $2(p-y)p(1-p)$. Из-за множителя $p(1-p)$ градиент становится малым, когда sigmoid насыщается. Поэтому MSE может медленнее исправлять уверенные ошибки и хуже подходит как основная функция потерь для классификации.

Эксперимент подтверждает теоретическое различие. При одинаковой архитектуре и данных BCE дает более простой ход обучения и обычно быстрее достигает хороших accuracy/F1. MSE остается полезной как метрика качества вероятностей. Но для обучения классификатора чаще выбирают перекрёстную энтропию.
"""))

cells.append(md(r"""
## Связь с исходной статьей

В статье `project_base.pdf` функции потерь рассматриваются как критерии, которые оптимизируются во время обучения. Метрики используются для оценки качества после обучения. В этой работе идея проверена на задаче классификации. Одна и та же сеть обучается с разными loss-функциями. Затем сравниваются итоговые метрики и поведение градиентов при backpropagation.
"""))

nb["cells"] = cells

with open("loss_backprop_comparison.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created loss_backprop_comparison.ipynb")
