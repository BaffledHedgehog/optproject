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
# Часть 2. `(L0, L1)`-гладкость, clipping и адаптивные методы в задаче обучения

Этот notebook является самостоятельной второй частью проекта. В нем сначала выписана теория, затем реализованы методы оптимизации и тесты, а после этого проведен эксперимент на задаче бинарной классификации `make_moons`.

Рассматриваем обучение нейронной сети как задачу

$$
\min_{\theta \in \mathbb{R}^d} F(\theta),
\qquad
F(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i),y_i).
$$

В первой части проекта сравнивались BCE и MSE. Здесь фиксируем BCE и сравниваем уже методы оптимизации:

- обычный full-batch GD;
- clipped GD;
- normalized GD;
- NSGD-M;
- GD с backtracking line search;
- шаг Поляка.

Дополнительно считаем локальную оценку гладкости

$$
\hat L_k =
\frac{\|g_k-g_{k-1}\|}
{\|x_k-x_{k-1}\|+\epsilon_{\mathrm{num}}},
$$

и проверяем, коррелирует ли она с $\|g_k\|$, как предполагает модель `(L0, L1)`-гладкости.
"""))

cells.append(md(r"""
## 1. Постановка задачи обучения

Обучение нейронной сети можно рассматривать как минимизацию empirical risk по параметрам модели:

$$
\min_{\theta \in \mathbb{R}^d} F(\theta),
\qquad
F(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i)+r(\theta).
$$

Здесь $\theta$ - вектор всех весов и bias сети, $f_\theta(x)$ - предсказание модели, $\ell$ - функция потерь, $r(\theta)$ - возможная регуляризация.

Для бинарной классификации сеть выдает логит

$$
z_\theta(x)\in\mathbb{R},
\qquad
p_\theta(x)=\sigma(z_\theta(x))=\frac{1}{1+\exp(-z_\theta(x))}.
$$

Для BCE:

$$
\ell_{\mathrm{BCE}}(p,y)=-y\log p-(1-y)\log(1-p),
$$

и после совмещения sigmoid с BCE:

$$
\frac{\partial \ell_{\mathrm{BCE}}}{\partial z}=p-y.
$$

Для MSE по вероятности:

$$
\ell_{\mathrm{MSE}}(p,y)=(p-y)^2,
$$

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial z}=2(p-y)p(1-p).
$$

Множитель $p(1-p)$ объясняет, почему MSE может давать слабый градиент при насыщенной sigmoid. В этой части проекта фиксируем BCE как основную loss-функцию и исследуем уже не выбор loss, а выбор метода оптимизации.
"""))

cells.append(md(r"""
## 2. Классическая гладкость и `(L0, L1)`-гладкость

Классическая `L`-гладкость требует

$$
\|\nabla^2 F(x)\| \le L.
$$

Эквивалентно, градиент является липшицевым:

$$
\|\nabla F(x)-\nabla F(y)\|\le L\|x-y\|.
$$

Отсюда следует стандартная лемма о спуске:

$$
F(y)\le F(x)+\langle \nabla F(x),y-x\rangle+\frac{L}{2}\|y-x\|^2.
$$

Для градиентного спуска

$$
x_{k+1}=x_k-\eta \nabla F(x_k)
$$

при $\eta\le 1/L$ получаем

$$
F(x_{k+1})\le F(x_k)-\frac{\eta}{2}\|\nabla F(x_k)\|^2.
$$

Проблема в том, что для нейронных сетей глобальная константа $L$ часто слишком большая. Локальная гладкость может резко меняться вдоль траектории обучения.

В статьях Zhang et al., Huebler et al. и Vankov et al. используется более гибкое условие:

$$
\|\nabla^2 F(x)\| \le L_0 + L_1\|\nabla F(x)\|.
$$

Это называется `(L0, L1)`-гладкостью. При $L_1=0$ получаем обычную $L_0$-гладкость. Если градиент большой, локальная гладкость тоже может быть большой. Поэтому фиксированный шаг GD может быть нестабильным, а clipping/normalization уменьшают эффективный шаг автоматически.

Для недважды дифференцируемой функции можно использовать first-order форму:

$$
\limsup_{\delta\to 0}
\frac{\|\nabla F(x+\delta)-\nabla F(x)\|}{\|\delta\|}
\le L_0+L_1\|\nabla F(x)\|.
$$
"""))

cells.append(md(r"""
## 3. Цель сходимости

Полная нейронная сеть задает невыпуклую задачу, поэтому обычно не доказывают достижение глобального минимума. Цель - найти $\varepsilon$-стационарную точку:

$$
\|\nabla F(x)\|\le \varepsilon.
$$

В стохастическом случае используют критерий

$$
\mathbb{E}\|\nabla F(x)\|\le \varepsilon
$$

или

$$
\mathbb{E}\|\nabla F(x)\|^2\le \varepsilon^2.
$$

Обозначим начальный разрыв:

$$
\Delta_0=F(x_0)-F^*,
$$

где $F^*$ - нижняя грань функции.
"""))

cells.append(md(r"""
## 4. Clipped GD и Normalized GD

Для clipped GD:

$$
x_{k+1}=x_k-\eta \operatorname{clip}_c(g_k),
\qquad
\operatorname{clip}_c(g)=\min\left\{1,\frac{c}{\|g\|}\right\}g.
$$

Если $\|g_k\|\le c$, то метод делает обычный шаг $-\eta g_k$. Если $\|g_k\|>c$, то длина градиента ограничивается:

$$
\|\operatorname{clip}_c(g_k)\|=c.
$$

Эквивалентная запись:

$$
x_{k+1}=x_k-h_k g_k,
\qquad
h_k=\min\left\{\eta,\frac{\eta c}{\|g_k\|}\right\}.
$$

Normalized GD:

$$
x_{k+1}=x_k-\alpha\frac{g_k}{\|g_k\|+\beta}.
$$

Если $\beta$ малое, длина update почти равна $\alpha$. Поэтому метод защищен от слишком больших градиентов, но при очень малом градиенте может не уменьшать шаг достаточно быстро.
"""))

cells.append(md(r"""
## 5. NSGD-M, Backtracking и шаг Поляка

NSGD-M использует momentum и нормализацию:

$$
m_k=\beta_km_{k-1}+(1-\beta_k)g_k,
\qquad
x_{k+1}=x_k-\eta_k\frac{m_k}{\|m_k\|}.
$$

Parameter-agnostic расписание из Huebler et al.:

$$
\beta_k=1-k^{-1/2},
\qquad
\eta_k=\frac{k^{-3/4}}{7}.
$$

Теоретически:

$$
\frac{1}{T}\sum_{k=1}^{T}\mathbb{E}\|\nabla F(x_k)\|
\le
\tilde O(T^{-1/4}),
$$

то есть для $\mathbb{E}\|\nabla F(x)\|\le \varepsilon$ требуется

$$
T=\tilde O(\varepsilon^{-4}).
$$

Метод не требует знания $L_0,L_1$, но в оценке появляется плохая зависимость от $L_1$.

Для full-batch backtracking используем условие Armijo:

$$
F(x_k-\eta g_k)\le F(x_k)-\gamma\eta\|g_k\|^2.
$$

Backtracking не требует заранее знать $L_0,L_1$. Для deterministic `(L0,L1)`-гладкой задачи справедлива оценка вида:

$$
\frac{1}{T}\sum_{k=1}^{T}\|\nabla F(x_k)\|^2
\le
O\left(\frac{L_0\Delta_0+L_1^2\Delta_0^2}{T}\right).
$$

Шаг Поляка:

$$
\eta_k=\frac{F(x_k)-F^*}{\|\nabla F(x_k)\|^2}.
$$

Он не требует $L_0,L_1$, но требует знать или оценивать $F^*$. В обучении сети иногда берут $F^*\approx 0$, но для безопасности ограничивают шаг сверху:

$$
\eta_k\leftarrow \min\{\eta_k,\eta_{\max}\}.
$$
"""))

cells.append(md(r"""
## 6. Улучшенные оценки Vankov et al.

Для `(L0,L1)`-гладкой функции используется верхняя оценка с функцией

$$
\varphi(t)=e^t-t-1.
$$

Тогда

$$
F(y)
\le
F(x)+\langle \nabla F(x),y-x\rangle
+
\frac{L_0+L_1\|\nabla F(x)\|}{L_1^2}
\varphi(L_1\|y-x\|).
$$

Минимизация этой верхней оценки по направлению антиградиента дает шаг

$$
\eta_k^*
=
\frac{1}{L_1\|\nabla F(x_k)\|}
\ln\left(
1+
\frac{L_1\|\nabla F(x_k)\|}
{L_0+L_1\|\nabla F(x_k)\|}
\right).
$$

Упрощенный вариант:

$$
\eta_k^{\mathrm{si}}
=
\frac{1}{L_0+\frac{3}{2}L_1\|\nabla F(x_k)\|}.
$$

Clipping-аппроксимация:

$$
\eta_k^{\mathrm{cl}}
=
\min\left\{
\frac{1}{2L_0},
\frac{1}{3L_1\|\nabla F(x_k)\|}
\right\}.
$$

Ключевой прогресс:

$$
F(x_k)-F(x_{k+1})
\ge
\frac{\|\nabla F(x_k)\|^2}
{2L_0+3L_1\|\nabla F(x_k)\|}.
$$

Для невыпуклой задачи:

$$
\min_{0\le k\le K}\|\nabla F(x_k)\|\le \varepsilon
$$

если

$$
K+1
\ge
\frac{2L_0F_0}{a\varepsilon^2}
+
\frac{3L_1F_0}{a\varepsilon}.
$$

Это удобно интерпретировать как улучшенную оценку:

$$
T
=
O\left(
\frac{L_0F_0}{\varepsilon^2}
+\frac{L_1F_0}{\varepsilon}
\right).
$$
"""))

cells.append(md(r"""
## 7. Стохастический градиент и неточность информации

В mini-batch обучении вместо полного градиента доступен стохастический:

$$
g_k=\nabla f(x_k,\xi_k).
$$

Обычно предполагают:

$$
\mathbb{E}[g_k\mid x_k]=\nabla F(x_k),
\qquad
\mathbb{E}\|g_k-\nabla F(x_k)\|^2\le \sigma^2.
$$

Если batch size равен $B$, то часто

$$
\sigma^2(B)\approx \frac{\sigma_1^2}{B}.
$$

При biased или неточной информации можно записать:

$$
\tilde g_k=\nabla F(x_k)+\xi_k+b_k,
$$

где

$$
\mathbb{E}[\xi_k\mid x_k]=0,
\qquad
\mathbb{E}\|\xi_k\|^2\le \sigma_k^2,
\qquad
\|b_k\|\le \delta_k.
$$

Тогда нельзя ожидать точность лучше шумового пола:

$$
\|\nabla F(x)\|\lesssim \varepsilon+C_1\sigma+C_2\delta.
$$

Практический адаптивный подход: увеличивать batch size, когда градиент становится маленьким:

$$
B_k
\ge
\frac{\hat\sigma_k^2}
{\alpha^2\|g_k\|^2+\varepsilon_{\mathrm{floor}}^2}.
$$
"""))

cells.append(code(r"""
import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = Path("figures_part2")
FIG_DIR.mkdir(exist_ok=True)

device
"""))

cells.append(md(r"""
## 8. Наглядные графики к теории

Ниже четыре иллюстрации:

1. BCE дает градиент по логиту без множителя насыщения sigmoid, а MSE содержит $p(1-p)$.
2. В модели `(L0,L1)` локальная гладкость растет вместе с нормой градиента.
3. Обычный GD имеет update norm, растущий линейно с $\|g\|$, а ClipGD/NGD ограничивают длину update.
4. В оценке Vankov et al. прогресс ведет себя как

$$
\frac{\|g\|^2}{2L_0+3L_1\|g\|}.
$$
"""))

cells.append(code(r"""
z_grid = np.linspace(-8, 8, 400)
p_grid = 1 / (1 + np.exp(-z_grid))
y_one = 1.0
grad_bce = p_grid - y_one
grad_mse = 2 * (p_grid - y_one) * p_grid * (1 - p_grid)

grad_norm_grid = np.linspace(0, 8, 400)
L0_demo, L1_demo = 1.0, 0.8
local_L = L0_demo + L1_demo * grad_norm_grid

lr_demo = 0.2
clip_radius_demo = 2.0
alpha_demo = 0.6
gd_update = lr_demo * grad_norm_grid
clip_update = lr_demo * np.minimum(grad_norm_grid, clip_radius_demo)
ngd_update = alpha_demo * grad_norm_grid / (grad_norm_grid + 1e-8)

progress = grad_norm_grid ** 2 / (2 * L0_demo + 3 * L1_demo * grad_norm_grid + 1e-12)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(z_grid, grad_bce, label="BCE: p - y", linewidth=2)
axes[0, 0].plot(z_grid, grad_mse, label="MSE: 2(p-y)p(1-p)", linewidth=2)
axes[0, 0].axhline(0, color="black", linewidth=0.8)
axes[0, 0].set_title("Gradient by logit for y=1")
axes[0, 0].set_xlabel("logit z")
axes[0, 0].set_ylabel("dL/dz")

axes[0, 1].plot(grad_norm_grid, local_L, color="tab:green", linewidth=2)
axes[0, 1].fill_between(grad_norm_grid, L0_demo, local_L, color="tab:green", alpha=0.15)
axes[0, 1].set_title(r"$(L_0,L_1)$ smoothness model")
axes[0, 1].set_xlabel(r"$\|\nabla F(x)\|$")
axes[0, 1].set_ylabel(r"$L_0 + L_1\|\nabla F(x)\|$")

axes[1, 0].plot(grad_norm_grid, gd_update, label="GD", linewidth=2)
axes[1, 0].plot(grad_norm_grid, clip_update, label="ClipGD", linewidth=2)
axes[1, 0].plot(grad_norm_grid, ngd_update, label="NGD", linewidth=2)
axes[1, 0].set_title("Effective update norm")
axes[1, 0].set_xlabel(r"$\|g\|$")
axes[1, 0].set_ylabel(r"$\|x_{k+1}-x_k\|$")

axes[1, 1].plot(grad_norm_grid, progress, color="tab:purple", linewidth=2)
axes[1, 1].set_title("Lower bound on one-step progress")
axes[1, 1].set_xlabel(r"$\|g\|$")
axes[1, 1].set_ylabel(r"$\|g\|^2/(2L_0+3L_1\|g\|)$")

for ax in axes.ravel():
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8) if ax.get_legend_handles_labels()[0] else None

plt.tight_layout()
plt.savefig(FIG_DIR / "part2_theory_visuals.png", dpi=160)
plt.show()
"""))

cells.append(md(r"""
## Тест 1. Формулы градиента BCE и MSE по логиту

Проверяем, что autograd дает те же выражения:

$$
\frac{\partial L_{\mathrm{BCE}}}{\partial z}=p-y,
\qquad
\frac{\partial L_{\mathrm{MSE}}}{\partial z}=2(p-y)p(1-p).
$$
"""))

cells.append(code(r"""
z = torch.linspace(-6, 6, 25, requires_grad=True)
y = torch.ones_like(z)
p = torch.sigmoid(z)

bce = F.binary_cross_entropy_with_logits(z, y, reduction="sum")
bce_grad = torch.autograd.grad(bce, z, retain_graph=True)[0]

mse = ((p - y) ** 2).sum()
mse_grad = torch.autograd.grad(mse, z)[0]

expected_bce = p.detach() - y
expected_mse = 2 * (p.detach() - y) * p.detach() * (1 - p.detach())

assert torch.allclose(bce_grad, expected_bce, atol=1e-6)
assert torch.allclose(mse_grad, expected_mse, atol=1e-6)

print("OK: gradients for BCE and MSE match the formulas.")
"""))

cells.append(md(r"""
## Данные и модель
"""))

cells.append(code(r"""
X, y = make_moons(n_samples=1200, noise=0.25, random_state=SEED)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train.reshape(-1, 1), device=device)
X_test_t = torch.tensor(X_test, device=device)
y_test_t = torch.tensor(y_test.reshape(-1, 1), device=device)

plt.figure(figsize=(5, 4))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=18, alpha=0.8)
plt.title("make_moons train set")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig(FIG_DIR / "dataset_make_moons_part2.png", dpi=160)
plt.show()
"""))

cells.append(code(r"""
class BinaryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_model(seed=SEED):
    torch.manual_seed(seed)
    model = BinaryMLP().to(device)
    return model


base_model = make_model(SEED)
initial_state = copy.deepcopy(base_model.state_dict())


def reset_model():
    model = make_model(SEED)
    model.load_state_dict(copy.deepcopy(initial_state))
    return model


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grad(model):
    chunks = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def apply_flat_update(model, update):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            size = p.numel()
            p.add_(update[offset:offset + size].view_as(p))
            offset += size


def set_flat_params(model, values):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            size = p.numel()
            p.copy_(values[offset:offset + size].view_as(p))
            offset += size


def train_loss(model):
    logits = model(X_train_t)
    return F.binary_cross_entropy_with_logits(logits, y_train_t)


def loss_and_grad(model):
    model.zero_grad(set_to_none=True)
    loss = train_loss(model)
    loss.backward()
    grad = flat_grad(model)
    return float(loss.detach().cpu()), grad


def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(np.float32)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "log_loss": log_loss(y_test, np.clip(probs, 1e-7, 1 - 1e-7)),
    }
"""))

cells.append(md(r"""
## Тест 2. Операторы clipping и normalization
"""))

cells.append(code(r"""
def clip_by_norm(g, c, eps=1e-12):
    norm = torch.linalg.norm(g)
    scale = torch.minimum(torch.tensor(1.0, device=g.device), c / (norm + eps))
    return scale * g


g = torch.tensor([3.0, 4.0])
clipped = clip_by_norm(g, torch.tensor(2.0))
assert torch.allclose(torch.linalg.norm(clipped), torch.tensor(2.0), atol=1e-6)

small = torch.tensor([0.3, 0.4])
not_clipped = clip_by_norm(small, torch.tensor(2.0))
assert torch.allclose(not_clipped, small, atol=1e-6)

alpha = 0.1
normalized_update = -alpha * g / torch.linalg.norm(g)
assert torch.allclose(torch.linalg.norm(normalized_update), torch.tensor(alpha), atol=1e-6)

print("OK: clipping and normalized update behave correctly.")
"""))

cells.append(md(r"""
## Реализация методов

Все методы используют одинаковую начальную инициализацию и один train/test split.

В истории сохраняются:

- loss;
- норма градиента;
- effective step norm;
- локальная оценка гладкости;
- доля clipping.
"""))

cells.append(code(r"""
def _append_history(history, method, epoch, loss, grad, prev_grad, params, prev_params, update_norm, clipped):
    grad_norm = float(torch.linalg.norm(grad).cpu())
    if prev_grad is None or prev_params is None:
        local_smoothness = np.nan
    else:
        dg = torch.linalg.norm(grad - prev_grad)
        dx = torch.linalg.norm(params - prev_params)
        local_smoothness = float((dg / (dx + 1e-12)).cpu())
    history.append({
        "method": method,
        "epoch": epoch,
        "loss": loss,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "local_smoothness": local_smoothness,
        "clipped": clipped,
    })


def run_gd(epochs=250, lr=0.08):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None

    for epoch in range(epochs):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        update = -lr * grad
        apply_flat_update(model, update)
        _append_history(history, "GD", epoch, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), False)
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)


def run_clip_gd(epochs=250, lr=0.08, clip_radius=1.0):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None

    for epoch in range(epochs):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        clipped_grad = clip_by_norm(grad, torch.tensor(clip_radius, device=device))
        update = -lr * clipped_grad
        apply_flat_update(model, update)
        was_clipped = bool(torch.linalg.norm(grad) > clip_radius)
        _append_history(history, "ClipGD", epoch, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), was_clipped)
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)


def run_normalized_gd(epochs=250, alpha=0.035, beta=1e-8):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None

    for epoch in range(epochs):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        update = -alpha * grad / (torch.linalg.norm(grad) + beta)
        apply_flat_update(model, update)
        _append_history(history, "NGD", epoch, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), False)
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)


def run_nsgdm(epochs=250):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None
    m = None

    for epoch in range(1, epochs + 1):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        if m is None:
            m = torch.zeros_like(grad)
        beta_t = 1.0 - epoch ** (-0.5)
        eta_t = 0.12 * epoch ** (-0.75)
        m = beta_t * m + (1.0 - beta_t) * grad
        update = -eta_t * m / (torch.linalg.norm(m) + 1e-12)
        apply_flat_update(model, update)
        _append_history(history, "NSGD-M", epoch - 1, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), False)
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)


def run_backtracking_gd(epochs=250, eta0=1.0, beta=0.5, gamma=1e-4):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None
    eta_prev = eta0

    for epoch in range(epochs):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        grad_sq = float(torch.dot(grad, grad).cpu())
        eta = eta_prev

        while True:
            trial_params = params - eta * grad
            set_flat_params(model, trial_params)
            trial_loss = float(train_loss(model).detach().cpu())
            if trial_loss <= loss - gamma * eta * grad_sq or eta < 1e-8:
                break
            eta *= beta

        update = -eta * grad
        set_flat_params(model, params)
        apply_flat_update(model, update)
        eta_prev = eta

        _append_history(history, "Backtracking", epoch, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), False)
        history[-1]["eta"] = eta
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)


def run_polyak(epochs=250, f_star=0.0, max_lr=1.0):
    model = reset_model()
    history = []
    prev_grad = None
    prev_params = None

    for epoch in range(epochs):
        loss, grad = loss_and_grad(model)
        params = flat_params(model)
        grad_sq = torch.dot(grad, grad)
        eta = max(0.0, min(max_lr, (loss - f_star) / (float(grad_sq.cpu()) + 1e-12)))
        update = -eta * grad
        apply_flat_update(model, update)
        _append_history(history, "Polyak", epoch, loss, grad, prev_grad, params, prev_params,
                        float(torch.linalg.norm(update).cpu()), False)
        history[-1]["eta"] = eta
        prev_grad = grad.detach().clone()
        prev_params = params.detach().clone()
    return model, pd.DataFrame(history)
"""))

cells.append(md(r"""
## Тест 3. Backtracking на квадратичной функции
"""))

cells.append(code(r"""
def armijo_step_quadratic(x, L=5.0, eta0=1.0, beta=0.5, gamma=1e-4):
    f = lambda z: 0.5 * L * z ** 2
    grad = L * x
    eta = eta0
    while f(x - eta * grad) > f(x) - gamma * eta * grad ** 2:
        eta *= beta
    return eta, f(x), f(x - eta * grad)


eta, before, after = armijo_step_quadratic(2.0)
assert after < before
assert eta > 0
print("OK: Armijo backtracking decreases a quadratic objective.")
"""))

cells.append(md(r"""
## Запуск эксперимента
"""))

cells.append(code(r"""
runs = []
models = {}

for name, runner in [
    ("GD", lambda: run_gd(epochs=250, lr=0.08)),
    ("ClipGD", lambda: run_clip_gd(epochs=250, lr=0.08, clip_radius=0.35)),
    ("NGD", lambda: run_normalized_gd(epochs=250, alpha=0.035)),
    ("NSGD-M", lambda: run_nsgdm(epochs=250)),
    ("Backtracking", lambda: run_backtracking_gd(epochs=250, eta0=1.0)),
    ("Polyak", lambda: run_polyak(epochs=250, f_star=0.0, max_lr=1.0)),
]:
    model, hist = runner()
    models[name] = model
    runs.append(hist)

history = pd.concat(runs, ignore_index=True)

metrics = []
for name, model in models.items():
    row = {"method": name}
    row.update(evaluate(model))
    final_hist = history[history["method"] == name].iloc[-1]
    row["final_train_loss"] = final_hist["loss"]
    row["final_grad_norm"] = final_hist["grad_norm"]
    row["mean_update_norm"] = history[history["method"] == name]["update_norm"].mean()
    row["clip_rate"] = history[history["method"] == name]["clipped"].mean()
    metrics.append(row)

metrics_df = pd.DataFrame(metrics).sort_values(["log_loss", "final_train_loss"])
metrics_df
"""))

cells.append(md(r"""
## Тест 4. Методы должны обучаться

Проверяем базовые свойства:

- loss у каждого метода уменьшился;
- хотя бы несколько методов дали accuracy выше 0.85;
- таблица метрик не содержит NaN.
"""))

cells.append(code(r"""
for method in history["method"].unique():
    h = history[history["method"] == method]
    assert h["loss"].iloc[-1] < h["loss"].iloc[0], f"{method}: loss did not decrease"

assert metrics_df["accuracy"].isna().sum() == 0
assert (metrics_df["accuracy"] >= 0.85).sum() >= 3

print("OK: optimization tests passed.")
"""))

cells.append(md(r"""
## Графики обучения
"""))

cells.append(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for method, h in history.groupby("method"):
    axes[0, 0].plot(h["epoch"], h["loss"], label=method)
    axes[0, 1].plot(h["epoch"], h["grad_norm"], label=method)
    axes[1, 0].plot(h["epoch"], h["update_norm"], label=method)

smooth = history.dropna(subset=["local_smoothness"])
for method, h in smooth.groupby("method"):
    axes[1, 1].scatter(h["grad_norm"], h["local_smoothness"], s=8, alpha=0.45, label=method)

axes[0, 0].set_title("Train loss")
axes[0, 1].set_title("Gradient norm")
axes[1, 0].set_title("Effective update norm")
axes[1, 1].set_title("Local smoothness vs gradient norm")

for ax in axes.ravel():
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

axes[1, 1].set_xlabel("||g_k||")
axes[1, 1].set_ylabel("L_hat")
plt.tight_layout()
plt.savefig(FIG_DIR / "part2_optimizer_comparison.png", dpi=160)
plt.show()
"""))

cells.append(md(r"""
## Проверка `(L0, L1)`-гипотезы по траектории

Оцениваем корреляцию между локальной гладкостью $\hat L_k$ и нормой градиента $\|g_k\|$.
Положительная корреляция не доказывает условие строго, но является эмпирическим признаком, что модель

$$
\hat L_k \approx L_0 + L_1\|g_k\|
$$

имеет смысл для данной задачи.
"""))

cells.append(code(r"""
corr_rows = []
for method, h in history.dropna(subset=["local_smoothness"]).groupby("method"):
    if len(h) < 3:
        corr = np.nan
        l0 = np.nan
        l1 = np.nan
    else:
        corr = h[["grad_norm", "local_smoothness"]].corr().iloc[0, 1]
        x = h["grad_norm"].to_numpy()
        y_l = h["local_smoothness"].to_numpy()
        A = np.column_stack([np.ones_like(x), x])
        l0, l1 = np.linalg.lstsq(A, y_l, rcond=None)[0]
    corr_rows.append({
        "method": method,
        "corr(L_hat, ||g||)": corr,
        "fit_L0": max(0.0, l0) if not np.isnan(l0) else np.nan,
        "fit_L1": max(0.0, l1) if not np.isnan(l1) else np.nan,
    })

corr_df = pd.DataFrame(corr_rows).sort_values("corr(L_hat, ||g||)", ascending=False)
corr_df
"""))

cells.append(md(r"""
## Тест 5. Локальная гладкость считается корректно
"""))

cells.append(code(r"""
assert history["grad_norm"].gt(0).all()
assert history["update_norm"].ge(0).all()
assert history["local_smoothness"].dropna().ge(0).all()
assert len(corr_df) == history["method"].nunique()

print("OK: local smoothness diagnostics are valid.")
"""))

cells.append(md(r"""
## Вывод

В этом notebook реализована практическая часть для методов, связанных с `(L0, L1)`-гладкостью.

Ключевые наблюдения, которые надо смотреть после выполнения:

- `ClipGD` ограничивает длину шага при больших градиентах и тем самым стабилизирует обучение;
- `NGD` делает шаг почти фиксированной длины, поэтому тоже защищается от резких областей;
- `NSGD-M` является parameter-agnostic вариантом с momentum и нормализацией;
- `Backtracking` подбирает шаг через условие Armijo и не требует заранее знать `L0`, `L1`;
- `Polyak` может работать хорошо, если нижняя оценка `F*` выбрана разумно;
- корреляция `corr(L_hat, ||g||)` показывает, насколько на этой задаче видна идея `(L0, L1)`-гладкости.

Для отчета можно использовать таблицы `metrics_df`, `corr_df` и график `figures_part2/part2_optimizer_comparison.png`.
"""))

nb["cells"] = cells

with open("part2/relaxed_smoothness_experiments.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created part2/relaxed_smoothness_experiments.ipynb")
