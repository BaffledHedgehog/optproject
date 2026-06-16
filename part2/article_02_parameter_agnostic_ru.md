# Статья 2. Parameter-Agnostic Optimization under Relaxed Smoothness

Источник: Florian Huebler, Junchi Yang, Xiang Li, Niao He. *Parameter-Agnostic Optimization under Relaxed Smoothness*, AISTATS 2024.  
Ссылка: https://proceedings.mlr.press/v238/hubler24a.html

Это подробный русский конспект-перевод, а не дословный перевод всей статьи.

## 1. Главная идея статьи

Статья отвечает на практический вопрос: можно ли получить теоретически обоснованный метод для relaxed smoothness, который не требует заранее знать параметры задачи?

Обычно в теории шаг зависит от неизвестных величин:

$$
L_0,\quad L_1,\quad \sigma,\quad \Delta_0.
$$

Здесь:

- `L0`, `L1` - параметры relaxed smoothness;
- `sigma` - уровень шума стохастического градиента;
- `Delta0 = F(x_0)-F^*` - начальный разрыв.

Но в обучении нейросети мы почти никогда не знаем эти величины. Поэтому авторы изучают parameter-agnostic методы.

`Parameter-agnostic` означает: параметры алгоритма не требуют знания параметров задачи. Например, шаг зависит только от номера итерации:

$$
\eta_t \sim t^{-3/4}.
$$

## 2. Relaxed smoothness

Статья работает с relaxed smoothness, близкой к `(L0,L1)`-гладкости:

$$
\|\nabla^2 F(x)\|
\le
L_0+L_1\|\nabla F(x)\|.
$$

Эквивалентная first-order идея:

$$
\|\nabla F(x)-\nabla F(y)\|
\lesssim
\left(L_0+L_1\|\nabla F(x)\|\right)\|x-y\|
$$

для достаточно близких `x` и `y`.

Это условие слабее классической глобальной `L`-гладкости:

$$
\|\nabla^2F(x)\|\le L.
$$

При `L1=0` relaxed smoothness превращается в обычную гладкость.

## 3. Почему parameter-agnostic важно

Если метод требует

$$
\eta \le \frac{1}{L_0+L_1\|\nabla F(x)\|},
$$

то надо знать `L0` и `L1`. В реальной задаче обучения это неудобно:

- оценивать Гессиан дорого;
- параметры меняются вдоль траектории;
- стохастический шум мешает точной оценке;
- плохая оценка `L0`, `L1` может сделать шаг слишком маленьким или нестабильным.

Parameter-agnostic алгоритм должен работать без этой информации.

## 4. Стохастическая постановка

Цель:

$$
\min_{x\in\mathbb{R}^d} F(x).
$$

Доступен стохастический градиент:

$$
g_t=\nabla f(x_t,\xi_t).
$$

Предположения стандартные:

$$
\mathbb{E}[g_t\mid x_t]=\nabla F(x_t),
$$

$$
\mathbb{E}\|g_t-\nabla F(x_t)\|^2\le \sigma^2.
$$

Так как задача может быть невыпуклой, цель - найти точку с малым градиентом:

$$
\mathbb{E}\|\nabla F(x)\|\le \varepsilon.
$$

## 5. NSGD-M

Основной алгоритм статьи - normalized stochastic gradient descent with momentum.

На шаге `t`:

$$
g_t=\nabla f(x_t,\xi_t).
$$

Momentum:

$$
m_t=\beta_t m_{t-1}+(1-\beta_t)g_t.
$$

Update:

$$
x_{t+1}=x_t-\eta_t\frac{m_t}{\|m_t\|}.
$$

Если `m_t` почти равен градиенту, то метод делает шаг в направлении антиградиента, но нормирует его длину.

## 6. Parameter-agnostic расписание

Авторы выбирают:

$$
\beta_t=1-t^{-1/2},
$$

$$
\eta_t=\frac{t^{-3/4}}{7}.
$$

Эти формулы не используют:

$$
L_0,\quad L_1,\quad \sigma,\quad \Delta_0.
$$

Именно поэтому алгоритм parameter-agnostic.

Интуиция:

- momentum сглаживает шум mini-batch градиентов;
- normalization защищает от больших градиентов;
- убывающий шаг обеспечивает асимптотическую сходимость.

## 7. Основная оценка

Для невыпуклой задачи авторы получают оценку среднего градиента вдоль траектории:

$$
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}\|\nabla F(x_t)\|
\le
\tilde O\left(T^{-1/4}\right),
$$

с множителями, зависящими от `L0`, `L1`, `sigma` и начального разрыва.

Отсюда для достижения

$$
\mathbb{E}\|\nabla F(x)\|\le \varepsilon
$$

нужно

$$
T=\tilde O(\varepsilon^{-4}).
$$

Это типичный порядок для стохастической невыпуклой оптимизации.

## 8. Цена parameter-agnostic подхода

Если алгоритм совсем не знает `L1`, в оценке может появиться плохая зависимость от `L1`, например экспоненциальная.

Упрощенно:

$$
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}\|\nabla F(x_t)\|
\le
\tilde O\left(
\frac{\Delta_0 e^{L_1^2}+\sigma+e^{L_1^2}L_0}{T^{1/4}}
\right).
$$

Если разрешить алгоритму знать `L1`, можно выбрать шаг осторожнее:

$$
\eta_t=\frac{t^{-3/4}}{12L_1}.
$$

Тогда оценка улучшается:

$$
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}\|\nabla F(x_t)\|
\le
\tilde O\left(
\frac{L_1\Delta_0+\sigma+L_0/L_1}{T^{1/4}}
\right).
$$

Смысл: полная независимость от параметров удобна, но за нее приходится платить худшими константами.

## 9. Backtracking для deterministic задачи

В статье также рассматривается deterministic режим, где доступен полный градиент и полный loss. Тогда можно использовать backtracking line search.

Обычный шаг:

$$
x_{t+1}=x_t-\eta_t\nabla F(x_t).
$$

Шаг `eta_t` подбирается через условие Armijo:

$$
F(x_t-\eta_t\nabla F(x_t))
\le
F(x_t)-\gamma\eta_t\|\nabla F(x_t)\|^2.
$$

Если условие не выполнено, шаг уменьшается:

$$
\eta_t \leftarrow \beta \eta_t,
\qquad
0<\beta<1.
$$

Backtracking тоже parameter-free по отношению к `L0`, `L1`: он сам находит допустимый шаг через проверку loss.

## 10. Оценка для backtracking

Для deterministic задачи при relaxed smoothness авторы получают оценку вида:

$$
\frac{1}{T}\sum_{t=1}^{T}
\|\nabla F(x_t)\|^2
\le
O\left(
\frac{L_0\Delta_0+L_1^2\Delta_0^2}{T}
\right).
$$

То есть средняя squared gradient norm убывает как

$$
O(T^{-1}).
$$

Для критерия

$$
\|\nabla F(x)\|\le \varepsilon
$$

это дает порядок

$$
T=O(\varepsilon^{-2}).
$$

## 11. Почему full-batch важен для Armijo

Условие Armijo требует честного сравнения:

$$
F(x_t-\eta g_t)
\quad \text{и} \quad
F(x_t).
$$

В full-batch режиме это можно сделать точно.

В mini-batch режиме loss шумный:

$$
F_B(x)=\frac{1}{B}\sum_{i\in B}\ell_i(x).
$$

Тогда условие Armijo может случайно выполниться или случайно провалиться. Поэтому stochastic line search требует дополнительных приемов:

- большие batch;
- отдельная validation/check batch;
- усреднение loss по нескольким batch;
- ограничение на число backtracking попыток.

## 12. Связь с нашим notebook

В notebook для второй части можно использовать:

1. `NSGD-M` как parameter-agnostic стохастический метод:

$$
\beta_t=1-t^{-1/2},
\qquad
\eta_t=Ct^{-3/4}.
$$

2. `Backtracking GD` как deterministic full-batch метод:

$$
F(x-\eta g)\le F(x)-\gamma\eta\|g\|^2.
$$

3. Сравнение с обычным GD:

$$
x_{k+1}=x_k-\eta g_k.
$$

4. Логирование:

$$
\|g_k\|,
\quad
\|x_{k+1}-x_k\|,
\quad
F(x_k).
$$

## 13. Короткий вывод

Главный вклад статьи:

1. relaxed smoothness лучше описывает обучение моделей, чем глобальная `L`-гладкость;
2. можно строить методы, которые не знают `L0`, `L1`, `sigma`;
3. NSGD-M дает стохастическую сходимость порядка

$$
\tilde O(T^{-1/4});
$$

4. deterministic backtracking дает parameter-free подбор шага и оценку порядка

$$
O(T^{-1})
$$

для средней squared gradient norm.
