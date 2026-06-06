# Теория для второй части проекта: методы оптимизации при `(L0, L1)`-гладкости

## 1. Постановка задачи обучения

В первой части проекта рассматривалась бинарная классификация и сравнение функций потерь BCE и MSE при обучении нейронной сети. Во второй части удобно записать обучение как задачу минимизации эмпирического риска по параметрам сети:

$$
\min_{\theta \in \mathbb{R}^d} F(\theta),
\qquad
F(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i) + r(\theta).
$$

Здесь:

- `theta` - вектор всех параметров сети;
- `f_theta(x)` - выход модели;
- `ell` - функция потерь;
- `r(theta)` - регуляризация, например `lambda ||theta||^2 / 2`;
- `F(theta)` - полный train loss.

Для бинарной классификации сеть обычно выдает логит

$$
z_\theta(x) \in \mathbb{R},
\qquad
p_\theta(x)=\sigma(z_\theta(x))=\frac{1}{1+\exp(-z_\theta(x))}.
$$

Для BCE:

$$
\ell_{\mathrm{BCE}}(p,y)
=-y\log p-(1-y)\log(1-p),
$$

и градиент по логиту:

$$
\frac{\partial \ell_{\mathrm{BCE}}}{\partial z}=p-y.
$$

Для MSE:

$$
\ell_{\mathrm{MSE}}(p,y)=(p-y)^2,
$$

и градиент по логиту:

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial z}
=2(p-y)p(1-p).
$$

Главное отличие: у MSE есть множитель `p(1-p)`, который исчезает при насыщении sigmoid. Поэтому в первой части BCE давала более прямой и устойчивый сигнал градиента.

Во второй части нас интересует не только вид функции потерь, но и то, как меняется локальная гладкость ландшафта `F(theta)` во время обучения.

## 2. Классическая гладкость и ее ограничение

Классическая `L`-гладкость означает липшицевость градиента:

$$
\|\nabla F(x)-\nabla F(y)\| \le L\|x-y\|
\quad \forall x,y.
$$

Если `F` дважды дифференцируема, это эквивалентно

$$
\|\nabla^2 F(x)\| \le L
\quad \forall x.
$$

Из этого следует стандартная лемма о спуске:

$$
F(y) \le F(x)+\langle \nabla F(x), y-x\rangle
+\frac{L}{2}\|y-x\|^2.
$$

Для обычного градиентного спуска

$$
x_{k+1}=x_k-\eta \nabla F(x_k)
$$

при `eta <= 1/L` получаем

$$
F(x_{k+1}) \le F(x_k)-\frac{\eta}{2}\|\nabla F(x_k)\|^2.
$$

Проблема: в нейронных сетях глобальная константа `L` часто слишком большая или фактически неограниченная. Экспериментально в работах по gradient clipping наблюдается, что локальная гладкость коррелирует с нормой градиента: когда градиент большой, локальная норма Гессиана тоже может быть большой.

## 3. `(L0, L1)`-гладкость

Работы, предложенные профессором, используют более слабое условие:

$$
\|\nabla^2 F(x)\| \le L_0 + L_1\|\nabla F(x)\|.
$$

Это называется `(L0, L1)`-гладкостью. При `L1 = 0` получаем обычную `L0`-гладкость.

Смысл:

- `L0` отвечает за обычную базовую гладкость;
- `L1 ||grad F(x)||` разрешает локальной гладкости расти в областях с большим градиентом;
- это лучше соответствует обучению нейронных сетей, где резкие области loss landscape часто совпадают с большими градиентами.

Для недважды дифференцируемой формулировки можно использовать первое-порядковое условие:

$$
\limsup_{\delta \to 0}
\frac{\|\nabla F(x+\delta)-\nabla F(x)\|}{\|\delta\|}
\le L_0+L_1\|\nabla F(x)\|.
$$

Более сильная глобальная first-order форма из Huebler et al. задается так: для всех `x, y` и `c > 0`, если `L1 ||x-y|| <= c`, то

$$
\|\nabla F(x)-\nabla F(y)\|
\le
\left(A_0(c)L_0 + A_1(c)L_1\|\nabla F(x)\|\right)\|x-y\|,
$$

где

$$
A_0(c)=1+e^c-\frac{e^c-1}{c},
\qquad
A_1(c)=\frac{e^c-1}{c}.
$$

Если `F` дважды непрерывно дифференцируема, эта форма эквивалентна гессианному условию.

## 4. Цель сходимости

Для нейронной сети задача обычно невыпуклая. Поэтому вместо глобального минимума ищут `epsilon`-стационарную точку:

$$
\|\nabla F(x)\| \le \varepsilon.
$$

В стохастическом случае часто оценивают:

$$
\mathbb{E}\|\nabla F(x)\| \le \varepsilon
$$

или

$$
\mathbb{E}\|\nabla F(x)\|^2 \le \varepsilon^2.
$$

Обозначим начальный разрыв:

$$
\Delta_0 = F(x_0)-F^*,
$$

где `F*` - нижняя грань функции.

## 5. Стохастический градиент и неточная информация

При mini-batch обучении вместо полного градиента используется стохастический оракул:

$$
g_k = \nabla f(x_k,\xi_k).
$$

Стандартные условия:

$$
\mathbb{E}[g_k \mid x_k]=\nabla F(x_k),
$$

$$
\mathbb{E}\|g_k-\nabla F(x_k)\|^2 \le \sigma^2.
$$

Для batch size `B` обычно

$$
\mathbb{E}\|g_k-\nabla F(x_k)\|^2 \le \frac{\sigma_1^2}{B}.
$$

Если информация неточная или biased, удобно записать:

$$
\tilde g_k=\nabla F(x_k)+\xi_k+b_k,
$$

где

$$
\mathbb{E}[\xi_k \mid x_k]=0,
\qquad
\mathbb{E}\|\xi_k\|^2 \le \sigma_k^2,
\qquad
\|b_k\| \le \delta_k.
$$

Тогда нельзя ожидать сходимость лучше шумового пола порядка `sigma_k + delta_k`. Практический критерий:

$$
\|\nabla F(x)\| \lesssim \varepsilon + \delta + \sigma.
$$

Для сохранения спуска желательно, чтобы bias был относительным:

$$
\delta_k \le \rho \|\nabla F(x_k)\|,
\qquad
0 \le \rho < 1.
$$

Тогда

$$
\langle \nabla F(x_k), \tilde g_k\rangle
\ge (1-\rho)\|\nabla F(x_k)\|^2
$$

в детерминированном случае с ошибкой `||tilde g_k - grad F(x_k)|| <= delta_k`.

## 6. Gradient clipping

Оператор clipping по норме:

$$
\operatorname{clip}_c(g)
=\min\left\{1,\frac{c}{\|g\|}\right\}g.
$$

Clipped GD/SGD:

$$
x_{k+1}=x_k-\eta \operatorname{clip}_c(g_k).
$$

Эквивалентная запись для полного градиента:

$$
x_{k+1}
=x_k-h_k \nabla F(x_k),
$$

где

$$
h_k=\min\left\{\eta_c,\frac{\gamma\eta_c}{\|\nabla F(x_k)\|}\right\}.
$$

Если градиент мал, метод ведет себя как обычный GD с шагом `eta_c`. Если градиент велик, длина шага ограничивается:

$$
\|x_{k+1}-x_k\| \le \eta c.
$$

Это важно при `(L0,L1)`-гладкости: большая норма градиента означает потенциально большую локальную гладкость, поэтому обычный фиксированный шаг может быть слишком агрессивным.

### Оценка Zhang et al. для deterministic clipped GD

При `(L0, L1)`-гладкости, нижней ограниченности `F >= F*` и выборе

$$
\eta_c=\frac{1}{10L_0},
\qquad
\gamma=\min\left\{\frac{1}{\eta_c},\frac{1}{10L_1\eta_c}\right\},
$$

итерационная сложность достижения `||grad F|| <= epsilon` ограничивается величиной вида

$$
O\left(
\frac{L_0\Delta_0}{\varepsilon^2}
+\frac{L_1^2\Delta_0}{L_0}
\right).
$$

Для обычного GD с фиксированным шагом требуется дополнительная верхняя оценка

$$
M=\sup\{\|\nabla F(x)\|: F(x)\le F(x_0)\}<\infty,
$$

и шаг порядка

$$
\eta \le \frac{1}{2(ML_1+L_0)}.
$$

Тогда сложность порядка

$$
O\left(
\frac{(ML_1+L_0)\Delta_0}{\varepsilon^2}
\right).
$$

То есть clipping убирает плохую зависимость от `M`, которая может быть большой при плохой инициализации.

### Стохастический clipped GD

При стохастическом градиенте и ограниченном шуме

$$
\|\hat \nabla F(x)-\nabla F(x)\|\le \tau
$$

используется адаптивный шаг вида

$$
h_k=\min\left\{
\frac{1}{16\eta L_1(\|g_k\|+\tau)},\eta
\right\},
$$

где

$$
\eta=\min\left\{
\frac{1}{20L_0},
\frac{1}{128L_1\tau},
\frac{1}{\sqrt{T}}
\right\}.
$$

Оценка имеет стохастический порядок по `epsilon`:

$$
T=O(\varepsilon^{-4})
$$

с дополнительными членами, зависящими от `L0`, `L1`, `tau` и `Delta_0`. Это типичный порядок для невыпуклой стохастической оптимизации по критерию `E||grad F|| <= epsilon`.

## 7. Normalized gradient descent

Нормализованный градиентный метод:

$$
x_{k+1}
=x_k-\eta_n\frac{\nabla F(x_k)}{\|\nabla F(x_k)\|+\beta}.
$$

Если `beta = 0`, то длина шага почти фиксирована:

$$
\|x_{k+1}-x_k\|=\eta_n.
$$

Связь с clipping:

$$
h_c=\min\left\{\eta_c,\frac{\gamma\eta_c}{\|\nabla F(x)\|}\right\},
\qquad
h_n=\frac{\eta_n}{\|\nabla F(x)\|+\beta}.
$$

При подходящем выборе `gamma eta_c = eta_n` и `eta_c = eta_n / beta` эти шаги отличаются только постоянным множителем:

$$
\frac{1}{2}h_c \le h_n \le 2h_c.
$$

Поэтому для невыпуклой задачи оценки clipped GD переносятся на normalized GD с точностью до констант.

## 8. Parameter-agnostic NSGD-M

Huebler et al. рассматривают parameter-agnostic алгоритм: шаги не требуют знания `L0`, `L1`, `sigma`, `Delta_0`.

Normalized SGD with Momentum:

$$
g_t=\nabla f(x_t,\xi_t),
$$

$$
m_t=\beta_t m_{t-1}+(1-\beta_t)g_t,
$$

$$
x_{t+1}=x_t-\eta_t\frac{m_t}{\|m_t\|}.
$$

Parameter-agnostic расписание:

$$
\beta_t=1-t^{-1/2},
\qquad
\eta_t=\frac{t^{-3/4}}{7}.
$$

При нижней ограниченности, `(L0,L1)`-гладкости и ограниченной дисперсии:

$$
\frac{1}{T}\sum_{t=1}^T
\mathbb{E}\|\nabla F(x_t)\|
\le
\tilde O\left(
\frac{\Delta_1 e^{L_1^2}+\sigma+e^{L_1^2}L_0}{T^{1/4}}
\right).
$$

Отсюда сложность по `epsilon`:

$$
T=\tilde O(\varepsilon^{-4}).
$$

Плюс: не надо знать параметры задачи.

Минус: появляется экспоненциальная зависимость от `L1`.

Если разрешить шагу зависеть от `L1`:

$$
\eta_t=\frac{t^{-3/4}}{12L_1},
$$

то оценка улучшается:

$$
\frac{1}{T}\sum_{t=1}^T
\mathbb{E}\|\nabla F(x_t)\|
\le
\tilde O\left(
\frac{L_1\Delta_1+\sigma+L_0/L_1}{T^{1/4}}
\right).
$$

Это показывает цену полной parameter-agnostic настройки.

## 9. Backtracking line search

Для deterministic задачи Huebler et al. показывают, что можно избежать знания `L0`, `L1` и одновременно не получить экспоненту по `L1`.

Armijo backtracking:

1. Берем параметры `beta, gamma in (0,1)`.
2. На шаге `t` выбираем минимальное `k`, такое что `eta_t = beta^k <= eta_{t-1}` и

$$
F(x_t-\beta^k\nabla F(x_t))
\le
F(x_t)-\gamma\beta^k\|\nabla F(x_t)\|^2.
$$

3. Делаем шаг:

$$
x_{t+1}=x_t-\eta_t\nabla F(x_t).
$$

Оценка:

$$
\frac{1}{T}\sum_{t=1}^T
\|\nabla F(x_t)\|^2
\le
\frac{4L_0\Delta_1+14L_1^2\Delta_1^2}
{\beta\gamma(1-\gamma)T}.
$$

Значит для `||grad F|| <= epsilon`:

$$
T=O\left(
\frac{L_0\Delta_1+L_1^2\Delta_1^2}{\varepsilon^2}
\right).
$$

Это хороший вариант для full-batch эксперимента на `make_moons`, потому что можно считать `F` точно и проверять условие Armijo.

В mini-batch режиме стохастический line search сложнее: случайный loss может нарушать условие Armijo даже при правильном направлении. Тогда нужны либо большие batches, либо периодический full-batch/validation line search, либо двухвыборочные проверки.

## 10. Более точные шаги из Vankov et al.

Vankov et al. выводят шаги из более точной верхней оценки роста функции. Вводится

$$
\varphi(t)=e^t-t-1.
$$

Для `(L0,L1)`-гладкой функции:

$$
F(y)
\le
F(x)+\langle \nabla F(x), y-x\rangle
+
\frac{L_0+L_1\|\nabla F(x)\|}{L_1^2}
\varphi(L_1\|y-x\|).
$$

Минимизация этой верхней оценки по направлению `-grad F(x)` дает оптимальный шаг:

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

Упрощенный шаг:

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

Эти шаги упорядочены:

$$
\eta_k^{\mathrm{cl}}
\le
\eta_k^{\mathrm{si}}
\le
\eta_k^*.
$$

Ключевая оценка прогресса:

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
\frac{3L_1F_0}{a\varepsilon},
$$

где `F0 = F(x0)-F*`, `a = 1` для `eta*` и `eta_si`, а `a = 1/2` для clipped шага.

Это улучшенная оценка по сравнению со старой оценкой clipped GD: появляется член `L1 F0 / epsilon`, а не постоянный член порядка `L1^2 F0 / L0`.

## 11. Выпуклый случай: логистическая регрессия или последний слой

Полная нейронная сеть невыпуклая. Но если исследовать:

- логистическую регрессию;
- линейный классификатор поверх фиксированных признаков;
- только последний слой при замороженной feature extractor,

то задача с BCE может быть выпуклой.

Для convex `(L0,L1)`-гладкой функции и шагов `eta*`, `eta_si`, `eta_cl`:

$$
F(x_K)-F^* \le \varepsilon
$$

если

$$
K
\ge
\frac{2}{a}\frac{L_0R^2}{\varepsilon}
+
\frac{3}{a}L_1R\ln\frac{F_0}{\varepsilon},
$$

где

$$
R=\|x_0-x^*\|.
$$

Грубая форма:

$$
K=O\left(
\frac{L_0R^2}{\varepsilon}
+(L_1R)^2
\right).
$$

## 12. Normalized gradient method для convex задачи

NGM:

$$
x_{k+1}
=
x_k-\frac{\beta_k}{\|\nabla F(x_k)\|}
\nabla F(x_k).
$$

При фиксированном горизонте `K`:

$$
\beta_k=\frac{\hat R}{\sqrt{K+1}},
\qquad
0\le k\le K-1.
$$

Если `R = ||x0-x*||`, то

$$
\bar R=\frac{1}{2}\left(\frac{R^2}{\hat R}+\hat R\right).
$$

Оценка:

$$
\min_{0\le k\le K} F(x_k)-F^* \le \varepsilon
$$

если

$$
K+1
\ge
\max\left\{
\frac{L_0\bar R^2}{\varepsilon},
\frac{4}{9}(L_1\bar R)^2
\right\}.
$$

Лучший случай при `hat R = R`:

$$
K=O\left(
\frac{L_0R^2}{\varepsilon}
+(L_1R)^2
\right).
$$

Преимущество: не нужно знать `L0`, `L1`. Нужно только оценить масштаб расстояния `R`.

## 13. Метод с шагами Поляка

Если известно оптимальное значение `F*`, можно использовать шаг Поляка:

$$
\eta_k
=
\frac{F(x_k)-F^*}{\|\nabla F(x_k)\|^2}.
$$

В overparameterized обучении иногда можно считать `F* approx 0`, особенно если train loss теоретически может быть почти нулевым.

Для convex `(L0,L1)`-гладкой функции:

$$
\min_{0\le k\le K}F(x_k)-F^* \le \varepsilon
$$

если

$$
K+1
\ge
\max\left\{
\frac{4L_0R^2}{\varepsilon},
(6L_1R)^2
\right\}.
$$

Преимущество: не нужны `L0`, `L1`.

Недостаток: нужно знать или хорошо оценить `F*`.

Если используется приближение `\hat F*`, то шаг

$$
\eta_k
=
\frac{F(x_k)-\hat F^*}{\|\nabla F(x_k)\|^2}
$$

безопасен только при `hat F* <= F*` или при дополнительном ограничении сверху:

$$
\eta_k \leftarrow \min\{\eta_k,\eta_{\max}\}.
$$

## 14. Ускоренный метод для convex `(L0,L1)`-гладких функций

Vankov et al. также рассматривают ускоренный метод AGMsDR. Он применим к выпуклой задаче и требует на каждом шаге одномерную минимизацию.

Основная оценка:

$$
F(x_k)-F^* \le \varepsilon
$$

если

$$
k
\ge
\sqrt{\frac{48L_0R^2}{a\varepsilon}}
+
\left\lceil
3\left(\frac{2}{a}L_1R\right)^{2/3}
\right\rceil
\left\lceil
\log_2\frac{2F_0}{\varepsilon}
\right\rceil.
$$

Число oracle-запросов порядка

$$
(\nu+1)k,
$$

где `nu` - число запросов для одномерной минимизации.

Для полной MLP-сети этот результат напрямую не применим, потому что задача невыпуклая. Но его можно исследовать на convex подзадаче: логистическая регрессия, последний слой, или фиксированные признаки.

## 15. Адаптивный подбор параметров для проекта

Профессор отдельно упомянул улучшенные оценки с адаптивно подбираемыми параметрами. Для эксперимента можно рассмотреть несколько уровней адаптивности.

### 15.1 Оценка локальной гладкости по траектории

На итерациях можно оценивать локальную константу:

$$
\hat L_k
=
\frac{\|g_k-g_{k-1}\|}
{\|x_k-x_{k-1}\|+\epsilon_{\mathrm{num}}}.
$$

По модели `(L0,L1)`:

$$
\hat L_k \approx L_0+L_1\|g_k\|.
$$

Тогда `L0`, `L1` можно подбирать регрессией:

$$
(\hat L_0,\hat L_1)
=
\arg\min_{a,b\ge 0}
\sum_{j\le k}
\left(\hat L_j-a-b\|g_j\|\right)^2.
$$

Практически лучше использовать robust fit или квантильную верхнюю оболочку, потому что нам нужна не средняя линия, а безопасная верхняя оценка:

$$
\hat L_j \le \hat L_0+\hat L_1\|g_j\|
$$

для большинства последних точек.

### 15.2 Adaptive clipping threshold

Если шаг clipping записан как

$$
\eta_k^{\mathrm{cl}}
=
\min\left\{
\frac{1}{2L_0},
\frac{1}{3L_1\|g_k\|}
\right\},
$$

то ему соответствует clipping radius порядка

$$
c \sim \frac{L_0}{L_1}.
$$

Адаптивный вариант:

$$
c_k=\frac{\hat L_{0,k}}{\hat L_{1,k}+\epsilon_{\mathrm{num}}}.
$$

Либо более устойчиво:

$$
c_k=\operatorname{Quantile}_{q}\{\|g_j\|: j\in[k-w,k]\},
$$

где `q` обычно `0.8-0.95`, `w` - окно.

### 15.3 Backtracking как parameter-free подбор шага

Для full-batch:

$$
\eta_k=\max\{\beta^m\eta_{k-1}: m\ge 0,\;
F(x_k-\beta^m\eta_{k-1}g_k)
\le F(x_k)-\gamma\beta^m\eta_{k-1}\|g_k\|^2\}.
$$

Этот вариант не требует `L0`, `L1` и имеет deterministic оценку:

$$
\frac{1}{T}\sum_{k=1}^T\|\nabla F(x_k)\|^2
=
O\left(
\frac{L_0\Delta_0+L_1^2\Delta_0^2}{T}
\right).
$$

### 15.4 Адаптация batch size при шуме

Если variance у mini-batch градиента:

$$
\mathbb{E}\|g_k-\nabla F(x_k)\|^2 \le \frac{\sigma_1^2}{B_k},
$$

то для целевой точности `epsilon` надо держать шум ниже уровня градиента:

$$
\frac{\sigma_1}{\sqrt{B_k}} \lesssim \varepsilon.
$$

Адаптивное правило:

$$
B_k
\ge
\frac{\hat \sigma_k^2}
{\alpha^2\|g_k\|^2+\epsilon_{\mathrm{floor}}^2}.
$$

Так batch растет, когда градиент становится малым.

## 16. Как применить к текущей задаче обучения

Для второй части проекта можно исследовать следующие методы на той же задаче `make_moons`:

1. `SGD` или `SGD + momentum` как baseline.
2. `ClipSGD`:

$$
x_{k+1}=x_k-\eta\operatorname{clip}_{c_k}(g_k).
$$

3. `NGD`:

$$
x_{k+1}=x_k-\alpha_k\frac{g_k}{\|g_k\|+\beta}.
$$

4. `NSGD-M`:

$$
m_k=\beta_km_{k-1}+(1-\beta_k)g_k,
\qquad
x_{k+1}=x_k-\eta_k\frac{m_k}{\|m_k\|}.
$$

5. `Backtracking GD` на full-batch loss.
6. `Polyak step` для случая, где можно принять `F* approx 0`.

Для каждого метода надо логировать:

- train loss;
- validation/test loss;
- accuracy, F1;
- `||g_k||`;
- effective step `||x_{k+1}-x_k||`;
- clipping ratio:

$$
\mathbf{1}\{\|g_k\|>c_k\};
$$

- оценку локальной гладкости:

$$
\hat L_k
=
\frac{\|g_k-g_{k-1}\|}
{\|x_k-x_{k-1}\|+\epsilon_{\mathrm{num}}};
$$

- корреляцию между `hat L_k` и `||g_k||`.

Главная проверка гипотезы:

$$
\hat L_k \approx L_0+L_1\|g_k\|.
$$

Если корреляция положительная, clipping/normalized методы теоретически оправданы именно для этой задачи.

## 17. Что считать "улучшенной оценкой" для отчета

В отчете можно выделить три уровня оценок.

### Старый уровень

Для clipped GD:

$$
T
=
O\left(
\frac{L_0\Delta_0}{\varepsilon^2}
+\frac{L_1^2\Delta_0}{L_0}
\right).
$$

### Улучшенный deterministic nonconvex уровень

Для шагов `eta*`, `eta_si`, `eta_cl`:

$$
T
=
O\left(
\frac{L_0F_0}{\varepsilon^2}
+\frac{L_1F_0}{\varepsilon}
\right).
$$

Эта оценка лучше отражает уменьшение нормы градиента и не содержит зависимости от начальной нормы градиента.

### Parameter-agnostic уровень

Для NSGD-M:

$$
\frac{1}{T}\sum_{t=1}^T
\mathbb{E}\|\nabla F(x_t)\|
=
\tilde O(T^{-1/4}),
$$

то есть

$$
T=\tilde O(\varepsilon^{-4}).
$$

Метод не требует `L0`, `L1`, но платит экспонентой по `L1`.

Для deterministic backtracking:

$$
\frac{1}{T}\sum_{t=1}^T
\|\nabla F(x_t)\|^2
=
O(T^{-1}),
$$

без знания `L0`, `L1`.

### Учет неточной информации

При biased/noisy градиенте целевая точность заменяется на эффективную:

$$
\varepsilon_{\mathrm{eff}}
=
\varepsilon + C_1\delta + C_2\sigma.
$$

В стохастическом случае для mini-batch:

$$
\sigma^2 \sim \frac{\sigma_1^2}{B}.
$$

Поэтому увеличение batch size улучшает noise floor, но увеличивает стоимость одной итерации.

## 18. Вывод для второй части

Теоретическая идея второй части такая:

1. Обучение сети записывается как невыпуклая оптимизация `F(theta)`.
2. Вместо глобальной `L`-гладкости используется `(L0,L1)`-гладкость:

$$
\|\nabla^2F(\theta)\|\le L_0+L_1\|\nabla F(\theta)\|.
$$

3. Из-за зависимости гладкости от нормы градиента обычный fixed-step SGD может быть слишком медленным или нестабильным.
4. Gradient clipping и normalized gradient автоматически уменьшают эффективный шаг при больших градиентах.
5. NSGD-M дает parameter-agnostic стохастическую сходимость, но с худшей зависимостью от `L1`.
6. Backtracking line search дает parameter-free deterministic сходимость без экспоненты по `L1`.
7. Новые шаги Vankov et al. дают улучшенную невыпуклую оценку:

$$
O\left(
\frac{L_0F_0}{\varepsilon^2}
+\frac{L_1F_0}{\varepsilon}
\right).
$$

8. При неточной информации оценки надо дополнять шумовым полом `delta + sigma`, а параметры clipping/шага подбирать по локальным оценкам гладкости или через backtracking.

## Источники

1. Jingzhao Zhang, Tianxing He, Suvrit Sra, Ali Jadbabaie. *Why gradient clipping accelerates training: A theoretical justification for adaptivity*, ICLR 2020. https://arxiv.org/abs/1905.11881
2. Florian Huebler, Junchi Yang, Xiang Li, Niao He. *Parameter-Agnostic Optimization under Relaxed Smoothness*, AISTATS 2024. https://proceedings.mlr.press/v238/hubler24a.html
3. Daniil Vankov, Anton Rodomanov, Angelia Nedich, Lalitha Sankar, Sebastian U. Stich. *Optimizing `(L0, L1)`-Smooth Functions by Gradient Methods*, arXiv:2410.10800, v3, 2025. https://arxiv.org/abs/2410.10800
