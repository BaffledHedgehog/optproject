# Часть 3: статьи и исследование

В [`info.txt`](info.txt) семь разных источников; ссылка `2510.03164` встречается дважды. Для каждого источника подготовлен подробный русский **перевод-конспект**: разделы, определения, основные формулы, теоремы и смысл доказательств. Это самостоятельное изложение содержания, а не построчный перевод полного текста.

| Статья | Русский файл |
| --- | --- |
| Why Do We Need Warm-up? A Theoretical Perspective | [`2510.03164.md`](Translations/2510.03164.md) |
| Frank-Wolfe Algorithms for `(L₀,L₁)`-smooth functions | [`2510.16468.md`](Translations/2510.16468.md) |
| Characterization of Gradient Dominance and Regularity Conditions for Neural Networks | [`1710.06910.md`](Translations/1710.06910.md) |
| Quasar-Convex Optimization: Fundamental Properties and High-Order Proximal-Point Methods | [`2604.26735.md`](Translations/2604.26735.md) |
| Heavy Ball and Nesterov Accelerations with Hessian-driven Damping for Nonconvex Optimization | [`2506.15632.md`](Translations/2506.15632.md) |
| Near-Optimal Methods for Minimizing Star-Convex Functions and Beyond | [`1906.11985.md`](Translations/1906.11985.md) |
| Study of the behaviour of Nesterov Accelerated Gradient in a non convex setting: the strongly quasar convex case | [`hal-04589853v3.md`](Translations/hal-04589853v3.md) |

Для записи HAL прямой документ оказался недоступен из-за проверки браузера. Её конспект составлен по связанной версии [arXiv:2405.19809v3](https://arxiv.org/abs/2405.19809); точное совпадение с HAL v3 не проверено.

Исследовательская работа: [`Work/research.md`](Work/research.md). Она содержит обзор близких публикаций, верхнюю модель и доказательство сходимости шага Франка–Вульфа при `(H₀,H₁)`-гладкости, а также воспроизводимый эксперимент. Отдельно: [`Work/novelty_audit.md`](Work/novelty_audit.md) — обновлённая проверка новизны по статьям до сентября 2026 года. Код: [`Work/experiment.py`](Work/experiment.py), результаты: [`Work/summary.csv`](Work/summary.csv), [`Work/trace.csv`](Work/trace.csv), [`Work/convergence.png`](Work/convergence.png).

Запуск из корня репозитория: `python PART3/Work/experiment.py`. Нужны `numpy` и `matplotlib`.
