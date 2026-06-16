# Русские материалы по статьям для второй части проекта

Это не дословный перевод PDF целиком, а подробные русские конспекты-переводы: постановка задачи, основные определения, алгоритмы, оценки сходимости и связь с экспериментом на задаче обучения.

## Файлы

1. `article_01_gradient_clipping_ru.md`

   Zhang, He, Sra, Jadbabaie. *Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity*, ICLR 2020.

   Основная идея: gradient clipping можно понимать не только как защиту от exploding gradients, но и как адаптивный выбор шага при relaxed smoothness.

2. `article_02_parameter_agnostic_ru.md`

   Huebler, Yang, Li, He. *Parameter-Agnostic Optimization under Relaxed Smoothness*, AISTATS 2024.

   Основная идея: получить сходимость при relaxed smoothness без знания параметров задачи `L0`, `L1`, дисперсии шума и начального разрыва.

3. `article_03_l0_l1_gradient_methods_ru.md`

   Vankov, Rodomanov, Nedich, Sankar, Stich. *Optimizing `(L0, L1)`-Smooth Functions by Gradient Methods*, arXiv 2024/2025.

   Основная идея: систематически изучить градиентные методы для `(L0,L1)`-гладких функций и получить улучшенные оценки без экспоненциальной зависимости от `L1`.

4. `article_04_convex_l0l1_adaptivity_ru.md`

   Gorbunov, Tupitsa, Choudhury, Aliev, Richtárik, Horváth, Takáč. *Methods for Convex `(L0, L1)`-Smooth Optimization: Clipping, Acceleration, and Adaptivity*, arXiv 2024.

   Основная идея: для выпуклого `(L0,L1)`-случая получить улучшенные оценки для clipping и шага Поляка, ускоренный метод и (главное для нас) **адаптивный** AdGD, подбирающий шаг по локальной оценке гладкости без знания `L0`, `L1`.

## Как использовать в проекте

Для отчета удобно брать:

- из первой статьи: мотивацию gradient clipping и normalized gradient;
- из второй статьи: объяснение слова `parameter-agnostic` и алгоритм NSGD-M;
- из третьей статьи: улучшенные оценки для `(L0,L1)`-smooth функций, Polyak step, normalized gradient и accelerated метод;
- из четвертой статьи: адаптивный AdGD как эталон «адаптивно подбираемых параметров» и улучшенные оценки в выпуклом случае.

Источники:

- https://arxiv.org/abs/1905.11881
- https://proceedings.mlr.press/v238/hubler24a.html
- https://arxiv.org/abs/2410.10800
- https://arxiv.org/abs/2409.14989
