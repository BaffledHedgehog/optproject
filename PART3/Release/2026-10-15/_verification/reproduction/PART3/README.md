# PART3: сертификатные методы Франка–Вульфа

Это текущий исследовательский проект. Каталоги первого и второго лабораторных проектов вне PART3 к нему не относятся.

Материалы для печати от **15.10.2026** подготовлены под именами **Yaroslav P. Lyakhov** и **Stepan Suvorov**. Основная статья выделяет сертификатную сложность существующего backtracking FW при ограничении кривизны через H₀+H₁(f−f*), включает полные доказательства и AFW-расширение. Первенство метода и общее эмпирическое превосходство не заявляются. Аффилиации и e-mail пока не указаны; внешняя подача и независимое рецензирование не проводились.

- [Основная статья, PDF](Paper/article.pdf) и [LaTeX](Paper/article.tex).
- [Инструкция к печатному комплекту и подаче](Paper/PRINT_README_RU.txt); [сборка и проверка архивов](Paper/prepare_release.py).
- [Сопоставление с первоисточниками](Work/publication_literature.json).

- [Исследовательская записка и полные доказательства](Work/research.md).
- [Расширенный технический отчёт, PDF](Paper/manuscript.pdf) и [LaTeX](Paper/manuscript.tex): дополнительные выводы и история исследований.
- [Аудит новизны и оставшиеся вопросы](Work/novelty_audit.md).
- [Расширенные результаты: 660 запусков, 11 методов, 6 семейств](Work/benchmark_results/results.md).
- [Новое сравнение: 840 запусков, 12 методов, 7 семейств](Work/adaptive_results/results.md), включая реальную пуассоновскую регрессию с доказанными `H₀,H₁>0`.
- [Контроль одинаковой инициализации: 210 повторных запусков](Work/adaptive_ablation/results.md) и [парная оценка неопределённости](Work/adaptive_ablation/initialization_comparison.csv). Устойчивое практическое преимущество нормировки пока не установлено.
- [350 запусков с AC-FW и обычным backtracking](Work/backtracking_results/results.md), [полная проверка 140 траекторий](Work/backtracking_results/backtracking_audit.csv) и [доказательство новой оценки](Paper/ordinary_backtracking.tex). 280 запусков повторяют прежние экземпляры и методы; 70 запусков AC-FW дополняют сравнение на тех же задачах.
- [Контроль начальной точки и правила адаптации: 280 запусков](Work/initialization_factorial/results.md), [8400 префиксов при заданных бюджетах оракулов](Work/initialization_factorial/budget.csv), [геометрия оптимальной грани](Work/initialization_factorial/face_summary.csv). Это четыре условия на прежних 70 экземплярах, включая 70 точных повторов AC-FW; новых независимых задач нет.
- [FW против away-step: 280 запусков](Work/away_results/results.md), [9240 бюджетных префиксов](Work/away_results/budget.csv), [проверка 70 полных AFW-BT траекторий](Work/away_results/audit.csv) и [доказательства расширения](Paper/away_theory.tex). 140 FW-запусков точно повторяют прежние результаты; 140 AFW-запусков добавляют сравнение на тех же экземплярах.
- [Радиус и единицы функции: 360 парных запусков](Work/robustness_results/results.md), [11880 бюджетных префиксов](Work/robustness_results/budget.csv), [240 сравнений масштабов](Work/robustness_results/scale_checks.csv) и [векторный рисунок](Work/robustness_results/radius_and_units.pdf). Это 30 условий с тремя зависимыми масштабами, без новых наборов данных; 30 прежних baseline воспроизведены точно. Все пуассоновские эталоны в этом диапазоне внутренние.
- [Переход активности: 280 сырых прогонов, 196 после исключения повторов](Work/constraint_results/results.md), [840 точностных префиксов](Work/constraint_results/accuracy.csv), [различные начальные вершины](Work/constraint_results/distinct_starts.csv), [84 подтверждённых повтора](Work/constraint_results/duplicate_checks.csv) и [рисунок](Work/constraint_results/constraint_transition.pdf). Основные сводки дают каждой из семи вершин одинаковый вес. Усиленная теория: [away_theory.tex](Paper/away_theory.tex).
- [Два дополнительных реальных набора: 648 запусков](Work/multidata_results/results.md), [21384 оракульных префикса](Work/multidata_results/budget.csv), [онлайн-сертификат против эталонной ошибки](Work/multidata_results/reference_diagnostics.csv), [векторный рисунок](Work/multidata_results/additional_counts.pdf). Все 30/24 вершины перебраны при активном и внутреннем радиусах, без случайных повторов. Сохранены также незавершённые запуски; преимущество по полному времени не утверждается.
- [Дополнение: 108 запусков исходного AC-FW](Work/count_comparator_results/results.md) на тех же условиях Seoul/Metro. Основная статья сопоставляет 756 записей семи методов на 108 условиях при одинаковом бюджете 1000 значений функции; это не 756 независимых задач. Все начальные обращения и отказы оплачиваются.
- [Причинные сертификаты на 432 повторённых траекториях](Work/certificate_results/results.md), [2160 парных профилей](Work/certificate_results/paired.csv), [71280 оплаченных бюджетных префиксов](Work/certificate_results/budget.csv), [векторный рисунок](Work/certificate_results/causal_certificates.pdf) и [доказательство корректности](Paper/tangent_certificates.tex). Блоки 8/32/128 не улучшили остановку; контроль известной μ использует дополнительную информацию. Это не независимые новые оптимизации.
- [Проверка фазовых доказательств и граничных случаев](Work/proof_audit_results/results.md), [точные счётчики](Work/proof_audit_results/checks.json), [архивный проверяющий код](Work/proof_audit_results/audit_source.py). Валидатор повторно исполняет рациональные проверки. Новых оптимизационных экспериментов нет; прежние десять архивов сохранены.
- [Графики сертификатов](Work/benchmark_results/certificates.png) и [стоимости оракулов](Work/benchmark_results/oracle_costs.png).
- [Параметры, версии зависимостей, SHA-256 кода и интервалы эталонов](Work/benchmark_results/manifest.json).

## Запуск

Из корня репозитория:

```powershell
python -m pip install -r PART3/requirements.txt
python -m unittest discover -s PART3/Work -p "test_*.py" -v
python PART3/Work/validate_results.py
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python PART3/Work/benchmark.py --seeds 10 --max-iter 1000 --tolerance 1e-5 --output PART3/Work/adaptive_results
```

Короткий запуск: `python PART3/Work/benchmark.py --seeds 2 --max-iter 100 --scenarios interior active --output PART3/Work/smoke_results`. Выбор каталога сохраняет основной эксперимент. Файлы результатов в выбранном каталоге перезаписываются при повторном запуске.

`Work/problems.py` задаёт задачи и константы; `Work/fw.py` реализует методы и безопасный сертификат неточной информации; `Work/auto_conditioned.py` — AC-FW; `Work/away_fw.py` — AFW с активными весами; `Work/benchmark.py` считает вызовы оракулов, строит таблицы и графики. Эталоны нужны только для оценки ошибок и геометрии после запуска: оптимизаторы их не читают. Адаптивные методы также не читают константы гладкости. Проверки: 58 тестов, валидация одиннадцати экспериментальных архивов и отдельный повторный точный рациональный аудит доказательств. Проверяются хеши, счётчики, активные веса и границы проб функции. Повторные траектории не считаются независимыми наблюдениями.

Воспроизвести абляцию: `python PART3/Work/benchmark.py --seeds 10 --max-iter 1000 --tolerance 1e-5 --methods adaptive_certificate adaptive_L_initial adaptive_L --output PART3/Work/adaptive_ablation`, затем `python PART3/Work/compare_initialization.py`. Вариант `adaptive_L_initial` доступен в отдельных сравнениях; по умолчанию работают прежние 12 методов.

Новое сравнение: `python PART3/Work/benchmark.py --seeds 10 --max-iter 1000 --tolerance 1e-5 --methods adaptive_L_initial adaptive_L adaptive_certificate AC_FW Armijo --output PART3/Work/backtracking_results`, затем `python PART3/Work/backtracking_audit.py PART3/Work/backtracking_results`. Аудит повторяет обычный backtracking из сохранённого снимка кода и проверяет все шаги, включая пропущенные при записи trace.csv. Для AC-FW использованы алгоритмы 1+2 [Giang-Tran et al.](https://arxiv.org/abs/2605.15512v2) и затухание по (6) с фиксированным `delta=1`; начальный переход и отказы оплачиваются. По умолчанию он не добавлен к прежним 12 методам.

Воспроизвести контроль начальной точки, сохранив записанные результаты:

```powershell
python PART3/Work/initialization_factorial/source/initialization_factorial.py --output PART3/Work/reproduced_factorial
python PART3/Work/face_diagnostics.py PART3/Work/reproduced_factorial
python PART3/Work/report_initialization.py PART3/Work/reproduced_factorial
python PART3/Work/validate_results.py PART3/Work/reproduced_factorial
```

Нужны те же переменные числа потоков и закреплённые версии зависимостей. Новый runner требует пустой каталог. В нём `AC_vertex` воспроизводит исходный AC-FW; остальные три условия — явно обозначенные контрольные варианты. Все получают две оценки функции, два градиента, два LMO, обе нижние оценки и одну общую секущую кривизну; первая проба у AC и BT совпадает. Сравнение выполняется отдельно по бюджетам функции, градиента и LMO. Рисунок для статьи доступен в [PNG](Work/initialization_factorial/initialization_publication.png) и [PDF](Work/initialization_factorial/initialization_publication.pdf); расширенный диапазон бюджетов сохранён отдельно в initialization_effect.*.

Воспроизвести away-step сравнение (с теми же переменными потоков):

```powershell
python PART3/Work/away_results/source/away_benchmark.py --output PART3/Work/reproduced_away
python PART3/Work/report_away.py PART3/Work/reproduced_away
python PART3/Work/validate_results.py PART3/Work/reproduced_away
```

Все четыре метода начинают из исходной вершины. BT не читает `H₀,H₁`; Q получает их аналитические значения. Дополнительные сканирования активного множества и оценки его атомов записаны отдельно. Положительные веса не удаляются по эвристическому порогу. В сравнении по 1000 значениям функции AFW-BT сертифицирует 6/10 стартов Bike Sharing, хотя к концу — 10/10; это разные бюджеты. [Рисунок PNG](Work/away_results/away_comparison.png) и [векторный PDF](Work/away_results/away_comparison.pdf).

Воспроизвести радиусы и масштабы (с теми же переменными потоков):

```powershell
python PART3/Work/robustness_results/source/robustness_benchmark.py --output PART3/Work/reproduced_robustness
python PART3/Work/robustness_results/report_source.py PART3/Work/reproduced_robustness
python PART3/Work/validate_results.py PART3/Work/reproduced_robustness
```

Сравниваются обычные FW/AFW с `M=1` и `M=U₀/D²`; допуск масштабируется вместе с целью. Интервалы эталонов пересчитаны для каждого радиуса, все 360 траекторий проверены до прореживания. `scale_checks.csv` сравнивает исходы, конечные счётчики и нормированные сертификаты при одинаковых бюджетах. Это не проверка побитового совпадения всех итератов.

Воспроизвести переход к активному ограничению (с теми же переменными потоков):

```powershell
python PART3/Work/constraint_results/source/constraint_transition.py --output PART3/Work/reproduced_constraint
python PART3/Work/constraint_results/report_source.py PART3/Work/reproduced_constraint
python PART3/Work/validate_results.py PART3/Work/reproduced_constraint
```

Начальные вершины группируются по фактическому правилу генерации, а не по уникальности seed: для пуассоновских seed 0–9 есть семь разных вершин. Прежние таблицы сохраняют исходные веса seed; новый отчёт исключает повторные старты. Данные и эталон общие для всех стартов одного радиуса. Новая оценка gap проверена на всех 280 полных журналах до прореживания; геометрическая константа не подгоняется.

Воспроизвести сравнение на дополнительных данных (с теми же переменными потоков):

```powershell
python PART3/Work/multidata_results/source/multidata_benchmark.py --output PART3/Work/reproduced_multidata
python PART3/Work/multidata_results/report_source.py PART3/Work/reproduced_multidata
python PART3/Work/validate_results.py PART3/Work/reproduced_multidata
```

Неизменённые официальные файлы включены в `Data/seoul/` и `Data/metro/`, источники и SHA-256 — в `source.json`. Восстановить их можно `python PART3/Work/fetch_count_data.py`. Оба набора — UCI, CC BY 4.0: [Seoul, doi:10.24432/C5F62R](https://doi.org/10.24432/C5F62R), [Metro, John Hogue, doi:10.24432/C5X60B](https://doi.org/10.24432/C5X60B). В новых целях используется train-only ограниченное преобразование `tanh`; оно описано отдельно от прежнего Bike Sharing. Эталонные интервалы служат только анализу после запуска. Полный набор — **49 автоматических тестов**, все девять архивов проверяются общим валидатором.

На реальных данных логистическая регрессия служит контролем `H₁=0`. Пуассоновская регрессия на UCI Bike Sharing даёт естественный случай `H₁>0`; seed в ней меняет только начальную вершину. Данные включены в `Data/bike_sharing/`, автор Hadi Fanaee-T, UCI, [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Исходные файлы не изменены; `source.json` содержит происхождение и SHA-256. При необходимости восстановить: `python PART3/Work/fetch_bike_data.py`. Качество прогноза в этом оптимизационном опыте не оценивается.

По умолчанию новый benchmark пишет в `adaptive_results/`. Прежний опыт на 660 запусках воспроизводится точным снимком: `python PART3/Work/benchmark_results/source/benchmark.py --seeds 10 --max-iter 1000 --tolerance 1e-5 --output PART3/Work/reproduced_baseline`. Каждый сохранённый опыт имеет собственный `source/`; хеши manifest относятся к этому снимку, а не к впоследствии изменённому текущему коду.

Исходный эксперимент сохранён без изменения: `python PART3/Work/experiment.py`. Его таблицы `summary.csv`, `trace.csv` и `convergence.png` относятся к старому протоколу остановки по известной ошибке функции и не смешиваются с `benchmark_results/`.

Таблицы и иллюстрация рукописи генерируются из CSV: `python PART3/Paper/make_tables.py`. Сборка PDF из `PART3/Paper`: `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex` (два запуска; каталог `build` нужно создать). Готовый `build/manuscript.pdf` копируется в `Paper/manuscript.pdf`. Рукопись не подавалась и требует независимой проверки теоремы, углублённого поиска совпадений и согласования авторами.

Воспроизвести причинные сертификатные профили (те же переменные числа потоков, новый пустой каталог):

```powershell
python PART3/Work/certificate_results/source/certificate_benchmark.py --output PART3/Work/reproduced_certificates
python PART3/Work/certificate_results/report_source.py PART3/Work/reproduced_certificates
python PART3/Work/validate_results.py PART3/Work/reproduced_certificates
```

Архив `multidata_results` нужен для проверки точного повторения исходных концов. Мониторы используют уже оплаченные f/g; каждый блочный LMO считается дополнительно. Срез по бюджетам учитывает стоимость отдельного профиля, хотя диагностический прогон вычисляет все пять параллельно по одной траектории. Вызовы `solve(..., on_iterate=...)` и `solve_away(..., on_iterate=...)` позволяют реализовать нативную остановку; исходный trace сохраняет прежнее определение scalar-сертификата, дополнительные счётчики ведёт монитор.

Воспроизвести точный рациональный аудит доказательств:

```powershell
python PART3/Work/proof_audit_results/audit_source.py --output PART3/Work/reproduced_proof_audit
python PART3/Work/validate_results.py PART3/Work/reproduced_proof_audit
```

Всего 57 тестов. Основной валидатор проверяет десять экспериментальных архивов и отдельный рациональный аудит. Старые консервативные `gap_envelope`/`certificate_threshold` сохранены для воспроизведения; новые `sharp_gap_envelope`/`sharp_certificate_threshold` используются только в анализе, а не как входы оптимизатора.

## Исходные источники

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

Конспекты исходных статей являются материалами обзора, а не собственными результатами проекта. `info.txt` сохранён как первоначальное описание направлений.
