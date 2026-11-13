# Код и данные прежних исследований

Этот каталог относится к модели кривизны через `H0 + H1*(f-f*)`
и связанным геометрическим результатам. Он не является кодом текущей v20:
[актуальный комплект](../Submission/2026-11-14-v20/README_RU.txt).

Код, первичные CSV и каталоги `source/` и `frozen/` сохранены, поскольку
они нужны для воспроизведения опубликованных в репозитории таблиц.
Замороженные исходники привязаны к контрольным суммам; похожие файлы
не являются свободно удаляемыми дублями.

- [Исследовательский текст](research.md).
- [Реестр утверждений](research_claims.json).
- [Исправления](research_corrections_2026_10_17.md).
- [Обзор литературы](literature_review_2026_10_17.md).
- [Выпуск от 01.11.2026](../Submission/2026-11-01/README_RU.txt).
- [Endpoint-протокол](endpoint_recovery_strict_results/manifest.json).
- [Проверки сертификатов](certificate_results/results.md).

Проверки из корня репозитория:

```powershell
python -m pip install -r PART3/requirements.txt
python -m unittest discover -s PART3/Work -p "test_*.py" -v
python PART3/Work/validate_results.py
```

Устаревшие отчёты о готовности выпусков и исправленные черновики удалены из
текущего дерева. Старые снимки остаются
[в истории Git](https://github.com/BaffledHedgehog/optproject/tree/3b0c6f2f8bc4c32eb16d9c133d546cb000a21ccd/PART3/Work).
