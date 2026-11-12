"""Restore the original v18 large files from verified local parts.

Run with Python 3 from this release directory:
    python restore_large_files.py

Only the standard library is used. Existing different files are refused.
The script performs no scientific computation and does not rebuild a ZIP.
"""

from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "ArchiveParts"


def checked_path(base, name):
    if not isinstance(name, str):
        raise ValueError("Путь в manifest.json должен быть строкой")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Недопустимый путь в manifest.json: " + name)
    target = base.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        raise ValueError("Путь выходит за пределы каталога выпуска: " + name)
    return target


def verify_bytes(data, expected_size, expected_hash, label):
    if len(data) != expected_size:
        raise ValueError("Не совпал размер: " + label)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Не совпала SHA-256: " + label)


def main():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Неизвестная версия manifest.json")
    limit = manifest["maximum_part_bytes"]
    if not isinstance(limit, int) or not 0 < limit <= 131072:
        raise ValueError("Неверный максимальный размер части")

    prepared = []
    targets = set()
    for entry in manifest["files"]:
        target = checked_path(ROOT, entry["path"])
        if target in targets:
            raise ValueError("Повторяющийся итоговый путь: " + entry["path"])
        targets.add(target)
        pieces = []
        for part in entry["parts"]:
            path = checked_path(ARCHIVE, part["path"])
            data = path.read_bytes()
            if len(data) > limit:
                raise ValueError("Превышен размер части: " + part["path"])
            verify_bytes(data, part["size"], part["sha256"], part["path"])
            pieces.append(data)
        encoded = b"".join(pieces)
        verify_bytes(encoded, entry["encoded_size"], entry["encoded_sha256"], entry["path"] + " (поток)")
        if entry["encoding"] == "gzip":
            original = gzip.decompress(encoded)
        elif entry["encoding"] == "raw":
            original = encoded
        else:
            raise ValueError("Неизвестный способ хранения: " + str(entry["encoding"]))
        verify_bytes(original, entry["original_size"], entry["original_sha256"], entry["path"])
        if target.is_symlink():
            raise FileExistsError("Вместо файла уже есть символическая ссылка: " + entry["path"])
        if target.exists():
            if not target.is_file() or target.read_bytes() != original:
                raise FileExistsError("Уже существует отличающийся файл; он не перезаписан: " + entry["path"])
            prepared.append((target, original, entry["path"], True))
        else:
            prepared.append((target, original, entry["path"], False))

    # All parts, decoded bytes and existing targets are checked before any write.
    for target, original, name, already_present in prepared:
        if already_present:
            print("Уже восстановлен и проверен: " + name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also refuses a file created after the checks above.
        with target.open("xb") as stream:
            stream.write(original)
        if target.read_bytes() != original:
            raise OSError("Не совпали записанные байты: " + name)
        print("Восстановлен и проверен: " + name)
    print("Готово. Файлы побайтово совпадают с исходной редакцией v18.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        sys.exit("Восстановление остановлено: " + str(error))
