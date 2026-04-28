from __future__ import annotations

from pathlib import Path


def read_modal_requirements(file_path: str | Path) -> list[str]:
    file_path = Path(file_path)
    values: list[str] = []
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(line)
    return values
