#!/usr/bin/env python3
"""Merge package pins from one or more requirements files into the root
requirements.txt, deduplicating by package name.

Usage: merge_requirements.py <requirements-file> [more-files...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-\[\]]+)")


def normalize_name(line: str) -> str | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http://", "https://")):
        return None
    match = NAME_RE.match(line)
    if not match:
        return None
    return match.group(1).split("[")[0].lower().replace("_", "-")


def load_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        name = normalize_name(line)
        if name is None:
            continue
        pins[name] = line
    return pins


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: merge_requirements.py <requirements-file> [more-files...]")

    merged = load_pins(ROOT_REQUIREMENTS)

    for arg in sys.argv[1:]:
        src_path = Path(arg)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path
        if not src_path.exists():
            raise SystemExit(f"requirements file not found: {src_path}")

        for name, line in load_pins(src_path).items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = line
                print(f"[merge_requirements] + {line}")
            elif existing != line:
                print(
                    f"[merge_requirements] ! conflict for '{name}': "
                    f"keeping '{existing}', ignoring '{line}' from {src_path.name} "
                    "(edit requirements.txt manually if you want the new pin)"
                )

    sorted_lines = [merged[name] for name in sorted(merged)]
    ROOT_REQUIREMENTS.write_text("\n".join(sorted_lines) + "\n", encoding="utf-8")
    print(f"[merge_requirements] wrote {len(sorted_lines)} packages to {ROOT_REQUIREMENTS}")


if __name__ == "__main__":
    main()
