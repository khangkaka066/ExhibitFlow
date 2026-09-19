#!/usr/bin/env python3
"""Launch devmem without installing it into the target project."""
from __future__ import annotations

import os
from pathlib import Path
import sys

tool = Path(__file__).resolve().parent
environment = os.environ.copy()
environment["DEVMEM_DB"] = str(tool / "state" / "devmem.db")
os.execvpe(sys.executable, [sys.executable, str(tool / "devmem.pyz"), *sys.argv[1:]], environment)
