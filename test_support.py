from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def workspace_test_directory(prefix: str):
    """Create a normal-ACL test directory inside the ignored repository tmp folder."""

    root = Path(__file__).resolve().parent / "tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
