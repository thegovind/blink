#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
HOME_RE = "/" + "home/" + r"[^\s'\"]+"
USERS_RE = "/" + "Users/" + r"[^\s'\"]+"
WIN_USERS_RE = "C:" + r"\\Users\\[^\s'\"]+"
GENERIC_PATTERNS = [
    re.compile(HOME_RE),
    re.compile(USERS_RE),
    re.compile(WIN_USERS_RE, re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
]
ALLOW_EMAIL = re.compile(r"\d+\+thegovind@users\.noreply\.github\.com", re.I)
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = (".lock",)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    hits = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts) or path.name.endswith(SKIP_SUFFIXES):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        size = path.stat().st_size
        if size > 5 * 1024 * 1024:
            hits.append(f"{rel}: file is over 5 MB ({size} bytes)")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in TOKEN_PATTERNS + GENERIC_PATTERNS:
            for m in pat.finditer(text):
                if "@" in m.group(0) and ALLOW_EMAIL.fullmatch(m.group(0)):
                    continue
                val = m.group(0)
                if val in {"127.0.0.1", "0.0.0.0"}:
                    continue
                hits.append(f"{rel}: {pat.pattern}: {val[:80]}")
    if hits:
        print("\n".join(hits))
        return 1
    print("leak scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
