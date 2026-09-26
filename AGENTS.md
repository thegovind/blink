# Coding agents

Read this before changing the repo.

## Layout

| Path | Use |
|---|---|
| `space/` | Runtime `blink.py` and the Space app |
| `serve.py` | HTTP server |
| `lab/` | Training and evaluation tools |
| `site/` | Docs builder |
| `skills/blink/` | blink Agent Skill |

## Commands

```sh
BLINK_MOCK=1 python -m unittest discover -s space -p 'test_*.py'
BLINK_MOCK=1 python -m unittest discover -s lab/tests -p 'test_*.py'
ruff check .
make site
python scripts/leak_scan.py .
```

## Rules

- Do not change answer math, rendering, or the default serving path without parity checks.
- Keep existing API fields. Only add fields.
- Run the tests and leak scan before a pull request.
