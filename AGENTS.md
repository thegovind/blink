# Coding in this repo

Start here before changing code.

## Layout

| Path | Use |
|---|---|
| `blink/` | Python package: engine and HTTP server |
| `space/` | Runtime `blink.py` and the Space app |
| `serve.py` | Standalone server shipped in each model repo |
| `lab/` | Training and evaluation tools |
| `site/` | Docs builder |
| `skills/blink/` | blink Agent Skill |

## Commands

```sh
make test
make lint
make site
make leak-scan
```

## Rules

- Run parity checks before changing answer math, rendering, or the default serving path.
- Keep every existing API field. Only add fields.
- Run the tests and leak scan before a pull request.
