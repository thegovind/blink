# Contributing

Thanks for helping improve blink.

- Keep changes small and testable.
- Add or update mock-mode tests for runtime and UI behavior.
- Do not add model weights, generated training data, secrets, private paths, or
  benchmark outputs that cannot be redistributed.
- Benchmark claims must include the request set, scorer, model revision, and all
  caveats needed to reproduce the number.
- Run `make test` and `make leak-scan` before opening a pull request.
