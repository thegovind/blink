# Agent experience (AX)

## Give your agent the blink skill

The skill tells an agent when and how to use blink for typed decisions with probabilities.

Copy `skills/blink/` from the [repo](https://github.com/thegovind/blink) into the agent's skills folder, such as `~/.agents/skills/` or `.claude/skills/`.

## Docs for agents

[`/llms.txt`](https://thegovind.github.io/blink/llms.txt) is the docs index.

Every docs page has a Markdown copy at the same path plus `.md`, such as [the API page](https://thegovind.github.io/blink/api.md).

Coding agents working in the repo should read [`AGENTS.md`](https://github.com/thegovind/blink/blob/main/AGENTS.md).

## Write good questions

- Use clear instructions.
- Keep options short and distinct.
- Give blink the state that matters.
