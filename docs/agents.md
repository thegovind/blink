# Agent experience (AX)

## Give your agent the blink skill

Use the skill to send typed questions and read `noul`, `choice`, and `score` answers.

For screenshots, enable vision on a self-hosted v1.3 or later server and [send a data URI](https://thegovind.github.io/blink/api.md#screenshots-servepy-only-opt-in).

Install with `npx skills add thegovind/blink`, or copy [`skills/blink/`](https://github.com/thegovind/blink/blob/main/skills/blink/SKILL.md) into your agent's skills folder (for example `.agents/skills/blink/`).

## Docs for agents

Start with [`/llms.txt`](https://thegovind.github.io/blink/llms.txt).

Add `.md` to any docs page path for Markdown, such as [the API page](https://thegovind.github.io/blink/api.md).

Read [`AGENTS.md`](https://github.com/thegovind/blink/blob/main/AGENTS.md) before changing the repo.

## Write good questions

- Use clear instructions.
- Keep options short and distinct.
- Give blink the state that matters.
