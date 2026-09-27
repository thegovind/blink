# Computer use

blink can pick an agent's next UI action as a typed `choice`. It reads page elements as text, or screenshots on a self-hosted server.

## Try it

| Link | What it does |
|---|---|
| [Screen click](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=screen) | Picks the next click on a screenshot with blink-mimo-9b. |
| [Open the result](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=screen&shot=catalog) | Opens a saved catalog run. |
| [Filter first](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=screen&shot=directory) | Opens a saved directory run. |
| [Pick a topic](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=screen&shot=lookup) | Opens a saved lookup run. |
| [Already done](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=screen&shot=done) | Opens a saved completed-task run. |
| [Next click](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=nextclick) | Picks the next action from a text list of page elements. |

## Run it yourself

| Link | Setup |
|---|---|
| [Screenshot input](https://thegovind.github.io/blink/api/#screenshots-servepy-only-opt-in) | Self-host `serve.py`. Use `--vision` for blink-mimo-9b or `--vision-tower` for blink-4b and blink-27b. |
| [blink-4b screenshot setup](https://huggingface.co/thegovind/blink-4b#screenshots-opt-in-self-hosted) | Use the base model's vision tower. |
| [blink-mimo-9b screenshot setup](https://huggingface.co/thegovind/blink-mimo-9b#screenshots-opt-in-self-hosted) | Use its own vision tower. |
| [blink-27b screenshot setup](https://huggingface.co/thegovind/blink-27b#screenshots-opt-in-self-hosted) | Use the base model's vision tower. |
| [Browser agents](https://thegovind.github.io/blink/api/#browser-agents) | Use jev-ultrafast as the loop and blink as the decision server. Apply the same [two-line base-URL change](https://github.com/browser-use/jev-ultrafast/pull/146). |

## Good to know

- Screen click is a demo and never clicks anything. Presets show saved runs from blink-mimo-9b. Uploads run live.
- The Space API endpoint stays text-only. Screenshots work with self-hosted `serve.py`, where image input is off by default.
- TypeSafe's hosted Jev is text-only. Image input is a self-hosted blink extension.
- The opt-in `serve_vllm.py` for blink-4b is text-only.
