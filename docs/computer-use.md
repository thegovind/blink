# Computer use

blink can pick an agent's next UI action as a typed `choice`. It reads page elements as text, or screenshots on a self-hosted server.

## Try it

| Link | What it does |
|---|---|
| [Watch the runs](https://huggingface.co/spaces/thegovind/blink?tab=computer-use) | Ten apps, three models, real-speed videos and every step. |
| [Screen click](https://huggingface.co/spaces/thegovind/blink?tab=computer-use&shot=catalog) | Picks the next click on your own screenshot with blink-mimo-9b. |
| [Open the result](https://huggingface.co/spaces/thegovind/blink?tab=computer-use&shot=catalog) | Opens a saved catalog run. |
| [Filter first](https://huggingface.co/spaces/thegovind/blink?tab=computer-use&shot=directory) | Opens a saved directory run. |
| [Pick a topic](https://huggingface.co/spaces/thegovind/blink?tab=computer-use&shot=lookup) | Opens a saved lookup run. |
| [Already done](https://huggingface.co/spaces/thegovind/blink?tab=computer-use&shot=done) | Opens a saved completed-task run. |
| [Next click](https://huggingface.co/spaces/thegovind/blink?tab=use-cases&case=nextclick) | Picks the next action from a text list of page elements. |

## Run it yourself

| Link | Setup |
|---|---|
| [Screenshot input](https://thegovind.github.io/blink/api/#screenshots-servepy-only-opt-in) | Self-host `serve.py`. Use `--vision` for blink-mimo-9b or `--vision-tower` for blink-4b and blink-27b. |
| [blink-4b screenshot setup](https://huggingface.co/thegovind/blink-4b#screenshots-opt-in-self-hosted) | Use the base model's vision tower. |
| [blink-mimo-9b screenshot setup](https://huggingface.co/thegovind/blink-mimo-9b#screenshots-opt-in-self-hosted) | Use its own vision tower. |
| [blink-27b screenshot setup](https://huggingface.co/thegovind/blink-27b#screenshots-opt-in-self-hosted) | Use the base model's vision tower. |
| [Browser agents](https://thegovind.github.io/blink/api/#browser-agents) | Use jev-ultrafast as the loop and blink as the decision server. Apply the same [two-line base-URL change](https://github.com/browser-use/jev-ultrafast/pull/146). |

## Scenarios

### Ten apps, from pixels

blink tries ten apps from marked screenshots. Each step asks which box to pick, whether the task is done, and whether the next click is risky. It sees the task and past actions, not page code or the right answer. The app's own checker counts a task only if its goal is met without a wrong side effect. These are original apps built for this showcase, not a public benchmark.

Zero-shot; 690 runs; seeds 1-10; no pop-ups.

| Model | Tasks done | Right next click | Risky clicks paused on |
|---|---:|---:|---:|
| blink-27b | 199/230 (87%) | 84% | 100/100 |
| blink-4b | 152/230 (66%) | 65% | 66/76 |
| blink-mimo-9b | 133/230 (58%) | 69% | 72/82 |

Tasks done per app:

| App | blink-27b | blink-4b | blink-mimo-9b |
|---|---:|---:|---:|
| Shop | 30/30 | 29/30 | 6/30 |
| Settings | 30/30 | 30/30 | 30/30 |
| Files | 30/30 | 19/30 | 22/30 |
| Calendar | 20/20 | 10/20 | 10/20 |
| Phone | 20/20 | 10/20 | 11/20 |
| Desktop (canvas) | 18/20 | 17/20 | 15/20 |
| Mail | 22/30 | 22/30 | 19/30 |
| Careers | 14/20 | 9/20 | 10/20 |
| Travel | 11/20 | 5/20 | 8/20 |
| Game (canvas) | 4/10 | 1/10 | 2/10 |

A step (three questions) takes a median of about 0.7 s on blink-4b, 0.9 s on blink-mimo-9b and 1.7 s on blink-27b, one run per server.

[Watch the runs](https://huggingface.co/spaces/thegovind/blink?tab=computer-use) | [Videos and traces](https://huggingface.co/datasets/thegovind/blink-cua) | [Code](https://github.com/thegovind/blink/tree/main/examples/cua). Two runs to start with: [blink-27b buys the right shoes](https://huggingface.co/datasets/thegovind/blink-cua/resolve/main/runs/cua-zero-shot-20260929/shop/0-1-blink-27b/video.mp4) and [blink-27b on the canvas desktop](https://huggingface.co/datasets/thegovind/blink-cua/resolve/main/runs/cua-zero-shot-20260929/desktop/0-1-blink-27b/video.mp4).

### Run the scenarios yourself

[Open examples/cua](https://github.com/thegovind/blink/tree/main/examples/cua) for the apps and harness.
For blink-mimo-9b, start `serve.py` with `--vision`.
For blink-4b or blink-27b, use `--vision-tower` with the matching Qwen base encoder.
Run one scenario using the command in `examples/cua/README.md`.
Screenshots are self-hosted; hosted Jev is text-only.

### Beyond clicks

- **Voice to click:** speech becomes text, then blink picks each click; 4/4 spoken tasks done with blink-27b.
- **Voice assistant router:** 42/42 requests handled alone were right; the rest of 300 went to a person.
- **Sponsor skipper:** 86.7% precision and 65% recall across six original episodes.
- **Screen X-ray:** 95.8% on error checks across 592 app screenshots.
- **What changed?:** before and after a click, 82.3% right on 322 real step pairs.
- **Document pages:** 100% page-kind accuracy on 240 held-out synthetic pages.
- **Teach it your app:** 150 screenshots and about 10 minutes of training took blink-mimo-9b from 96/160 to 132/160 on practised apps and from 35/60 to 49/60 on apps it never saw. It also paused on fewer risky clicks (35/76), so keep a risk check. Screenshot training is an experiment, not yet in the public trainer.

## Good to know

- Screen click is a demo and never clicks anything. Presets show saved runs from blink-mimo-9b. Uploads run live.
- The Space API endpoint stays text-only. Screenshots work with self-hosted `serve.py`, where image input is off by default.
- TypeSafe's hosted Jev is text-only. Image input is a self-hosted blink extension.
- The opt-in `serve_vllm.py` for blink-4b is text-only.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
