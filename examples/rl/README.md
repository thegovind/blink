# Reinforcement learning with blink

blink never writes text, but it still makes choices, and a choice is an action. Its policy is the probability it
gives each offered option, read in one forward pass, so the log-probability of any choice is exact. RL works. The
question is when it is the right tool.

| Your feedback | Use | Why |
|---|---|---|
| What happened after each decision, learned later: the ticket was escalated, the charge was fraud | **Learn from outcomes**: a fine-tune with the log score | For one decision the log-score gradient is exact, so no sampling is needed. It is blink's usual loss with the outcome as the label. |
| One result after several decisions: an agent clicks, types and stops, then the task passes or fails | **RL in an environment**: GRPO or GSPO | The credit for the final result has to be shared among the decisions that led to it. |

## One decision: learn from outcomes

1. Log each decision: its state and question, and later what happened.
2. Write one row per decision with `"outcome"` set to the option that turned out to be right.
   [`examples/train/prepare.py`](../train/prepare.py) turns it into the label and splits the data.
3. Fine-tune with [`examples/train`](../train/README.md) (a GPU, local or on Azure AI Foundry) or
   [`examples/mlx/train_mlx.py`](../mlx/README.md) (a Mac).

The loss is −log p(outcome), the log score. Its expected value is lowest only when p equals the true
probabilities, so the model stays calibrated while it learns. When several people judge the same case, pass their
counts as `"votes"` and the target becomes the share each option got.

### Do not reward right answers

`python examples/rl/reward_demo.py` trains one softmax policy on the same stream of outcomes with three rewards.
It is pure Python and takes a few seconds:

```text
Train longer:
                          outcomes           accuracy         confidence           hit rate  calibration error
accuracy reward (GRPO)      10,000              0.730              0.717              0.505              0.212
accuracy reward (GRPO)      40,000              0.805              0.794              0.522              0.274
accuracy reward (GRPO)     160,000              0.835              0.861              0.527              0.333
log score                   10,000              0.830              0.538              0.522              0.018
log score                   40,000              0.850              0.539              0.530              0.013
log score                  160,000              0.945              0.536              0.534              0.004
```

An accuracy reward teaches certainty. With more training its confidence climbs from 0.72 to 0.86 while its top
pick still comes true about 53% of the time. The log score picks the most likely option more often (0.83 to 0.94)
and its calibration error falls to 0.004. The reason is simple: expected accuracy is linear in the probabilities,
so it is highest when all the mass sits on one option. A proper score, log or Brier, is highest at the true
probabilities.

The same thing happened to blink-4b. These research runs made 200 full-parameter updates per reward on generated
questions with known answer distributions, then read 2,000 fresh ones. None was released.

| Reward | Picks the most likely answer | Calibration error |
|---|---:|---:|
| none (blink-4b v1.0) | 83.1% | 0.023 |
| right or wrong (GRPO) | 58.1% | 0.431 |
| log score | 88.4% | 0.010 |
| Brier score | 84.4% | 0.029 |

All three runs also lost 25 to 32 of the 111 hard public JevBench items, which they had not trained on. Narrow
data with no replay makes a model forget, whatever the reward. Mix in anchor rows when you train on one task; the
[customize guide](https://thegovind.github.io/blink/customize/) shows how.

### Mistakes with different costs

Keep the probabilities honest and put the costs in the decision rule. With a calibrated p you can pick the action
with the lowest expected cost, or set a threshold: approve a refund automatically below a 2% fraud risk and send
the rest to a person. The Space's computer-use demo works this way, with fixed thresholds (done at 0.60, risky at
0.50). If you train the costs into the model instead, its numbers stop being probabilities, and every new
threshold needs a new model.

## Several decisions: RL in an environment

This is the recipe that was run for blink-mimo-9b in the ten offline apps of [`examples/cua`](../cua/README.md):

1. **An environment.** `reset` opens a seeded app with a task. `step` applies one action. The app's own `check()`
   grades the episode once, at the end. An oracle exists for controls and is never shown to the model.
2. **Controls before training.** A perfect policy should pass nearly every episode and a random one nearly none.
   Your model, sampling at temperature 1, must pass some and fail some episodes within a group. A group that all
   passes or all fails carries no learning signal.
3. **Rollouts.** Sample the choices that drive the episode (which element to click, which value to enter) from
   blink's probabilities, and record each choice's probability. Run a group of 8 episodes from the same start.
4. **Advantage.** A = (R − group mean) / (group standard deviation + 0.1). Every sampled choice in an episode gets
   that episode's A.
5. **Update a LoRA** with a clipped policy gradient. GRPO clips each choice's probability ratio at 0.2. GSPO clips
   one length-normalised ratio per episode at 0.05.
6. **Keep the gates calibrated.** "Is the task done?" and "Is this click risky?" were not sampled. They were trained
   with a class-balanced log score against the environment's own labels, so the safety gate stayed honest.
7. **Stop** when groups stop having mixed results, and pick checkpoints on seeds you did not train on.
8. **Read everything again.** The update changes weights that also answer text questions. Re-read your text and
   calibration checks, then read an untouched held-out set once.

What happened, in 230-episode reads against a same-harness v1.0 read (research runs, not released):

| | Tasks passed (220, game excluded) | Risky clicks paused | Safe clicks paused | Text calibration error (111 hard JevBench items) |
|---|---:|---:|---:|---:|
| blink-mimo-9b v1.0 | 132 | 88.0% | 30.0% | 0.136 |
| GRPO | 194 | 100% | 0.1% | 0.206 |
| GSPO | 190 | 100% | 1.6% | 0.171 |
| GRPO averaged with v1.0 | 193 | 100% | 1.8% | 0.151 |

RL made a much better computer-use agent. It also made the shared model's text probabilities sharper. Averaging
the RL weights with v1.0 kept most of the gain, but on a frozen held-out set of text tasks it was still 1.1 points
less accurate (95% interval −1.7 to −0.4), with calibration error 0.129 against 0.104. So it was not released. If
a model only drives one agent, that trade may be fine. If the same weights also answer text questions, measure
both.

## Code

- [`reward_demo.py`](reward_demo.py): the simulation above (`--help` for its options).
- Outcome learning: [`examples/train/prepare.py`](../train/prepare.py), then [`examples/train`](../train/README.md)
  or [`examples/mlx/train_mlx.py`](../mlx/train_mlx.py).
- Rollouts in the ten apps: `run_episode` in [`examples/cua/harness/agent.py`](../cua/harness/agent.py) takes a
  `policy` hook. `policy=None` is the evaluated, greedy harness. A sampling policy is a few lines:

```python
import random


class SamplingPolicy:
    """Sample the element and value choices from blink's probabilities; keep the deployed done and risky rules."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.choices = []  # (question, key, probability), for the policy gradient

    def choose(self, qkey, answer, question, greedy, context):
        probs = answer.get("probabilities")
        if qkey not in ("element", "value") or not probs:
            return greedy
        keys = list(probs)
        key = self.rng.choices(keys, weights=[probs[k] for k in keys])[0]
        self.choices.append((qkey, key, probs[key]))
        return key
```

- The trainer behind the runs above, with its separate sampling and training processes and a LoRA update that reads
  screenshots, is not published. To write your own, note that a choice's log-probability is the log-softmax of the
  offered letters' scores at the answer position: the same readout [`lab/jevlab/train.py`](../../lab/jevlab/train.py)
  uses in `forward_batch`. Most RL libraries assume a policy that writes tokens, so compute these log-probabilities
  yourself and pass them to your GRPO or GSPO loss.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
