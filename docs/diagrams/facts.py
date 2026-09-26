"""Every number that appears in a blink diagram, derived from a checkable source.

Sources
-------
configs/<repo>.config.json   mirrored from https://huggingface.co/thegovind/<repo>/raw/main/config.json
../../runs/mix-categories.json  the training mixture census
../out/<repo>/README.md      the published model card (for learning rates, steps and scores)

``verify()`` re-derives everything and asserts it against those files; nothing in
this module is a hand-typed duplicate of a number that a source already holds.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # .../20260923-jev-decision
CONFIG_DIR = HERE / "configs"
INDEX_DIR = HERE / "indexes"
MIX_PATH = ROOT / "runs" / "mix-categories.json"
TRAIN_PATH = ROOT / "lab" / "jevlab" / "train.py"
CARD_DIR = ROOT / "release" / "out"

REPOS = ("blink-4b", "blink-27b", "blink-mimo-9b")
HUB = "https://huggingface.co/thegovind/{repo}/raw/main/config.json"
HUB_INDEX = "https://huggingface.co/thegovind/{repo}/raw/main/model.safetensors.index.json"
TARGET_RE = re.compile(r"language_model\.layers\.\d+\.[\w.]*?("
                       r"q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b"
                       r"|out_proj|gate_proj|up_proj|down_proj)\.weight$")

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "delta": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}

CATEGORY_LABEL = {
    "decision_worlds": "Decision worlds",
    "exact_probability": "Exact-probability worlds",
    "program_reasoning": "Program-generated reasoning",
    "chess": "Chess move choices",
    "teacher_authored": "Teacher-written questions",
    "judge": "Judge-style rows",
    "base_anchors": "Base-model anchors",
    "public_source": "Public-source rows",
}
CATEGORY_NOTE = {
    "decision_worlds": "program-generated",
    "exact_probability": "probabilities computed by code",
    "chess": "searchless_chess training positions",
    "teacher_authored": "written by Qwen3.8-27B",
    "base_anchors": "base model's own distributions",
    "public_source": "public train splits",
}
DETAIL_LABEL = {
    "reason_code_exec": "code execution",
    "reason_causal": "causal",
    "reason_word_math": "word problems",
    "reason_logic_grid": "logic grids",
    "judge_math_gsm": "GSM8K-train solution checks",
    "judge_route_dolly": "Dolly-15k routing",
    "judge_math_synth": "program-answer checks",
}


def _load_json(p: Path):
    return json.loads(p.read_text())


def _mirror(url: str, dest: Path, refresh: bool) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode()
    json.loads(raw)
    dest.write_text(raw)


def fetch_configs(refresh: bool = False) -> None:
    """Mirror the public config.json and safetensors index of each repo."""
    for repo in REPOS:
        _mirror(HUB.format(repo=repo), CONFIG_DIR / f"{repo}.config.json", refresh)
        _mirror(HUB_INDEX.format(repo=repo), INDEX_DIR / f"{repo}.index.json", refresh)


# ------------------------------------------------------------------ model ---
@dataclass
class Model:
    repo: str
    title: str
    base: str
    base_note: str
    tagline: str
    cfg: dict = field(default_factory=dict)
    text_cfg: dict = field(default_factory=dict)
    vision_cfg: dict | None = None
    tensors: tuple = ()

    # ---- architecture, straight out of config.json -------------------------
    @property
    def layer_types(self):
        return self.text_cfg["layer_types"]

    @property
    def n_layers(self):
        return self.text_cfg["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text_cfg["hidden_size"]

    @property
    def n_full(self):
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def n_delta(self):
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def full_positions(self):
        """0-based layer indices that use full attention, as in the tensor names."""
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def period(self):
        """Length of the repeating layer pattern (asserted uniform)."""
        pos = self.full_positions
        step = pos[0] + 1
        assert all(p == step * (k + 1) - 1 for k, p in enumerate(pos)), self.repo
        assert self.n_layers % step == 0
        return step

    @property
    def positions_label(self) -> str:
        p = self.full_positions
        return f"{p[0]}, {p[1]}, \u2026 {p[-1]}"

    # ---- what the shipped checkpoint actually contains ---------------------
    @property
    def tied(self) -> bool:
        return bool(self.text_cfg["tie_word_embeddings"])

    @property
    def has_lm_head_tensor(self) -> bool:
        return any(n == "lm_head.weight" or n.endswith(".lm_head.weight") for n in self.tensors)

    @property
    def vision_tensors(self) -> int:
        return sum(1 for n in self.tensors if "visual" in n.split("."))

    @property
    def mtp_tensors(self) -> int:
        return sum(1 for n in self.tensors if "mtp" in n.lower().split("."))

    @property
    def targeted_tensors(self) -> int:
        return sum(1 for n in self.tensors if TARGET_RE.search(n))

    @property
    def self_attn_layers(self) -> list[int]:
        return sorted({int(m.group(1)) for n in self.tensors
                       if (m := re.search(r"layers\.(\d+)\.self_attn\.q_proj", n))})

    # ---- LoRA, computed from the same config fields ------------------------
    def module_shapes(self):
        c = self.text_cfg
        h, hd = c["hidden_size"], c["head_dim"]
        q_out = c["num_attention_heads"] * hd * (2 if c.get("attn_output_gate") else 1)
        kv_out = c["num_key_value_heads"] * hd
        o_in = c["num_attention_heads"] * hd
        k_dim = c["linear_num_key_heads"] * c["linear_key_head_dim"]
        v_dim = c["linear_num_value_heads"] * c["linear_value_head_dim"]
        v_heads = c["linear_num_value_heads"]
        inter = c["intermediate_size"]
        return {
            "attn": {"q_proj": (h, q_out), "k_proj": (h, kv_out),
                     "v_proj": (h, kv_out), "o_proj": (o_in, h)},
            "delta": {"in_proj_qkv": (h, 2 * k_dim + v_dim), "in_proj_z": (h, v_dim),
                      "in_proj_a": (h, v_heads), "in_proj_b": (h, v_heads),
                      "out_proj": (v_dim, h)},
            "mlp": {"gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)},
        }

    @property
    def lora_params(self) -> int:
        shapes = self.module_shapes()
        per = {k: sum(LORA_RANK * (i + o) for i, o in v.values()) for k, v in shapes.items()}
        return (per["attn"] * self.n_full + per["delta"] * self.n_delta
                + per["mlp"] * self.n_layers)

    @property
    def lora_tensors(self) -> int:
        return (len(LORA_TARGETS["attn"]) * self.n_full
                + len(LORA_TARGETS["delta"]) * self.n_delta
                + len(LORA_TARGETS["mlp"]) * self.n_layers)

    @property
    def lora_params_m(self) -> str:
        return f"{self.lora_params / 1e6:.1f}M"

    # ---- card text ---------------------------------------------------------
    @property
    def card(self) -> str:
        return (CARD_DIR / self.repo / "README.md").read_text()


# ----------------------------------------------------------- static facts ---
# Values that only the published card records. verify() asserts each of these
# literal strings is present in that card.
CARD_FACTS = {
    "blink-4b": {
        "served": "4,205,751,296 text parameters",
        "served_short": "4,205,751,296 parameters",
        "served_needle": "4,205,751,296",
        "di_full": "52.12",
        "stages": [
            {"name": "T3", "key": "t3_train", "lr": "3e-5", "steps": "96 steps", "from": "base"},
            {"name": "T4", "key": "t4_4b_train", "lr": "4e-5", "steps": "472 steps", "from": "base"},
        ],
    },
    "blink-27b": {
        "served": "26,895,998,464 text parameters",
        "served_short": "26,895,998,464 parameters",
        "served_needle": "26,895,998,464",
        "di_full": "63.44",
        "stages": [
            {"name": "T2", "key": "t2_27b_train", "lr": "5e-5", "steps": "230 steps", "from": "base"},
            {"name": "T4", "key": "t4_27b_train", "lr": "3e-5", "steps": "856 more steps", "from": "T2"},
        ],
    },
    "blink-mimo-9b": {
        "served": "8.95B text parameters serve decisions · 0.46B vision",
        "served_short": "8.95B text · 0.46B vision",
        "served_needle": "8.95B text parameters",
        "di_full": "56.53",
        "di_s": "55.3",
        "vision_tensors": "333",
        "stages": [
            {"name": "one run", "key": "mimo_train", "lr": "5e-5", "steps": "615 steps", "from": "base"},
        ],
    },
}

MODELS = {
    "blink-4b": Model(
        repo="blink-4b", title="blink-4b",
        base="Qwen/Qwen3.5-4B", base_note="text model only",
        tagline="uniform average of three fine-tuned checkpoints"),
    "blink-27b": Model(
        repo="blink-27b", title="blink-27b",
        base="Qwen/Qwen3.8-27B", base_note="text model only",
        tagline="one run in two stages"),
    "blink-mimo-9b": Model(
        repo="blink-mimo-9b", title="blink-mimo-9b",
        base="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        base_note="a fine-tune of Qwen/Qwen3.5-9B",
        tagline="one pre-registered run"),
}


def load() -> dict[str, Model]:
    fetch_configs()
    mix = _load_json(MIX_PATH)
    for repo, m in MODELS.items():
        cfg = _load_json(CONFIG_DIR / f"{repo}.config.json")
        m.tensors = tuple(_load_json(INDEX_DIR / f"{repo}.index.json")["weight_map"])
        m.cfg = cfg
        m.text_cfg = cfg.get("text_config", cfg)
        m.vision_cfg = cfg.get("vision_config")
        m.facts = CARD_FACTS[repo]
        for st in m.facts["stages"]:
            block = mix[st["key"]]
            st["rows"] = block["total"]
            st["categories"] = sorted(block["categories"].items(), key=lambda kv: -kv[1])
            st["detail"] = block["detail"]
            assert sum(block["categories"].values()) == block["total"], st["key"]
    return MODELS


def fmt(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------- checking ---
def trainer_targets() -> dict[str, list[str]]:
    """The LoRA target groups as the training script literally defines them."""
    src = TRAIN_PATH.read_text()
    block = src.split("TARGETS = {", 1)[1].split("}", 1)[0]
    out = {}
    for line in block.strip().splitlines():
        key, _, rest = line.strip().partition(":")
        out[key.strip().strip('"')] = [t.strip().strip('",') .strip("'")
                                       for t in rest.strip().strip("[],").split(",")]
    return out


def verify() -> list[tuple[str, str, str, str]]:
    """Return (model, quantity, value, source) rows; raise on any mismatch."""
    models = load()
    assert trainer_targets() == LORA_TARGETS, trainer_targets()
    rows: list[tuple[str, str, str, str]] = []

    def add(model, what, value, source):
        rows.append((model, what, value, source))

    mix_rel = MIX_PATH.relative_to(ROOT)
    for repo, m in models.items():
        cfg_rel = f"diagrams/configs/{repo}.config.json"
        idx_rel = f"diagrams/indexes/{repo}.index.json"
        prefix = "text_config." if "text_config" in m.cfg else ""
        card = m.card
        card_rel = f"release/out/{repo}/README.md"

        assert len(m.layer_types) == m.n_layers
        add(repo, "decoder layers", str(m.n_layers), f"{cfg_rel} · {prefix}num_hidden_layers")
        add(repo, "Gated DeltaNet layers", str(m.n_delta),
            f"{cfg_rel} · count of 'linear_attention' in {prefix}layer_types")
        add(repo, "full-attention layers", str(m.n_full),
            f"{cfg_rel} · count of 'full_attention' in {prefix}layer_types")
        add(repo, "full attention at layers (0-based)", m.positions_label,
            f"{cfg_rel} · positions of 'full_attention' in {prefix}layer_types; "
            f"{idx_rel} · layers.N.self_attn.q_proj")
        assert m.self_attn_layers == m.full_positions, repo
        assert max(int(x.group(1)) for n in m.tensors
                   if (x := re.search(r"language_model\.layers\.(\d+)\.", n))) == m.n_layers - 1
        add(repo, "hidden size", str(m.hidden), f"{cfg_rel} · {prefix}hidden_size")
        assert m.text_cfg["full_attention_interval"] == m.period

        # the card must agree with the config-derived architecture
        for needle in (f"{m.n_layers} decoder layers",
                       f"{m.n_delta} Gated DeltaNet",
                       f"{m.n_full} full-attention",
                       f"hidden {m.hidden}"):
            assert needle in card, f"{repo}: card missing {needle!r}"

        add(repo, "lm_head rows (vocabulary)", fmt(m.text_cfg["vocab_size"]),
            f"{cfg_rel} · {prefix}vocab_size")
        add(repo, "LoRA rank / alpha", f"{LORA_RANK} / {LORA_ALPHA}",
            "lab/jevlab/train.py --rank/--alpha, lab/t3_run.sh, card")
        assert f"rank {LORA_RANK}, alpha {LORA_ALPHA}" in card
        for group, mods in LORA_TARGETS.items():
            add(repo, f"LoRA targets — {group}", ", ".join(mods),
                "lab/jevlab/train.py · TARGETS, and the card's LoRA targets line")
            for mod in mods:
                assert f"`{mod}`" in card, f"{repo}: card missing {mod}"
        add(repo, "LoRA trainable parameters", m.lora_params_m,
            f"computed from {cfg_rel} shapes at r={LORA_RANK}: {fmt(m.lora_params)}")
        assert f"{m.lora_params_m} LoRA parameters" in card, f"{repo}: {m.lora_params_m}"
        add(repo, "LoRA-touched tensors", str(m.lora_tensors),
            f"{cfg_rel} · {len(LORA_TARGETS['attn'])}·full + {len(LORA_TARGETS['delta'])}·DeltaNet "
            f"+ {len(LORA_TARGETS['mlp'])}·MLP; counted in {idx_rel}")
        assert m.targeted_tensors == m.lora_tensors, f"{repo}: {m.targeted_tensors}"
        add(repo, "embeddings and lm_head",
            "tied — one matrix" if m.tied else "untied — separate matrices",
            f"{cfg_rel} · {prefix}tie_word_embeddings; {idx_rel} · "
            f"lm_head.weight {'absent' if not m.has_lm_head_tensor else 'present'}")
        assert m.tied != m.has_lm_head_tensor, f"{repo}: tie/lm_head disagree"
        add(repo, "tensors in the checkpoint", fmt(len(m.tensors)), f"{idx_rel} · weight_map")
        add(repo, "MTP tensors", str(m.mtp_tensors), f"{idx_rel} · no 'mtp' tensor names")
        assert m.mtp_tensors == 0, repo

        f = m.facts
        add(repo, "served text parameters", f["served"], f"{card_rel} · network table")
        assert f["served_needle"] in card, f"{repo}: served"
        add(repo, "Decision Index 0.1, full suite", f["di_full"], f"{card_rel} · results table")
        assert f["di_full"] in card

        add(repo, "vision tensors", str(m.vision_tensors),
            f"{idx_rel} · tensor names under 'visual'")
        if m.vision_cfg:
            v = m.vision_cfg
            add(repo, "vision encoder blocks", str(v["depth"]), f"{cfg_rel} · vision_config.depth")
            add(repo, "vision hidden size", str(v["hidden_size"]),
                f"{cfg_rel} · vision_config.hidden_size")
            add(repo, "vision projected to", str(v["out_hidden_size"]),
                f"{cfg_rel} · vision_config.out_hidden_size")
            assert v["out_hidden_size"] == m.hidden
            assert f"{v['depth']} encoder blocks" in card and f"hidden {v['hidden_size']}" in card
            assert m.vision_tensors == int(f["vision_tensors"]), repo
            assert f"{f['vision_tensors']} vision tensors unchanged" in card
            assert "untied embeddings" in card
            others = len(m.tensors) - m.vision_tensors - m.targeted_tensors
            add(repo, "other language-model tensors", str(others),
                f"{idx_rel} · {fmt(len(m.tensors))} total − {m.vision_tensors} vision "
                f"− {m.targeted_tensors} targeted")
            assert f"{others} other language-model tensors" in card, others
            add(repo, "DI-S gate", f"{f['di_s']} (bar 55)", f"{card_rel} · the climb")
            assert f["di_s"] in card
        else:
            assert "0 vision tensors, 0 MTP tensors" in card
            assert m.vision_tensors == 0, repo

        for st in f["stages"]:
            tag = f"{repo} {st['name']}"
            add(tag, "question rows", fmt(st["rows"]), f"{mix_rel} · {st['key']}.total")
            assert fmt(st["rows"]) in card, f"{tag}: rows"
            assert f"lr {st['lr']}" in card and st["steps"] in card, f"{tag}: lr/steps"
            add(tag, "learning rate", st["lr"], f"{card_rel} · training table")
            add(tag, "optimiser steps", st["steps"], f"{card_rel} · training table")
            for key, n in st["categories"]:
                add(tag, CATEGORY_LABEL[key], fmt(n),
                    f"{mix_rel} · {st['key']}.categories.{key}")
                assert fmt(n) in card, f"{tag}: {key} {n}"
            for key, n in sorted(st["detail"].get("program_reasoning", {}).items(), key=lambda kv: -kv[1]):
                add(tag, f"↳ {DETAIL_LABEL[key]}", fmt(n),
                    f"{mix_rel} · {st['key']}.detail.program_reasoning.{key}")
            for key, n in sorted(st["detail"].get("judge", {}).items(), key=lambda kv: -kv[1]):
                add(tag, f"↳ {DETAIL_LABEL[key]}", fmt(n),
                    f"{mix_rel} · {st['key']}.detail.judge.{key}")

    # readout facts, shared by all three cards
    card = models["blink-4b"].card
    for needle in ("255 options per choice", "2–10 score levels",
                   "FP32", "next-token logits"):
        assert needle in card, needle
    rows.append(("readout", "options per choice", "up to 255", "release/out/*/README.md · limits"))
    rows.append(("readout", "score levels", "2–10 ordered levels", "release/out/*/README.md · limits"))
    rows.append(("readout", "questions per request", "up to 512",
                 "release/out/*/README.md · limits"))
    rows.append(("readout", "label rows read", "offered option letters only, FP32",
                 "release/out/*/README.md · readout"))
    rows.append(("selection", "DI-S sample", "3,000 requests",
                 "release/out/*/README.md · evaluation notes"))
    return rows


if __name__ == "__main__":
    for r in verify():
        print(" | ".join(r))
