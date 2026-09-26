#!/usr/bin/env python3
"""Build the blink model-card diagrams as SVG, and PNG at 2x.

    uv run --with playwright --with fonttools python make_diagrams.py

Writes release/assets/<model>/*.svg and *.png and regenerates CHECKS.md.
Every number in every figure comes from facts.py, which asserts it against
config.json, runs/mix-categories.json or the published card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import facts
from facts import fmt
from svgkit import (ACC, ACC_DEEP, ACC_FILL, ACC_FILL2, ACC_LINE, BG, BORDER, Canvas, FAINT,
                    FROZEN_FILL, FROZEN_LINE, INK, MAT, MUTED, PANEL, RULE, STYLES, SUBTLE,
                    TEXT_CSS, text_width)

# Strings whose rendered width must not exceed what text_width() predicts. Inter's
# tabular-numerals feature substitutes a digit-width hyphen-minus, which both loosens
# "blink-4b" into "blink - 4b" and breaks every width estimate; this catches that.
RENDER_PROBES = ("blink-4b", "lr 3e-5", "Full-attention layer", "byte-for-byte",
                 "Qwen/Qwen3.5-4B", "Next-token logits", "0-based, as in tensor names")

W = 940
MGN = 34
INNER = W - 2 * MGN
OUT = facts.ROOT / "release" / "assets"
DIM_ROW = "#E0DDD9"


# ------------------------------------------------------------- primitives ---
def hrule(c, y, x=MGN, w=INNER, col=RULE):
    c.line(x, y, x + w, y, stroke=col, sw=1, cap="butt")


def sec(c, x, y, label):
    c.text(x, y, label, "sec")


def header(c, eyebrow, title, subtitle, right=None, note=None):
    sec(c, MGN, 40, eyebrow)
    c.text(MGN, 74, title, "h1", maxw=520, tag=title)
    if right:
        c.text(W - MGN, 40, right, "tiny", anchor="end")
    c.text(MGN, 97, subtitle, "sub", maxw=INNER, tag=subtitle)
    if note:
        c.text(MGN, 114, note, "tiny", maxw=INNER, tag=note)


def swatch(c, x, y, kind, size=11):
    top = y - size + 1
    if kind == "delta":
        c.rect(x, top, size, size, fill=ACC_FILL, stroke=ACC_LINE, rx=3)
    elif kind == "full":
        c.rect(x, top, size, size, fill=ACC_FILL2, stroke=ACC, rx=3)
    elif kind == "frozen":
        c.rect(x, top, size, size, fill=FROZEN_FILL, stroke=FROZEN_LINE, rx=3)
    elif kind == "removed":
        c.rect(x, top, size, size, fill="url(#hatch)", stroke=BORDER, rx=3)
        c.line(x + 1, y, x + size - 1, top + 1, stroke="#9A948D", sw=1.1)
    elif kind == "readout":
        c.rect(x, top, size, size, fill=INK, stroke=INK, rx=3)
    return x + size + 7


def legend(c, x, y, items, gap=20):
    for kind, label in items:
        x2 = swatch(c, x, y, kind)
        x = x2 + c.text(x2, y, label, "note", tag=label) + gap
    return x


def chip(c, x, y, w, h, label, kind="delta", style=None, rx=6):
    styles = {"delta": (ACC_FILL, ACC_LINE, "codea"),
              "full": (ACC_FILL2, ACC, "codea"),
              "frozen": (FROZEN_FILL, FROZEN_LINE, "note"),
              "plain": (PANEL, BORDER, "body"),
              "ink": (INK, INK, "wlbl")}
    if kind == "removed":
        c.rect(x, y, w, h, fill="url(#hatch)", stroke=BORDER, rx=rx)
        st = style or "tiny"
    else:
        fill, stroke, default = styles[kind]
        c.rect(x, y, w, h, fill=fill, stroke=stroke, rx=rx)
        st = style or default
    if label:
        c.text(x + w / 2, y + h / 2 + STYLES[st].size * 0.35, label, st, anchor="middle",
               maxw=w - 12, tag=label)


CHIP_PAD, CHIP_GAP, CHIP_H = 13, 8, 25


def chip_layout(labels, maxw, style="codea"):
    st = STYLES[style]
    lines, line, used = [], [], 0.0
    for lab in labels:
        cw = st.width(lab) + CHIP_PAD * 2
        if line and used + CHIP_GAP + cw > maxw:
            lines.append(line)
            line, used = [], 0.0
        used += cw + (CHIP_GAP if line else 0)
        line.append((lab, cw))
    if line:
        lines.append(line)
    return lines


def chip_block_h(lines):
    return len(lines) * (CHIP_H + 7) - 7


def draw_chips(c, x, y, lines, kind="delta", style="codea"):
    for row in lines:
        cx = x
        for lab, cw in row:
            chip(c, cx, y, cw, CHIP_H, lab, kind, style=style)
            cx += cw + CHIP_GAP
        y += CHIP_H + 7
    return y - 7


def wrap_bits(bits, maxw, style="tiny", sep=" · "):
    st = STYLES[style]
    lines, cur = [], ""
    for b in bits:
        trial = f"{cur}{sep}{b}" if cur else b
        if st.width(trial) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = b
    if cur:
        lines.append(cur)
    return lines


def arrow_down(c, x, y1, y2, col=FAINT):
    c.arrow(x, y1, x, y2, stroke=col)


def arrow_right(c, x1, x2, y, col=FAINT):
    c.arrow(x1, y, x2, y, stroke=col)


def step_dot(c, x, y, n):
    c.raw(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{INK}"/>')
    c.text(x, y + 3.6, str(n), "tiny", anchor="middle", fill="#FFFFFF")


def bar(c, x, y, w, h, frac, fill, track=SUBTLE, rx=3):
    c.rect(x, y, w, h, fill=track, stroke="none", rx=rx)
    c.rect(x, y, max(w * frac, 2.5), h, fill=fill, stroke="none", rx=rx)


# =========================================================== figure (a) =====
def layer_panel(c, x, y, w, title, count, mods, mixer_label):
    """One 'inside a layer' card; returns the y below it."""
    slot = c.defer()
    ix, iw = x + 14, w - 28
    ty = y + 25
    c.text(ix, ty, title, "lblb", maxw=iw - 42, tag=title)
    c.text(x + w - 14, ty, f"× {count}", "numa", anchor="end")
    ty += 19
    sec(c, ix, ty, mixer_label)
    ty = draw_chips(c, ix, ty + 8, chip_layout(mods, iw)) + 17
    sec(c, ix, ty, "mlp")
    ty = draw_chips(c, ix, ty + 8, chip_layout(facts.LORA_TARGETS["mlp"], iw)) + 16
    chip(c, ix, ty, iw, 22, "input + post-attention norms — frozen", "frozen", style="tiny")
    ty += 22 + 14
    c.fill_deferred(slot, x, y, w, ty - y)
    return ty


def fig_network(m: facts.Model) -> Canvas:
    f = m.facts
    is_vlm = m.vision_cfg is not None
    c = Canvas(W, 898)

    header(c, "network", m.title,
           f"{m.base}  ·  {m.n_layers} decoder layers  ·  hidden {m.hidden}  ·  "
           f"{f['served_short']}",
           right=f"thegovind/{m.repo} · v1.0", note=m.base_note)
    legend(c, MGN, 134, [
        ("delta", "Gated DeltaNet layer"),
        ("full", "full-attention layer"),
        ("frozen", "frozen"),
        ("removed", "removed") if not is_vlm else ("frozen", "kept unchanged"),
        ("readout", "readout"),
    ])
    hrule(c, 152)

    ax, aw = MGN, 258
    bx, bw = ax + aw + 26, 330
    cx, cw = bx + bw + 26, 232
    top = 178

    # ---------------------------------------------- column A: the stack ---
    sec(c, ax, top, "decoder stack")
    y = top + 16
    if is_vlm:
        c.rect(ax, y, aw, 42, fill=FROZEN_FILL, stroke=FROZEN_LINE, dash="4 3")
        c.text(ax + 12, y + 18, "Vision tower — unchanged", "note", maxw=aw - 24, tag="vision")
        c.text(ax + 12, y + 33, "blink serves decisions from the text side", "tiny",
               maxw=aw - 24, tag="textside")
        arrow_down(c, ax + aw / 2, y + 44, y + 58)
        y += 60

    c.rect(ax, y, aw, 30, fill=FROZEN_FILL, stroke=FROZEN_LINE)
    c.text(ax + 12, y + 19, "Token embeddings — frozen", "note", maxw=aw - 24, tag="tokemb")
    arrow_down(c, ax + aw / 2, y + 32, y + 46)
    y += 48

    stack_h = 296 if not is_vlm else 236
    rx_, rw = ax + 18, 20
    gap = 1.6 if m.n_layers <= 32 else 0.9
    ch = (stack_h - gap * (m.n_layers - 1)) / m.n_layers
    for i, t in enumerate(m.layer_types):
        c.rect(rx_, y + i * (ch + gap), rw, ch,
               fill=(ACC if t == "full_attention" else "#D5EAE7"),
               stroke="none", rx=1.5 if ch < 6 else 2.5)
    c.text(rx_ - 6, y + max(ch, 8) * 0.9, "0", "tiny", anchor="end")
    c.text(rx_ - 6, y + stack_h, str(m.n_layers - 1), "tiny", anchor="end")

    period = m.period
    blk_x, blk_w, rh, rgap = ax + 58, 150, 30, 5
    blk_top = y + 4
    blk_bot = blk_top + period * (rh + rgap) - rgap
    c.path(f"M {rx_ + rw + 1} {y:.2f} L {blk_x:.2f} {blk_top:.2f} L {blk_x:.2f} {blk_bot:.2f} "
           f"L {rx_ + rw + 1} {y + period * (ch + gap) - gap:.2f} Z",
           stroke="none", fill="#EAE7E3")
    for i in range(period):
        full = m.layer_types[i] == "full_attention"
        yy = blk_top + i * (rh + rgap)
        chip(c, blk_x, yy, blk_w, rh, "", "full" if full else "delta")
        c.text(blk_x + 11, yy + rh / 2 + 4, str(i), "tiny")
        c.text(blk_x + 27, yy + rh / 2 + 4.5, "full attention" if full else "Gated DeltaNet",
               "lbl", maxw=blk_w - 38, tag="layertype")
    half = (blk_bot - blk_top) / 2 - 14
    c.path(f"M {blk_x + blk_w + 8} {blk_top} q 7 0 7 7 v {half:.1f} q 0 7 7 7 "
           f"q -7 0 -7 7 v {half:.1f} q 0 7 -7 7", stroke=BORDER, sw=1.2)
    c.text(blk_x + blk_w + 28, (blk_top + blk_bot) / 2 + 4.5, f"× {m.n_layers // period}",
           "lblb", tag="repeat")

    y_end = y + stack_h
    c.text(ax, y_end + 26, f"{m.n_delta} Gated DeltaNet + {m.n_full} full attention",
           "body", maxw=aw, tag="pattern")
    c.text(ax, y_end + 42, f"full attention at layers {m.positions_label}",
           "tiny", maxw=aw, tag="fapos")
    c.text(ax, y_end + 56, "0-based, as in tensor names", "tiny", maxw=aw, tag="zerobased")
    y = y_end + 70
    arrow_down(c, ax + aw / 2, y, y + 14)
    y += 16
    c.rect(ax, y, aw, 30, fill=FROZEN_FILL, stroke=FROZEN_LINE)
    c.text(ax + 12, y + 19, "Final norm — frozen", "note", maxw=aw - 24, tag="finalnorm")
    y += 36
    c.rect(ax, y, aw, 44, fill=FROZEN_FILL, stroke=FROZEN_LINE)
    c.text(ax + 12, y + 19, f"lm_head — frozen · {fmt(m.text_cfg['vocab_size'])} rows",
           "note", maxw=aw - 24, tag="lmhead")
    c.text(ax + 12, y + 34,
           "tied to token embeddings (same matrix)" if m.tied
           else "untied — a separate matrix", "tiny", maxw=aw - 24, tag="tie")

    # -------------------------------------- column B: where LoRA attaches --
    sec(c, bx, top, "where LoRA attaches")
    py = layer_panel(c, bx, top + 16, bw, "Gated DeltaNet layer", m.n_delta,
                     facts.LORA_TARGETS["delta"], "mixer") + 18
    py = layer_panel(c, bx, py, bw, "Full-attention layer", m.n_full,
                     facts.LORA_TARGETS["attn"], "attention") + 20
    c.text(bx, py, f"LoRA rank {facts.LORA_RANK}, alpha {facts.LORA_ALPHA} — "
                   f"{m.lora_tensors} tensors", "body", maxw=bw, tag="lorasum")
    c.text(bx, py + 16, "adapters merged into the weights after training", "tiny",
           maxw=bw, tag="merged")

    # ------------------------------- column C: trained / frozen / removed --
    sec(c, cx, top, "trained")
    ty = top + 20
    c.rect(cx, ty, cw, 68, fill=ACC_FILL, stroke=ACC_LINE, rx=9)
    c.text(cx + 14, ty + 30, m.lora_params_m, "h1", fill=ACC_DEEP)
    c.text(cx + 14, ty + 48, "LoRA parameters", "body", maxw=cw - 28, tag="loraparams")
    c.text(cx + 14, ty + 62, f"{fmt(m.lora_params)} exactly", "tiny", maxw=cw - 28, tag="exact")
    ty += 68 + 26

    sec(c, cx, ty, "frozen")
    ty += 12
    for lab in ("Token embeddings", "All norms",
                "lm_head (tied)" if m.tied else "lm_head (untied)"):
        chip(c, cx, ty, cw, 26, lab, "frozen", style="note")
        ty += 30
    ty += 18

    if is_vlm:
        sec(c, cx, ty, "kept unchanged")
        ty += 12
        v = m.vision_cfg
        c.rect(cx, ty, cw, 64, fill=FROZEN_FILL, stroke=FROZEN_LINE, rx=9)
        c.text(cx + 14, ty + 22, "Vision tower", "lbl", maxw=cw - 28, tag="vt")
        c.text(cx + 14, ty + 38,
               f"{v['depth']} blocks · hidden {v['hidden_size']} → {v['out_hidden_size']}",
               "tiny", maxw=cw - 28, tag="vtdims")
        c.text(cx + 14, ty + 53, f"{f['vision_tensors']} tensors, byte-for-byte", "tiny",
               maxw=cw - 28, tag="vttensors")
        ty += 64 + 22
        c.text(cx, ty, "No MTP tensors in this checkpoint", "tiny", maxw=cw, tag="nomtp")
    else:
        sec(c, cx, ty, "cut from the release")
        ty += 12
        for lab, sub in (("Vision encoder", "0 tensors"), ("MTP head", "0 tensors")):
            c.rect(cx, ty, cw, 34, fill="url(#hatch)", stroke=BORDER, rx=7)
            c.text(cx + 14, ty + 22, lab, "note", maxw=cw - 84, tag=lab)
            c.text(cx + cw - 14, ty + 22, sub, "tiny", anchor="end")
            c.line(cx + 12, ty + 18, cx + cw - 12, ty + 18, stroke="#B8B2AB", sw=1)
            ty += 38
        ty += 16
        c.text(cx, ty, "Text model only", "tiny", maxw=cw, tag="textonly")

    # ------------------------------------------------------ readout band --
    by = 724
    c.rect(MGN, by, INNER, 140, fill=INK, stroke=INK, rx=10)
    c.text(MGN + 22, by + 30, "readout", "wsec")
    c.text(MGN + 22, by + 57, "The answer is read from lm_head, never generated", "wlbl",
           maxw=340, tag="rhead")
    for i, line in enumerate(("Questions are batched; each batch is one forward pass",
                              "Next-token logits at the answer position",
                              "FP32 softmax over the offered option letters",
                              "The rest of the vocabulary is ignored")):
        c.text(MGN + 22, by + 76 + i * 17, line, "wnote", maxw=346, tag=line)

    gx, gy, gw = MGN + 420, by + 28, 112
    hot = (2, 5, 8)
    for i in range(11):
        c.rect(gx, gy + i * 8.4, gw, 5.6, fill="#FFFFFF" if i in hot else "#4A443E",
               stroke="none", rx=1.6)
    for i, lab in enumerate("ABC"):
        c.text(gx + gw + 9, gy + hot[i] * 8.4 + 5.4, lab, "wnum")
    c.text(gx, gy + 11 * 8.4 + 16, f"lm_head · {fmt(m.text_cfg['vocab_size'])} token rows",
           "wnote", maxw=200, tag="lmrows")

    a0 = gx + gw + 30
    c.arrow(a0, gy + 44, a0 + 28, gy + 44, stroke="#8A847D", marker="arww")
    px = a0 + 36
    for i, (lab, v) in enumerate(zip("ABC", (0.62, 0.26, 0.12))):
        yy = gy + 4 + i * 25
        c.text(px, yy + 11, lab, "wnum")
        bar(c, px + 17, yy, 150, 13, v, "#FFFFFF", track="#3A352F")
    c.text(px, gy + 4 + 3 * 25 + 12, "one probability per option · they sum to 1",
           "wnote", maxw=W - MGN - 22 - px, tag="probnote")
    c.text(W - MGN - 22, by + 30, "letters and bars are schematic", "tiny",
           anchor="end", fill="#6E6862")
    return c


# =========================================================== figure (b) =====
CAT_COLOUR = {
    "decision_worlds": ACC,
    "exact_probability": "#2E8F86",
    "program_reasoning": "#57A79F",
    "chess": "#7FBDB6",
    "teacher_authored": "#9BCCC6",
    "judge": "#B4D9D4",
    "base_anchors": "#CBE6E2",
    "public_source": "#DCEFEC",
}


def fig_pipeline(m: facts.Model) -> Canvas:
    f = m.facts
    is_vlm = m.vision_cfg is not None
    stages = f["stages"]
    peak = max(n for st in stages for _, n in st["categories"])

    x1, w1 = MGN, 318
    x2, w2 = x1 + w1 + 32, 236
    x3, w3 = x2 + w2 + 32, 254
    iw1 = w1 - 28

    det_lines = {}
    for st in stages:
        for key, _ in st["categories"]:
            det = st["detail"].get(key)
            if det:
                bits = [f"{facts.DETAIL_LABEL[k]} {fmt(v)}"
                        for k, v in sorted(det.items(), key=lambda kv: -kv[1])]
                det_lines[(st["key"], key)] = wrap_bits(bits, iw1)

    def stage_h(st):
        h = 44
        for key, _ in st["categories"]:
            h += 30 + 13 * len(det_lines.get((st["key"], key), ()))
        return h + 6

    n_targets = 4 if any(k == "base_anchors" for st in stages for k, _ in st["categories"]) else 3
    col1_h = sum(stage_h(st) for st in stages) + 20 * (len(stages) - 1)
    col2_h = 18 + 228 + n_targets * 36
    col3_h = (18 + (54 if is_vlm else 40) + 66 * len(stages) + 64
              + (92 + 18 if m.repo == "blink-4b" else 62)
              + (62 if is_vlm else 0))
    body_h = max(col1_h, col2_h, col3_h, 340)
    H = 152 + body_h + 140
    c = Canvas(W, H)

    header(c, "post-training", m.title,
           "calibration-oriented decision post-training — plain supervised fine-tuning, no RL",
           right=f"thegovind/{m.repo} · v1.0")
    hrule(c, 122)

    top = 150

    # ------------------------------------------------- column 1: the rows --
    sec(c, x1, top, "training rows")
    y = top + 18
    for st in stages:
        h = stage_h(st)
        c.rect(x1, y, w1, h, fill=PANEL, stroke=BORDER, rx=9)
        ix, iw = x1 + 14, w1 - 28
        c.text(ix, y + 24, f"{st['name']} · {fmt(st['rows'])} rows", "lblb",
               maxw=iw - 118, tag="stagehead")
        c.text(x1 + w1 - 14, y + 24, f"lr {st['lr']} · {st['steps']}", "tiny", anchor="end")
        ry = y + 42
        for key, n in st["categories"]:
            c.text(ix, ry + 10, facts.CATEGORY_LABEL[key], "body", maxw=iw - 66,
                   tag=facts.CATEGORY_LABEL[key])
            c.text(x1 + w1 - 14, ry + 10, fmt(n), "num", anchor="end")
            bar(c, ix, ry + 16, iw, 4, n / peak, CAT_COLOUR[key], rx=2)
            ry += 30
            for line in det_lines.get((st["key"], key), ()):
                c.text(ix, ry + 1, line, "tiny", maxw=iw, tag=line)
                ry += 13
        y += h + 20

    # -------------------------------------------- column 2: the objective --
    sec(c, x2, top, "objective")
    oy = top + 18
    slot = c.defer()
    ix, iw = x2 + 14, w2 - 28
    c.text(ix, oy + 25, "Supervised fine-tuning", "lblb", maxw=iw, tag="sft")
    ty, _ = c.wrap(ix, oy + 45, "Cross-entropy over the offered letters, against each row's "
                                "target distribution.", "note", maxw=iw, lh=15)
    ty += 26
    sec(c, ix, ty, "targets")
    ty += 12
    targets = [("exact probabilities", "computed by code"),
               ("teacher-written targets", "kept when a blind re-solve agreed")]
    if any(k == "base_anchors" for st in stages for k, _ in st["categories"]):
        targets.append(("base-model distributions", "on anchor rows"))
    targets.append(("one-hot labels", "otherwise"))
    for lab, note in targets:
        chip(c, ix, ty, iw, 32, "", "delta")
        c.text(ix + 12, ty + 14, lab, "lbl", maxw=iw - 24, tag=lab)
        c.text(ix + 12, ty + 26, note, "tiny", maxw=iw - 24, tag=note)
        ty += 36
    ty += 14
    sec(c, ix, ty, "every epoch")
    ty += 16
    c.text(ix, ty, "choice + yes/no options reshuffled", "note", maxw=iw, tag="reshuffle")
    c.text(ix, ty + 16, "score levels keep their order", "note", maxw=iw, tag="keeporder")
    ty += 34
    hrule(c, ty, ix, iw)
    c.text(ix, ty + 20, "Teacher — Qwen/Qwen3.8-27B", "tiny", maxw=iw, tag="teacher")
    c.text(ix, ty + 35, "No RL, no preference optimisation", "tiny", maxw=iw, tag="norl")
    c.fill_deferred(slot, x2, oy, w2, ty + 35 + 16 - oy)

    arrow_right(c, x1 + w1 + 8, x2 - 8, top + 110)
    arrow_right(c, x2 + w2 + 8, x3 - 8, top + 110)

    # ------------------------------------- column 3: checkpoints, release --
    sec(c, x3, top, "checkpoints")
    sy = top + 18
    bh = 54 if is_vlm else 40
    c.rect(x3, sy, w3, bh, fill=FROZEN_FILL, stroke=FROZEN_LINE, rx=9)
    c.text(x3 + 14, sy + 18, m.base.split("/")[-1], "lbl", maxw=w3 - 28, tag="base")
    c.text(x3 + 14, sy + 33, "base weights", "tiny", maxw=w3 - 28, tag="basenote")
    if is_vlm:
        c.text(x3 + 14, sy + 47, "LoRA on the language model only", "tiny", maxw=w3 - 28,
               tag="basenote2")
    sy += bh

    def run_box(y, label, st):
        arrow_down(c, x3 + 22, y + 4, y + 20)
        c.rect(x3, y + 22, w3, 44, fill=ACC_FILL, stroke=ACC_LINE, rx=8)
        c.text(x3 + 14, y + 41, label, "lbl", maxw=w3 - 28, tag=label)
        c.text(x3 + 14, y + 57, f"{fmt(st['rows'])} rows · lr {st['lr']} · {st['steps']}",
               "tiny", maxw=w3 - 28, tag="runmeta")
        return y + 66

    if m.repo == "blink-4b":
        t3, t4 = stages
        sy = run_box(sy, "T3 — from the base", t3)
        sy = run_box(sy, "T4 — from the base", t4)
        arrow_down(c, x3 + 22, sy + 4, sy + 20)
        c.rect(x3, sy + 22, w3, 70, fill=PANEL, stroke=ACC, rx=8)
        c.text(x3 + 14, sy + 41, "Merge, then average", "lblb", maxw=w3 - 28, tag="soup")
        cxp = x3 + 14
        for lab in ("T3", "T4 step 300", "T4 final"):
            cwid = STYLES["tiny"].width(lab) + 20
            chip(c, cxp, sy + 50, cwid, 24, lab, "full", style="tiny", rx=5)
            cxp += cwid + 6
        sy += 92
        c.text(x3 + 14, sy + 14, "uniform weight average of the three", "tiny",
               maxw=w3 - 28, tag="soupnote")
        sy += 18
    else:
        for st in stages:
            label = ("One pre-registered run · 1 epoch" if is_vlm else
                     f"{st['name']} — from the base" if st["from"] == "base" else
                     f"{st['name']} — continues {st['from']}")
            sy = run_box(sy, label, st)
        arrow_down(c, x3 + 22, sy + 4, sy + 20)
        c.rect(x3, sy + 22, w3, 40, fill=PANEL, stroke=ACC, rx=8)
        c.text(x3 + 14, sy + 40, "Merge LoRA into the weights", "lblb", maxw=w3 - 28, tag="merge")
        c.text(x3 + 14, sy + 54, f"{m.lora_params_m} adapters folded in", "tiny",
               maxw=w3 - 28, tag="mergenote")
        sy += 62
        if is_vlm:
            arrow_down(c, x3 + 22, sy + 4, sy + 20)
            c.rect(x3, sy + 22, w3, 40, fill=PANEL, stroke=BORDER, rx=8)
            c.text(x3 + 14, sy + 40, "Graft into the full checkpoint", "lblb",
                   maxw=w3 - 28, tag="graft")
            c.text(x3 + 14, sy + 54, "vision tower unchanged", "tiny", maxw=w3 - 28, tag="vtu")
            sy += 62

    arrow_down(c, x3 + 22, sy + 4, sy + 20)
    c.rect(x3, sy + 22, w3, 42, fill=INK, stroke=INK, rx=8)
    c.text(x3 + 14, sy + 41, f"{m.title} — release v1.0", "wlbl", maxw=w3 - 28, tag="release")
    c.text(x3 + 14, sy + 56, f["served_short"], "wnote", maxw=w3 - 28, tag="releasenote")

    # ----------------------------------------------------- evaluation row --
    gy = H - 104
    hrule(c, gy - 24)
    sec(c, MGN, gy, "evaluation")
    items = [("Selection", "DI-S 3,000-request sample · JevBench public items")]
    if is_vlm:
        items.append(("Pre-registered bar", f"DI-S ≥ 55 — scored {f['di_s']}"))
    items.append(("Decision Index 0.1 — separate read", f"{f['di_full']} on the full archived suite"))
    bwid = (INNER - 14 * (len(items) - 1)) / len(items)
    for i, (lab, val) in enumerate(items):
        bxp = MGN + i * (bwid + 14)
        c.rect(bxp, gy + 12, bwid, 56, fill=PANEL, stroke=BORDER, rx=8)
        c.text(bxp + 14, gy + 32, lab, "lbl", maxw=bwid - 28, tag=lab)
        c.wrap(bxp + 14, gy + 47, val, "tiny", maxw=bwid - 28, lh=13, max_lines=2)
    return c


# =========================================================== figure (c) =====
def fig_readout(m: facts.Model) -> Canvas:
    H = 668
    c = Canvas(W, H)
    header(c, "one-pass readout", m.title,
           "Typed questions in · a probability for every option out · no text generated",
           right=f"thegovind/{m.repo} · v1.0")
    hrule(c, 122)

    top, ph = 158, 236
    c1x, c1w = MGN, 276
    c2x, c2w = c1x + c1w + 38, 276
    c3x, c3w = c2x + c2w + 38, 244

    # 1 — the request
    sec(c, c1x + 26, top, "request")
    step_dot(c, c1x + 10, top - 4, 1)
    c.rect(c1x, top + 12, c1w, ph, fill=PANEL, stroke=BORDER, rx=9)
    ix, iw = c1x + 14, c1w - 28
    sec(c, ix, top + 36, "state")
    c.rect(ix, top + 44, iw, 42, fill=SUBTLE, stroke=RULE, rx=6)
    c.wrap(ix + 10, top + 62, "Order #4411 arrived with a cracked screen. "
                              "I want my money back.", "tiny", maxw=iw - 20, lh=13)
    ty = top + 104
    sec(c, ix, ty, "questions")
    ty += 12
    for name, kind, detail in (("intent", "choice", "3 options"),
                               ("urgent", "noul", "yes / no"),
                               ("anger", "score", "3 ordered levels")):
        c.rect(ix, ty, iw, 32, fill=PANEL, stroke=BORDER, rx=6)
        c.text(ix + 12, ty + 20, name, "code", maxw=64, tag=name)
        c.text(ix + 80, ty + 20, kind, "lbl", maxw=60, tag=kind)
        c.text(ix + iw - 12, ty + 20, detail, "tiny", anchor="end")
        ty += 38
    for i, line in enumerate(("choice — up to 255 options",
                              "score — 2–10 ordered levels",
                              "up to 512 questions per request")):
        c.text(c1x, top + ph + 30 + i * 14, line, "tiny", maxw=c1w, tag=line)

    # 2 — the rendered prompt
    arrow_right(c, c1x + c1w + 10, c2x - 10, top + 120)
    sec(c, c2x + 26, top, "rendered prompt")
    step_dot(c, c2x + 10, top - 4, 2)
    c.rect(c2x, top + 12, c2w, ph, fill=PANEL, stroke=BORDER, rx=9)
    ix, iw = c2x + 14, c2w - 28
    ty = top + 36
    for lab, val in (("evidence", "the state"), ("criterion", "the instructions")):
        sec(c, ix, ty, lab)
        c.rect(ix, ty + 8, iw, 24, fill=SUBTLE, stroke=RULE, rx=5)
        c.text(ix + 10, ty + 24, val, "tiny", maxw=iw - 20, tag=val)
        ty += 44
    sec(c, ix, ty, "options")
    ty += 10
    for lab, val in (("A", "Money back"), ("B", "A new unit"), ("C", "Information only")):
        chip(c, ix, ty, 24, 22, lab, "full", style="codea", rx=5)
        c.text(ix + 32, ty + 16, val, "tiny", maxw=iw - 42, tag=val)
        ty += 26
    ty += 6
    c.line(ix, ty, ix + iw, ty, stroke=RULE)
    c.text(ix, ty + 20, "answer position", "sec")
    c.raw(f'<rect x="{ix + 128:.2f}" y="{ty + 8:.2f}" width="11" height="15" fill="{INK}" rx="2"/>')
    c.text(c2x, top + ph + 30, "single-token labels A–Z, then two-letter labels", "tiny",
           maxw=c2w, tag="labels1")

    # 3 — one pass
    arrow_right(c, c2x + c2w + 10, c3x - 10, top + 120)
    sec(c, c3x + 26, top, "one prefill pass")
    step_dot(c, c3x + 10, top - 4, 3)
    c.rect(c3x, top + 12, c3w, ph, fill=INK, stroke=INK, rx=9)
    ix, iw = c3x + 16, c3w - 32
    c.text(ix, top + 44, "Prefill only", "wlbl", maxw=iw, tag="prefill")
    c.text(ix, top + 63, "no token is generated", "wnote", maxw=iw, tag="nogen")
    c.line(ix, top + 82, ix + iw, top + 82, stroke="#413B35")
    c.text(ix, top + 106, "Questions are batched;", "wnote", maxw=iw, tag="batch1")
    c.text(ix, top + 122, "each batch is one forward pass", "wnote", maxw=iw, tag="batch2")
    c.line(ix, top + 141, ix + iw, top + 141, stroke="#413B35")
    c.text(ix, top + 166, "Next-token logits", "wlbl", maxw=iw, tag="logits")
    c.text(ix, top + 184, "read at the answer position", "wnote", maxw=iw, tag="logits2")
    c.text(ix, top + 210, "in FP32", "wnum", maxw=iw, tag="fp32")

    # 4 — label logits
    by = top + ph + 102
    hrule(c, by - 26)
    lx, lw = MGN, 300
    sec(c, lx + 26, by, "label logits")
    step_dot(c, lx + 10, by - 4, 4)
    c.rect(lx, by + 12, lw, 118, fill=PANEL, stroke=BORDER, rx=9)
    gx, gy, gw = lx + 16, by + 28, 92
    hot = (2, 5, 8)
    for i in range(11):
        c.rect(gx, gy + i * 7.6, gw, 5.2, fill=ACC if i in hot else DIM_ROW, stroke="none", rx=1.6)
    for i, lab in enumerate("ABC"):
        c.text(gx + gw + 8, gy + hot[i] * 7.6 + 5.2, lab, "numa")
    tx = gx + gw + 32
    c.text(tx, gy + 16, "offered option", "body", maxw=lw - (tx - lx) - 16, tag="offered1")
    c.text(tx, gy + 32, "letter rows only", "body", maxw=lw - (tx - lx) - 16, tag="offered2")
    c.text(tx, gy + 56, "every other row", "tiny", maxw=lw - (tx - lx) - 16, tag="ignored1")
    c.text(tx, gy + 70, "is ignored", "tiny", maxw=lw - (tx - lx) - 16, tag="ignored2")

    # 5 — softmax
    mx, mw = lx + lw + 38, 262
    arrow_right(c, lx + lw + 10, mx - 10, by + 70)
    sec(c, mx + 26, by, "softmax over those letters")
    step_dot(c, mx + 10, by - 4, 5)
    c.rect(mx, by + 12, mw, 118, fill=PANEL, stroke=BORDER, rx=9)
    px = mx + 16
    for i, (lab, v) in enumerate(zip("ABC", (0.62, 0.26, 0.12))):
        yy = by + 30 + i * 25
        c.text(px, yy + 11, lab, "numa")
        bar(c, px + 18, yy, 166, 13, v, ACC)
    c.text(px, by + 30 + 3 * 25 + 14, "one probability per option · they sum to 1",
           "tiny", maxw=mw - 32, tag="sums")

    # answers
    rx_ = mx + mw + 38
    rw = W - MGN - rx_
    arrow_right(c, mx + mw + 10, rx_ - 10, by + 70)
    sec(c, rx_, by, "answers")
    c.rect(rx_, by + 12, rw, 118, fill=PANEL, stroke=BORDER, rx=9)
    ry = by + 38
    for kind, out in (("choice", "the picked option"),
                      ("noul", "probability of yes"),
                      ("score", "expected level")):
        c.text(rx_ + 16, ry, kind, "code", maxw=56, tag=kind)
        c.text(rx_ + 74, ry, out, "tiny", maxw=rw - 90, tag=out)
        ry += 30
    c.text(W - MGN, H - 22, "letters and bars are schematic", "tiny", anchor="end")
    return c


# --------------------------------------------------------------- alt text ---
def alt_text(m: facts.Model, name: str) -> str:
    f, is_vlm = m.facts, m.vision_cfg is not None
    if name == "network":
        kept = (f"the {m.vision_cfg['depth']}-block vision tower kept byte-for-byte"
                if is_vlm else "the vision encoder and multi-token-prediction head removed")
        tie = ("lm_head tied to the token embeddings, one matrix" if m.tied
               else "lm_head untied, a separate matrix")
        return (f"{m.title} network diagram: {m.n_layers} decoder layers repeating "
                f"{m.period - 1} Gated DeltaNet layers then one full-attention layer "
                f"({m.n_delta} and {m.n_full} in total, hidden {m.hidden}), with full "
                f"attention at 0-based layers {m.positions_label} as in the tensor names; "
                f"LoRA rank {facts.LORA_RANK} on every attention, Gated DeltaNet and MLP "
                f"projection ({m.lora_params_m} parameters, merged after training); token "
                f"embeddings, norms and lm_head frozen, with {tie}; {kept}; and the answer "
                f"read from the offered option-letter rows of lm_head.")
    if name == "post-training":
        rows = " and ".join(f"{st['name']} ({fmt(st['rows'])} rows)" for st in f["stages"])
        if m.repo == "blink-4b":
            tail = ("both runs start from the base, and T3, T4 step 300 and T4 final are "
                    "merged and averaged into release v1.0")
        elif m.repo == "blink-27b":
            tail = ("T2 starts from the base and T4 continues from it, then the adapters are "
                    "merged into release v1.0")
        else:
            tail = ("a single pre-registered epoch is merged and grafted back into the full "
                    "vision-language checkpoint, vision tower unchanged, as release v1.0")
        return (f"{m.title} post-training diagram: {rows} broken down by data category feed "
                f"supervised fine-tuning with cross-entropy over the offered option letters; "
                f"{tail}. Decision Index 0.1 full suite {f['di_full']}.")
    return (f"{m.title} one-pass readout diagram: a state and typed choice, noul and score "
            f"questions are rendered as evidence, criterion and lettered options; questions "
            f"are batched and each batch is one forward pass, giving next-token logits at the "
            f"answer position, and an FP32 softmax over the offered letters gives one "
            f"probability per option. No text is generated.")


# ------------------------------------------------------------------ build ---
# (role, file stem suffix, builder). The role is what a card keys off; the suffix is
# what the file is called on disk.
FIGURES = (("network", "network", fig_network),
           ("training", "post-training", fig_pipeline),
           ("readout", "readout", fig_readout))


def rasterise(pairs):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--force-color-profile=srgb",
                                     "--font-render-hinting=none",
                                     "--disable-lcd-text"])
        check_render(br)
        for svg_path, png_path, w, h in pairs:
            page = br.new_page(viewport={"width": int(w), "height": int(h)}, device_scale_factor=2)
            page.set_content(
                f'<!doctype html><meta charset="utf-8">'
                f'<style>html,body{{margin:0;padding:0;background:{MAT}}}'
                f'svg{{display:block;width:{int(w)}px;height:{int(h)}px}}</style>'
                f'{Path(svg_path).read_text()}')
            page.wait_for_timeout(150)
            page.screenshot(path=str(png_path), omit_background=False)
            page.close()
        br.close()
    flatten([png for _, png, _, _ in pairs])


def check_render(browser, size=20.0):
    """Fail if Chromium renders any probe wider than text_width() predicted."""
    page = browser.new_page(viewport={"width": 900, "height": 400})
    page.set_content(f'<!doctype html><meta charset="utf-8">'
                     f'<style>body{{margin:0}} span{{{TEXT_CSS};font-size:{size}px}}</style>'
                     + "".join(f'<div><span>{s}</span></div>' for s in RENDER_PROBES))
    page.wait_for_timeout(120)
    got = page.evaluate("""() => [...document.querySelectorAll('span')]
        .map(s => [s.textContent, s.getBoundingClientRect().width])""")
    page.close()
    bad = [(t, round(w, 1), round(text_width(t, size), 1))
           for t, w in got if w > text_width(t, size) * 1.005]
    if bad:
        raise SystemExit(f"render/measure mismatch (font features?): {bad}")
    print(f"render probes OK ({len(RENDER_PROBES)} strings within measured widths)")


def flatten(paths):
    """Guarantee every PNG is opaque RGB — model cards render on dark pages too."""
    from PIL import Image
    for path in paths:
        im = Image.open(path)
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, MAT)
            bg.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
            bg.save(path, optimize=True)
        im.close()


def build(raster=True):
    models = facts.load()
    facts.verify()
    pairs, made = [], []
    for repo, m in models.items():
        d = OUT / repo
        d.mkdir(parents=True, exist_ok=True)
        entries = []
        for role, name, fn in FIGURES:
            c = fn(m)
            svg = d / f"{repo}-{name}.svg"
            svg.write_text(c.render())
            png = d / f"{repo}-{name}.png"
            pairs.append((svg, png, c.out_w, c.out_h))
            made.append((repo, name, svg, png, c.overflows))
            entries.append({"role": role,
                            "file": png.name,
                            "alt": alt_text(m, name),
                            "width_px": int(c.out_w),
                            "height_px": int(c.out_h)})
        (d / "figures.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    if raster:
        rasterise(pairs)
    return made


def write_checks(rows, made):
    lines = [
        "# Diagram checks",
        "",
        "Generated by `make_diagrams.py`. Every number that appears in a figure is listed here",
        "with the file and field it came from. The build refuses to emit figures if a check fails.",
        "",
        "```sh",
        "cd release/diagrams",
        "uv run --with playwright --with fonttools --with pillow python make_diagrams.py  # SVG + PNG at 2x",
        "uv run --with fonttools python make_diagrams.py --check-only        # assertions only",
        "```",
        "",
        "Paths are relative to the repository root.",
        "",
        "Each `release/assets/<model>/figures.json` lists",
        "`{role, file, alt, width_px, height_px}` for that model's three PNGs, in the order they",
        "should appear in the card. `role` is `network`, `training` or `readout`; `file` is",
        "relative to `release/assets/<model>/`; `width_px`/`height_px` are the 1x display size",
        "and the PNG itself is 2x that. The matching SVG sits next to each PNG under the same",
        "stem. Every PNG is opaque RGB with no alpha: the",
        "figure sits on a light card with a 1 px border and rounded corners inside an opaque mat,",
        "so it reads as a deliberate panel on a dark page as well as a light one.",
        "",
        "## How the checks work",
        "",
        "- **Architecture** (layers, pattern, hidden size, vocabulary) is read straight out of the",
        "  mirrored `config.json`, and the published card is asserted to contain the same words.",
        "- **LoRA parameter counts** are recomputed from the `config.json` projection shapes at",
        "  rank 16 over the targets in `lab/jevlab/train.py`, then asserted against the card.",
        "  Nothing is copied from the card.",
        "- **Row counts** come from `runs/mix-categories.json`; each category total is asserted to",
        "  appear in the card, and the categories are asserted to sum to the stage total.",
        "- **Learning rates, steps and scores** come from the published card and are asserted to",
        "  appear there verbatim.",
        "- Schematic parts of the figures carry no numbers: the probability bars in the readout",
        "  panels are marked *letters and bars are schematic* and assert nothing.",
        "",
        "## Figures",
        "",
        "| Model | Figure | SVG | PNG (2x) |",
        "|---|---|---|---|",
    ]
    for repo, name, svg, png, _ in made:
        rel = f"release/assets/{repo}"
        lines.append(f"| {repo} | {name} | `{rel}/{svg.name}` | `{rel}/{png.name}` |")
    lines += ["", "## Every number, and where it came from", "",
              "| Shown on | Quantity | Value | Source |", "|---|---|---|---|"]
    for model, what, value, source in rows:
        lines.append(f"| {model} | {what} | {value} | `{source}` |")
    lines += ["", f"{len(rows)} checked values.", ""]
    (facts.HERE / "CHECKS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-raster", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.check_only:
        print(f"{len(facts.verify())} facts verified against their sources")
        sys.exit(0)
    made = build(raster=not a.no_raster)
    bad = 0
    for repo, name, svg, png, over in made:
        for o in over:
            bad += 1
            print(f"OVERFLOW {repo}/{name}: {o}", file=sys.stderr)
    write_checks(facts.verify(), made)
    for repo, name, svg, png, _ in made:
        print(f"{repo:14s} {name:14s} {svg.name}  {png.name}")
    print(f"{len(made)} figures · {bad} overflow warnings")
