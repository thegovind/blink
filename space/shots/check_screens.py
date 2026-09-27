"""Screen click at a desk and on a phone, light and dark: the presets and a visitor's own screenshot.

    cd space && BLINK_MOCK=1 BLINK_MODELS=thegovind/blink-4b,thegovind/blink-mimo-9b BLINK_PORT=7931 \\
        uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7931/ BLINK_SHOTS=/tmp/shots uv run --with playwright python shots/check_screens.py

Every page mark must sit on the box burned into the screenshot (to a pixel), every mark must
carry its probability, the chosen one must be lit, and nothing may scroll sideways. The upload
flow runs in full: a generated screenshot, the grid, boxes dragged with a mouse (tapped corner to
corner on the phone), Decide, a box removed by its number, and the grid again. Screenshots go to
BLINK_SHOTS (default: this folder). Exits non-zero on any problem, a console error included.

BLINK_SAVED=1 checks the saved runs instead, against a server that answers from its recordings
alone (no GPU, no mock, no drafter), as a visitor first sees the tab:
    cd space && BLINK_ENGINE=replay BLINK_AUTHOR=off BLINK_MODELS=thegovind/blink-4b,thegovind/blink-mimo-9b \\
        BLINK_PORT=7932 uv run --with gradio==6.28.0 python app.py &
    BLINK_SAVED=1 BLINK_URL=http://127.0.0.1:7932/ BLINK_SHOTS=/tmp/shots \\
        uv run --with playwright python shots/check_screens.py
Each preset must then show its saved run exactly as recorded, read here by the Space's own code, and
a visitor's screenshot, which has no saved run, must say so.
"""

import asyncio
import json
import os
import sys
import tempfile

from playwright.async_api import async_playwright

from live import answer_wait, app_error, sign_in

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7931/").rstrip("/")
OUT = os.environ.get("BLINK_SHOTS") or HERE
DESK = {"width": 1456, "height": 900}
PHONE = {"width": 390, "height": 844}
SCHEMES = (("light", "light", ""), ("dark", "dark", ""), ("theme-dark", "light", "&__theme=dark"))
SAVED = os.environ.get("BLINK_SAVED") == "1"  # the server answers from its recordings alone
KIND = "screen-saved" if SAVED else "screen"
LIME = "rgb(212, 245, 92)"
UPLOAD = (1280, 800)
# the two buttons drawn on the generated screenshot, in its own pixels
BUTTONS = ((60, 300, 380, 380), (420, 300, 740, 380))

with open(os.path.join(HERE, "..", "screens", "screens.json"), encoding="utf-8") as fh:
    META = json.load(fh)
PAD = META["style"]["stroke"] + META["style"]["halo"]

GEOMETRY = """() => {
  const img = document.querySelector('#screen-out .blk-shot img');
  const r = img.getBoundingClientRect();
  const read = (el) => { const b = el.getBoundingClientRect();
    return {n: el.dataset.n, x: b.left - r.left, y: b.top - r.top, w: b.width, h: b.height}; };
  return {
    img: {w: r.width, h: r.height, nw: img.naturalWidth, nh: img.naturalHeight, done: img.complete},
    marks: Array.from(document.querySelectorAll('#screen-out .blk-som')).map(read),
    tags: Array.from(document.querySelectorAll('#screen-out .blk-somtag')).map(read),
  };
}"""

PAINTED = """() => Array.from(document.querySelectorAll('#screen-out .blk-som')).map((el) => ({
  n: el.dataset.n, win: el.classList.contains('win'),
  p: parseFloat(getComputedStyle(el).getPropertyValue('--sp')),
  shadow: getComputedStyle(el).boxShadow,
  tag: getComputedStyle(document.querySelector(`#screen-out .blk-somtag[data-n="${el.dataset.n}"]`)).backgroundColor,
  pct: (document.querySelector(`#screen-out .blk-somtag[data-n="${el.dataset.n}"] i`) || {}).textContent || '',
}))"""

OVERFLOW = """() => {
  const w = window.innerWidth;
  const wide = [];
  for (const el of document.querySelectorAll('#screen-out *, .blk-screenrow .blk-side *')) {
    if (el.offsetParent === null) continue;
    if (el.closest('.blk-marks')) continue;  // marks are clipped by their frame on purpose
    const b = el.getBoundingClientRect();
    if (b.width && (b.right > w + 1 || b.left < -1)) wide.push(el.className || el.tagName);
  }
  return {page: document.documentElement.scrollWidth, win: w, wide: wide.slice(0, 5)};
}"""

PAINT = """() => {
  const surface = (e) => {
    for (let n = e; n; n = n.parentElement) {
      const bg = getComputedStyle(n).backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') return bg;
    }
    return 'none';
  };
  const seen = [];
  for (const e of document.querySelectorAll('#screen-out *, .blk-screenrow .blk-side *')) {
    if (e.offsetParent === null || e.children.length) continue;
    const text = (e.textContent || '').trim();
    if (!text) continue;
    seen.push([text.slice(0, 40), getComputedStyle(e).color, surface(e)]);
  }
  return seen;
}"""


def sample_png(path: str) -> None:
    """A made-up settings page: nobody's screenshot, drawn here."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", UPLOAD, (244, 246, 251))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, UPLOAD[0], 90], fill=(31, 42, 68))
    draw.rounded_rectangle([60, 160, 620, 240], 12, outline=(154, 163, 181), width=2, fill="white")
    draw.rounded_rectangle(list(BUTTONS[0]), 12, fill=(43, 108, 255))
    draw.rounded_rectangle(list(BUTTONS[1]), 12, outline=(204, 51, 51), width=3, fill="white")
    img.save(path)


async def answered(page, timeout=20000):
    await page.wait_for_selector("#screen-out .blk-shot.answered", timeout=answer_wait(timeout))
    await page.wait_for_selector("#screen-out .blk-scr:not(.busy)", timeout=answer_wait(timeout))
    await page.wait_for_timeout(1300)  # the glows settle


def recorded() -> tuple[dict, dict]:
    """What each preset's saved run says, read by the Space's own code from its recordings, and the
    card's own words: the page must show exactly this."""
    os.environ.pop("BLINK_MOCK", None)
    os.environ["BLINK_ENGINE"] = "replay"
    os.environ.setdefault("BLINK_MODELS", "thegovind/blink-4b,thegovind/blink-mimo-9b")
    sys.path.insert(0, os.path.join(HERE, ".."))
    import screens
    import ui

    said = {}
    for shot in screens.presets():
        out = screens.decide(shot, prefer="saved")
        _tone, act, head, _detail = screens.verdict(out["answers"])
        said[shot.key] = {"act": act, "head": head, "choice": out["answers"]["element"]["choice"],
                          "probs": out["answers"]["element"]["probabilities"]}
    return said, ui.SCREEN


async def missed(page, note: str, timeout=20000) -> str:
    """The card after a Decide that the recordings can't answer: not answered, and the note in its place."""
    try:
        await page.wait_for_function(
            """(note) => { const s = document.querySelector('#screen-out .blk-scr');
              if (!s || s.classList.contains('busy')) return false;
              const idle = s.querySelector('.blk-sv-idle');
              return !!s.querySelector('.blk-shot.answered') || (!!idle && idle.innerText.trim() === note); }""",
            arg=note, timeout=answer_wait(timeout))
    except Exception:  # noqa: BLE001 - report what the card says instead of a bare timeout
        pass
    if await page.locator("#screen-out .blk-shot.answered").count():
        return "answered"
    return (await page.locator("#screen-out .blk-sv-idle").first.inner_text()).strip()


def inside(geo) -> list[str]:
    """Every number on the screenshot: the frame clips, so one that runs off it is cut, not scrolled."""
    w, h = geo["img"]["w"], geo["img"]["h"]
    return [f"number {t['n']} runs off the screenshot at {[round(t[k], 1) for k in 'xywh']}" for t in geo["tags"]
            if t["x"] < -0.5 or t["y"] < -0.5 or t["x"] + t["w"] > w + 0.5 or t["y"] + t["h"] > h + 0.5]


def aligned(geo, boxes, size, tag_boxes=None) -> list[str]:
    """Each page mark against the box burned into the image, in displayed pixels."""
    W, H = size
    sx, sy = geo["img"]["w"] / W, geo["img"]["h"] / H
    off = []
    for mark, (x1, y1, x2, y2) in zip(geo["marks"], boxes):
        want = ((x1 - PAD) * sx, (y1 - PAD) * sy, (x2 - x1 + 2 * PAD) * sx, (y2 - y1 + 2 * PAD) * sy)
        got = (mark["x"], mark["y"], mark["w"], mark["h"])
        if max(abs(a - b) for a, b in zip(got, want)) > 1.0:
            off.append(f"mark {mark['n']} at {[round(v, 1) for v in got]} not {[round(v, 1) for v in want]}")
    for tag, rect in zip(geo["tags"], tag_boxes or []):
        tx1, ty1, tx2, ty2 = (rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy)
        if not (tag["x"] <= tx1 + 1 and tag["y"] <= ty1 + 1 and tag["x"] + tag["w"] >= tx2 - 1
                and tag["y"] + tag["h"] >= ty2 - 1):
            off.append(f"number {tag['n']} does not cover the burned-in one")
    if len(geo["marks"]) != len(boxes):
        off.append(f"{len(geo['marks'])} marks for {len(boxes)} boxes")
    return off + inside(geo)


async def run(pw, size, scheme, ok, sample: str, said: dict, words: dict) -> list:
    name, color, query = scheme
    tag = f"{size['width']}px {name}"
    phone = size is PHONE
    browser = await pw.chromium.launch()
    ctx = await browser.new_context(viewport=size, device_scale_factor=2, color_scheme=color,
                                    has_touch=phone, is_mobile=phone)
    page = await ctx.new_page()
    await sign_in(page)
    errors = []
    page.on("console", lambda m: errors.append(m.text) if app_error(m) else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    await page.goto(f"{URL}/?tab=use-cases&case=screen{query}", wait_until="load")
    await page.wait_for_selector("#screen-out .blk-shot img", timeout=60000)
    await answered(page)
    if name != "light":
        ok(f"{tag} the page is marked dark", await page.evaluate("() => document.body.classList.contains('dark')"))
    head = await page.evaluate("() => (document.querySelector('.blk-top .blk-meta') || {}).textContent || ''")
    ok(f"{tag} the masthead names blink-mimo-9b", "blink-mimo-9b" in head)
    ok(f"{tag} the card says which model reads screens",
       "blink-mimo-9b" in await page.locator(".blk-eyechip").first.inner_text())
    by = await page.locator("#screen-out .blk-by").first.inner_text()
    ok(f"{tag} and that model answered: {by}", by == "blink-mimo-9b")

    for i, preset in enumerate(META["presets"]):
        if i:
            await page.get_by_role("button", name=preset["label"], exact=True).first.click()
            await page.wait_for_timeout(300)
            await answered(page)
        geo = await page.evaluate(GEOMETRY)
        where = f"{tag} {preset['key']}"
        ok(f"{where} screenshot loaded {geo['img']['nw']}x{geo['img']['nh']}",
           geo["img"]["done"] and (geo["img"]["nw"], geo["img"]["nh"]) == tuple(preset["size"]))
        off = aligned(geo, [b["box"] for b in preset["boxes"]], preset["size"], [b["tag"] for b in preset["boxes"]])
        ok(f"{where} {len(geo['marks'])} marks on their boxes {off[:2]}", not off)
        painted = await page.evaluate(PAINTED)
        wins = [m for m in painted if m["win"]]
        glow = all(("rgba(212, 245, 92" if m["win"] else "rgba(36, 71, 245") in m["shadow"] for m in painted)
        pcts = all(m["pct"].endswith("%") and abs(float(m["pct"][:-1]) - 100 * m["p"]) <= 0.51 for m in painted)
        ok(f"{where} every mark glows with its probability", glow and pcts and len(painted) == len(preset["boxes"]))
        act = await page.evaluate("() => document.querySelector('#screen-out .blk-shot').dataset.act")
        want = said.get(preset["key"])
        if want is None:  # the mock is shaped by each preset's own answer
            want = {"act": "done" if preset["expect"].get("done") == "yes" else None,
                    "choice": preset["expect"].get("element")}
        if want["act"] == "done":
            ok(f"{where} a finished task lights nothing ({act})", act == "done" and not wins)
        else:
            lit = wins[0]["n"] if len(wins) == 1 else None
            ok(f"{where} one mark is lit, {lit}, in the flash colour",
               lit == want["choice"] and wins[0]["tag"] == LIME and act == (want["act"] or act))
        if said:
            head = await page.evaluate("() => document.querySelector('#screen-out .blk-sv h3').firstChild.textContent")
            tags = await page.locator("#screen-out .blk-saved").all_text_contents()
            mock = await page.locator("#screen-out .blk-mock").count()
            same = len(painted) == len(want["probs"]) and all(
                abs(m["p"] - want["probs"][m["n"]]) < 1e-4 for m in painted)
            ok(f"{where} shows its saved run as recorded: {head!r} {tags}",
               head == want["head"] and tags == [words["saved"]] and not mock and same)
        rings = await page.locator("#screen-out .blk-ring").count()
        ok(f"{where} both checks are drawn", rings == 2)
        room = await page.evaluate(OVERFLOW)
        ok(f"{where} nothing scrolls sideways {room}", room["page"] <= room["win"] and not room["wide"])
        if i == 0:
            await page.screenshot(path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-preset.png"), full_page=True)
            await page.locator("#screen-out .blk-shot").screenshot(
                path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-stage.png"))
        if preset["key"] == "done":
            await page.screenshot(path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-done.png"), full_page=True)
    paint = await page.evaluate(PAINT)

    # a visitor's own screenshot: the grid first, then their boxes. The browser reads the file
    # when the page uploads it, so it has to outlive this call
    await page.locator(".blk-screenrow input[type=file]").set_input_files(sample)
    try:
        await page.wait_for_selector('#screen-out .blk-shot[data-draw="1"]', timeout=answer_wait(20000))
    except Exception:  # noqa: BLE001 - say what the page said instead of a bare timeout
        said = await page.evaluate("() => (document.querySelector('.blk-screenrow .blk-trouble') || {}).innerText")
        await page.screenshot(path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-upload-failed.png"))
        ok(f"{tag} the screenshot uploads ({said!r})", False)
        await browser.close()
        return []
    await page.wait_for_timeout(700)
    cells = await page.locator("#screen-out .blk-som.cell").count()
    task = await page.locator("#screen-task textarea").input_value()
    ok(f"{tag} upload shows a numbered 3x3 grid and an empty task ({cells}, {task!r})", cells == 9 and task == "")
    await page.get_by_text("Grid 4\u00d74", exact=True).first.click()
    await page.wait_for_timeout(900)
    ok(f"{tag} the 4x4 grid has 16 cells", await page.locator("#screen-out .blk-som.cell").count() == 16)
    await page.locator("#screen-run").click()
    await page.wait_for_timeout(800)
    trouble = await page.locator(".blk-screenrow .blk-problems").first.inner_text()
    ok(f"{tag} Decide without a task says so: {trouble!r}", "Add a task." in trouble)
    await page.locator("#screen-task textarea").fill("Save the new display name.")
    await page.locator("#screen-run").click()
    if said:
        note = await missed(page, words["miss"])
        ok(f"{tag} a screenshot with no saved run says so: {note!r}", note == words["miss"])
    else:
        await answered(page)
        heat = await page.evaluate(PAINTED)
        act = await page.evaluate("() => document.querySelector('#screen-out .blk-shot').dataset.act")
        lit = sum(1 for m in heat if m["win"])
        ok(f"{tag} the grid answers as a heat map ({act}, {lit} lit)",
           len(heat) == 16 and lit == (0 if act == "done" else 1) and all(m["pct"] or m["p"] < 0.05 for m in heat))
    off = inside(await page.evaluate(GEOMETRY))
    ok(f"{tag} every cell's number stays on the screenshot {off[:2]}", not off)
    before = await page.locator("#screen-out .blk-shot-frame").bounding_box()
    top = await page.evaluate("() => window.scrollY")

    for x1, y1, x2, y2 in BUTTONS:
        # where the screenshot is now, as a visitor sees it; drawing must not move it
        frame = await page.locator("#screen-out .blk-shot-frame").bounding_box()
        a = (frame["x"] + x1 / UPLOAD[0] * frame["width"], frame["y"] + y1 / UPLOAD[1] * frame["height"])
        b = (frame["x"] + x2 / UPLOAD[0] * frame["width"], frame["y"] + y2 / UPLOAD[1] * frame["height"])
        if phone:  # a drag scrolls a phone, so a box is two taps: one corner, then the other
            await page.touchscreen.tap(*a)
            await page.wait_for_timeout(250)
            await page.touchscreen.tap(*b)
        else:
            await page.mouse.move(*a)
            await page.mouse.down()
            await page.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, steps=5)
            await page.mouse.move(*b, steps=5)
            await page.mouse.up()
        await page.wait_for_timeout(1000)
    after = await page.locator("#screen-out .blk-shot-frame").bounding_box()
    moved = await page.evaluate("() => window.scrollY") - top
    ok(f"{tag} drawing leaves the screenshot where it was on screen ({before['y']:.0f} -> {after['y']:.0f},"
       f" scrolled {moved:.0f})", abs(after["y"] - before["y"]) <= 1 and abs(after["x"] - before["x"]) <= 1)
    lines = [ln for ln in (await page.locator("#screen-boxes textarea").input_value()).splitlines() if ln.strip()]
    got = [tuple(int(v) for v in ln.split(",")) for ln in lines]
    tol = 4 if phone else 2  # a tap lands on a device pixel, the image has more of them
    close = len(got) == 2 and all(max(abs(p - q) for p, q in zip(g, w)) <= tol * UPLOAD[0] / frame["width"]
                                  for g, w in zip(got, BUTTONS))
    ok(f"{tag} two boxes drawn where they were drawn {got}", close)
    marks = await page.locator("#screen-out .blk-som:not(.cell)").count()
    grid_off = await page.locator("#screen-grid input").first.is_disabled()
    ok(f"{tag} the boxes replace the grid ({marks} marks, grid picker off)", marks == 2 and grid_off)
    geo = await page.evaluate(GEOMETRY)
    ok(f"{tag} drawn marks sit on the drawn boxes {aligned(geo, got, UPLOAD)[:2]}", not aligned(geo, got, UPLOAD))
    await page.locator("#screen-run").click()
    if said:
        note = await missed(page, words["miss"])
        ok(f"{tag} nor with boxes drawn: {note!r}", note == words["miss"])
    else:
        await answered(page)
        painted = await page.evaluate(PAINTED)
        ok(f"{tag} the answer glows both boxes and lights one",
           len(painted) == 2 and sum(1 for m in painted if m["win"]) == 1
           and all(("rgba(212, 245, 92" if m["win"] else "rgba(36, 71, 245") in m["shadow"] for m in painted))
    off = inside(await page.evaluate(GEOMETRY))
    ok(f"{tag} and their numbers stay on the screenshot {off[:2]}", not off)
    if not said:
        ok(f"{tag} what blink read is one tap away",
           await page.locator("#screen-out summary", has_text="What blink read").count() == 1)
    room = await page.evaluate(OVERFLOW)
    ok(f"{tag} the upload scrolls nothing sideways {room}", room["page"] <= room["win"] and not room["wide"])
    await page.screenshot(path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-upload.png"), full_page=True)
    await page.locator("#screen-out .blk-shot").screenshot(
        path=os.path.join(OUT, f"{KIND}-{size['width']}-{name}-upload-stage.png"))
    paint += await page.evaluate(PAINT)

    # a box's number removes it; clearing them brings the grid back
    await page.locator('#screen-out .blk-somtag[data-n="2"]').click()
    await page.wait_for_timeout(1000)
    left = [ln for ln in (await page.locator("#screen-boxes textarea").input_value()).splitlines() if ln.strip()]
    ok(f"{tag} a box's number removes it ({len(left)} left)", len(left) == 1)
    await page.get_by_role("button", name="Clear boxes", exact=True).first.click()
    await page.wait_for_timeout(1000)
    ok(f"{tag} clearing brings the grid back",
       await page.locator("#screen-out .blk-som.cell").count() == 16
       and not await page.locator("#screen-grid input").first.is_disabled())

    # and a preset is one tap away again
    await page.get_by_role("button", name=META["presets"][1]["label"], exact=True).first.click()
    await answered(page)
    ok(f"{tag} a preset takes the card back",
       await page.locator('#screen-out .blk-shot[data-draw="0"]').count() == 1
       and not await page.locator("#screen-boxes").is_visible())
    search = await page.evaluate("() => window.location.search")
    ok(f"{tag} the address names the case: {search}", "case=screen" in search)
    ok(f"{tag} no console errors {errors[:3]}", not errors)
    await ctx.close()
    await browser.close()
    return paint


async def main() -> int:
    problems: list[str] = []

    def ok(name, good):
        print(f"  {name:96s} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(name)

    os.makedirs(OUT, exist_ok=True)
    said, words = recorded() if SAVED else ({}, {})
    tmp = tempfile.TemporaryDirectory()
    sample = os.path.join(tmp.name, "settings.png")
    sample_png(sample)
    async with async_playwright() as pw:
        for size in (DESK, PHONE):
            painted = {}
            for scheme in SCHEMES:
                painted[scheme[0]] = await run(pw, size, scheme, ok, sample, said, words)
            for name in ("dark", "theme-dark"):
                light, dark = painted["light"], painted[name]
                pairs = [(a, b) for a, b in zip(light, dark) if a[0] == b[0]]
                off = [(a[0], a[1:], b[1:]) for a, b in pairs if a[1:] != b[1:]]
                ok(f"{size['width']}px {name} draws the card as light does ({len(pairs)} texts, {off[:2]})",
                   not off and len(pairs) >= 0.9 * max(len(light), 1))
    tmp.cleanup()
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nscreen click works at a desk and on a phone, light and dark"
          + (", from its saved runs" if SAVED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
