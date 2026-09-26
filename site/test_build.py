"""Tests for the docs site builder and link checker.

    uv run --no-project --with-requirements site/requirements.txt python -m unittest discover -s site -p 'test_*.py'
"""

from __future__ import annotations

import html
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402
import check  # noqa: E402

try:
    import markdown_it  # noqa: F401
    HAVE_MD = True
except ImportError:
    HAVE_MD = False

SITE = build.CONFIG["site_url"]
REPO = f"https://github.com/{build.CONFIG['repo']}"
FIGURE = ('<svg class="blk-fig" viewBox="0 0 10 10">'
          '<g class="blk-stage"><text class="blk-stage-t">State</text><text class="blk-stage-s">any text</text></g>'
          '<g class="blk-stage"><text class="blk-stage-t">Answers</text><text class="blk-stage-s">one pass</text></g>'
          '<text>no generated text</text></svg>')
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 4, 3) + b"\x08\x02\x00\x00\x00" + b"\0" * 16
SKILL = "---\nname: blink\nlicense: Apache-2.0\ndescription: >\n  Typed decisions.\n---\n\n# blink\n\nUse it.\n"
SOURCES = {
    "docs/index.md": ("# blink\n\nSend state and questions.\n\n"
                      "## Try it in the Space\n\nOpen the [live Space](https://huggingface.co/spaces/thegovind/blink).\n\n"
                      "## Call the API\n\nRead the [API guide](api.md).\n\n"
                      "## Quickstart\n\n```sh\npython serve.py\n```\n"),
    "docs/api.md": ("# API\n\nThe wire format.\n\n## Endpoints\n\n| Path | Use |\n|---|---|\n| `/v1/systemone` | Ask |\n\n"
                    "See [agents](agents.md#give-it-the-skill), [the index](llms.txt) and [home](index.md).\n\n"
                    "## Clients\n\n```python\nprint(1)\n```\n\n```js\nconsole.log(1)\n```\n\n```sh\necho 1\n```\n\n"
                    "## Request\n\n```json\n{\"see\": \"[not a link](nowhere.md)\"}\n```\n\n```json\n{}\n```\n\n"
                    "## Response\n\n## Errors\n"),
    "docs/agents.md": ("# Agents\n\n## Give it the skill\n\nThe [skill](../skills/blink/SKILL.md) in its "
                       "[folder](../skills/blink/), the [coding guide](../AGENTS.md), "
                       f"[llms.txt]({SITE}llms.txt), the [API copy]({SITE}api.md) and the "
                       f"[old address]({SITE}wire-format/).\n"),
    "docs/models.md": ("# Models\n\n| Model |\n|---|\n| [thegovind/blink-4b](https://huggingface.co/thegovind/blink-4b) |\n\n"
                       "![readout](img/r.png)\n"),
    "docs/llms.txt": f"# blink\n\n> Typed decisions.\n\n- [API]({SITE}api.md): the wire format\n",
    "skills/blink/SKILL.md": SKILL,
    "AGENTS.md": "# Coding agents\n",
    "LICENSE": "Apache-2.0\n",
    "space/results.py": f"def pipeline_figure():\n    return {FIGURE!r}\n",
}


def make_repo(root: Path, **overrides: str) -> None:
    for rel, text in {**SOURCES, **overrides}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    (root / "docs/img").mkdir(parents=True, exist_ok=True)
    (root / "docs/img/r.png").write_bytes(PNG)


class Helpers(unittest.TestCase):
    def test_github_slugs(self):
        cases = {
            "Errors and limits": "errors-and-limits",
            "Point a TypeSafe SDK at it": "point-a-typesafe-sdk-at-it",
            "Agent experience (AX)": "agent-experience-ax",
            "request-mixed.json": "request-mixedjson",
            "Kullback–Leibler divergence": "kullbackleibler-divergence",
            "snake_case stays": "snake_case-stays",
        }
        for text, want in cases.items():
            self.assertEqual(build.github_slug(text), want, text)

    def test_repeated_headings_are_numbered(self):
        s = build.Slugger()
        self.assertEqual([s.slug("Usage"), s.slug("Usage"), s.slug("Usage")], ["usage", "usage-1", "usage-2"])

    def test_paths(self):
        self.assertEqual(build.rel_url("", ""), "./")
        self.assertEqual(build.rel_url("", "api/"), "api/")
        self.assertEqual(build.rel_url("api", ""), "../")
        self.assertEqual(build.rel_url("models/blink-4b", "models/"), "../../models/")
        self.assertEqual([build.markdown_path(p.slug) for p in build.PAGES], ["index.md", "api.md", "agents.md", "models.md"])
        self.assertEqual(build.page_path(""), "")
        self.assertEqual(build.page_path("api"), "api/")

    def test_table_code_wraps_only_at_spaces_and_slashes(self):
        self.assertEqual(build.cell_code("/v1/systemone"),
                         '<code class="cell"><span>/v1</span><wbr><span>/systemone</span></code>')
        self.assertEqual(build.cell_code("--batch-window-ms"), '<code class="cell"><span>--batch-window-ms</span></code>')
        self.assertEqual(build.cell_code("https://x.org/a"),
                         '<code class="cell"><span>https://x.org</span><wbr><span>/a</span></code>')
        self.assertEqual(build.cell_code('{"true": "<y>"}'),
                         '<code class="cell"><span>{&quot;true&quot;:</span> <span>&quot;&lt;y&gt;&quot;}</span></code>')

    def test_redirects_point_at_pages(self):
        slugs = {p.slug for p in build.PAGES}
        self.assertFalse(slugs & set(build.REDIRECTS))
        self.assertTrue(set(build.REDIRECTS.values()) <= slugs)
        self.assertEqual(build.REDIRECTS["wire-format"], "api")
        for name in build.CONFIG["models"]:
            self.assertEqual(build.REDIRECTS[f"models/{name}"], "models")
        for old in ("overview", "examples", "results", "evaluation", "history", "first-principles", "changelog",
                    "contributing", "security"):
            self.assertEqual(build.REDIRECTS[old], "")


class Strings(unittest.TestCase):
    def test_every_string_is_used(self):
        own = json.loads((HERE / "strings.json").read_text(encoding="utf-8"))
        reused = json.loads((HERE / "strings-reused.json").read_text(encoding="utf-8"))["strings"]
        self.assertTrue(all(isinstance(v, str) and v for v in own.values()))
        self.assertFalse(set(own) & set(reused))
        code = (HERE / "build.py").read_text(encoding="utf-8")
        for key in set(own) | set(reused):
            self.assertIn(f'"{key}"', code, f"{key} is never used")


class ReusedStrings(unittest.TestCase):
    """Each reused string is still, word for word, in the text it was copied from."""

    def setUp(self):
        self.root = HERE.parent
        if not (self.root / "space" / "ui.py").exists():
            self.skipTest("not inside the repository")
        self.reused = json.loads((HERE / "strings-reused.json").read_text(encoding="utf-8"))["strings"]
        self.assertEqual(set(self.reused), {"hero.chip", "footer.note"})

    def test_chip_is_the_spaces(self):
        ui = (self.root / "space" / "ui.py").read_text(encoding="utf-8")
        self.assertIn(f'"{self.reused["hero.chip"]["text"]}"', ui)

    def test_footer_is_the_spaces(self):
        out = subprocess.run([sys.executable, "-c", "import ui; print(ui.footer())"], cwd=self.root / "space",
                             capture_output=True, text=True, env={**os.environ, "BLINK_MOCK": "1"}, check=True).stdout
        self.assertEqual(html.unescape(re.sub(r"<[^>]+>", "", out)).strip(), self.reused["footer.note"]["text"])


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class Resolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_repo(self.root)
        self.site = build.Site(self.root, self.root / "_site")
        self.page = {p.slug: p for p in build.PAGES}

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_and_site_files_stay_on_the_site(self):
        r, p = self.site.resolve, self.page
        self.assertEqual(r(p[""], "api.md"), "api/")
        self.assertEqual(r(p["api"], "agents.md#give-it-the-skill"), "../agents/#give-it-the-skill")
        self.assertEqual(r(p["api"], "index.md"), "../")
        self.assertEqual(r(p["api"], "llms.txt"), "../llms.txt")
        self.assertEqual(r(p["api"], "/models/"), "../models/")
        self.assertEqual(r(p["api"], "/wire-format/"), "../wire-format/")
        self.assertEqual(r(p["agents"], SITE + "llms.txt"), "../llms.txt")
        self.assertEqual(r(p["agents"], SITE + "api.md"), "../api.md")
        self.assertEqual(r(p[""], SITE), "./")
        self.assertEqual(r(p["api"], "#request"), "#request")

    def test_repository_files_go_to_github(self):
        r, p = self.site.resolve, self.page
        self.assertEqual(r(p["agents"], "../skills/blink/SKILL.md"), f"{REPO}/blob/main/skills/blink/SKILL.md")
        self.assertEqual(r(p["agents"], "../skills/blink/"), f"{REPO}/tree/main/skills/blink")
        self.assertEqual(r(p["agents"], "../AGENTS.md"), f"{REPO}/blob/main/AGENTS.md")
        self.assertEqual(r(p["models"], "https://huggingface.co/thegovind/blink-4b"),
                         "https://huggingface.co/thegovind/blink-4b")

    def test_images_are_copied_to_the_site(self):
        self.assertEqual(self.site.resolve(self.page["models"], "img/r.png", image=True), "../docs/img/r.png")
        self.assertIn("docs/img/r.png", self.site.assets)

    def test_a_link_to_nothing_public_stops_the_build(self):
        for href in ("../private/notes.md", "../../outside.md", "missing.md", "/nowhere/"):
            with self.assertRaises(build.BuildError, msg=href):
                self.site.resolve(self.page["api"], href)

    def test_markdown_copies_use_full_addresses_and_link_to_markdown(self):
        api = self.site.markdown_copy(self.page["api"])
        self.assertTrue(api.startswith(f"> {build.load_strings()['md.header']} {SITE}llms.txt\n\n# API\n"))
        self.assertIn(f"[agents]({SITE}agents.md#give-it-the-skill)", api)
        self.assertIn(f"[the index]({SITE}llms.txt)", api)
        self.assertIn(f"[home]({SITE}index.md)", api)
        self.assertIn('{"see": "[not a link](nowhere.md)"}', api)
        agents = self.site.markdown_copy(self.page["agents"])
        self.assertIn(f"[skill]({REPO}/blob/main/skills/blink/SKILL.md)", agents)
        self.assertIn(f"[old address]({SITE}api.md)", agents)
        models = self.site.markdown_copy(self.page["models"])
        self.assertIn(f"![readout]({SITE}docs/img/r.png)", models)


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class Built(unittest.TestCase):
    """A small repository, built and checked the way the real one is."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_repo(cls.root)
        cls.out = cls.root / "_site"
        cls.report = build.Site(cls.root, cls.out).build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, rel: str) -> str:
        return (self.out / rel).read_text(encoding="utf-8")

    def test_outputs(self):
        self.assertEqual(self.report, {"pages": 4, "markdown": 4, "redirects": len(build.REDIRECTS), "files": 2,
                                       "images": 1})
        for rel in ("index.html", "api/index.html", "agents/index.html", "models/index.html", "index.md", "api.md",
                    "agents.md", "models.md", "llms.txt", "skills/blink/SKILL.md", "404.html", "sitemap.xml",
                    ".nojekyll", "assets/site.css", "assets/site.js", "assets/favicon.svg", "docs/img/r.png"):
            self.assertTrue((self.out / rel).is_file(), rel)
        self.assertEqual(self.read("llms.txt"), SOURCES["docs/llms.txt"])
        self.assertEqual(self.read("skills/blink/SKILL.md"), SKILL)

    def test_every_link_resolves(self):
        result = check.check(self.out, self.root)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"]["redirects"], len(build.REDIRECTS))
        self.assertGreater(result["counts"]["text"], 0)

    def test_redirect_stubs(self):
        for old, new in build.REDIRECTS.items():
            stub = self.read(f"{old}/index.html")
            target = build.rel_url(old, build.page_path(new))
            self.assertIn(f'<meta http-equiv="refresh" content="0; url={target}">', stub, old)
            self.assertIn(f'<link rel="canonical" href="{SITE}{build.page_path(new)}">', stub, old)
            self.assertIn('<meta name="robots" content="noindex">', stub, old)
            self.assertTrue((self.out / build.page_path(new) / "index.html").is_file(), new)

    def test_sitemap_lists_only_pages(self):
        locs = re.findall(r"<loc>([^<]+)</loc>", self.read("sitemap.xml"))
        self.assertEqual(locs, [SITE + build.page_path(p.slug) for p in build.PAGES])

    def test_pages_point_at_their_markdown(self):
        self.assertIn('<link rel="alternate" type="text/markdown" href="index.md">', self.read("index.html"))
        self.assertIn('<link rel="alternate" type="text/markdown" href="../api.md">', self.read("api/index.html"))

    def test_home_hero_cards_and_figure(self):
        home = self.read("index.html")
        self.assertIn('<p class="lede">Send state and questions.</p>', home)
        cards = re.findall(r'<a class="card" href="([^"]+)">(.*?)</a>', home, re.S)
        self.assertEqual([href for href, _ in cards], ["https://huggingface.co/spaces/thegovind/blink", "api/"])
        self.assertTrue(all("<a " not in inner for _, inner in cards))
        self.assertIn("<b>Call the API</b><span>Read the API guide.</span>", home)
        self.assertIn('class="blk-fig"', home)
        self.assertIn("<b>Answers</b>", home)
        self.assertIn('id="quickstart"', home)
        self.assertNotIn("{{", home)

    def test_contents_list_only_on_long_pages(self):
        self.assertIn('class="toc"', self.read("api/index.html"))
        self.assertNotIn('class="toc"', self.read("agents/index.html"))
        self.assertIn('class="page solo"', self.read("agents/index.html"))

    def test_one_example_in_several_languages_becomes_tabs(self):
        api = self.read("api/index.html")
        self.assertEqual(api.count('class="code code-tabs"'), 1)
        self.assertIn('<div class="tabs" role="tablist" aria-labelledby="clients">', api)
        self.assertEqual(re.findall(r'role="tab" class="tab" id="(code-1-\d)" aria-controls="code-1-\d-panel" '
                                    r'aria-selected="(true|false)"', api),
                         [("code-1-0", "true"), ("code-1-1", "false"), ("code-1-2", "false")])
        self.assertIn('<div class="code-panel on" role="tabpanel" id="code-1-0-panel" aria-labelledby="code-1-0" '
                      'data-lang="python"><pre><code class="language-python">print(1)\n</code></pre></div>', api)
        self.assertEqual(api.count('<div class="code">'), 2)  # two json blocks in a row stay apart
        copy = self.read("api.md")
        self.assertIn("```python\nprint(1)\n```\n\n```js\nconsole.log(1)\n```", copy)

    def test_table_code_in_pages(self):
        self.assertIn('<td><code class="cell"><span>/v1</span><wbr><span>/systemone</span></code></td>',
                      self.read("api/index.html"))


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class Cards(unittest.TestCase):
    def home(self, text: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_repo(root, **{"docs/index.md": text})
            site = build.Site(root, root / "_site")
            site.pitch = "blink"
            return site.home_page()

    def test_a_list_of_links_becomes_cards(self):
        home = self.home("# blink\n\nPitch.\n\n- [Space](https://huggingface.co/spaces/thegovind/blink): try it\n"
                         "- [API](api.md) \u2014 call it\n")
        self.assertEqual(re.findall(r'<a class="card" href="([^"]+)"><b>([^<]+)</b><span>([^<]*)</span>', home),
                         [("https://huggingface.co/spaces/thegovind/blink", "Space", "try it"), ("api/", "API", "call it")])

    def test_one_section_is_not_a_card_row(self):
        home = self.home("# blink\n\nPitch.\n\n## Only\n\nSee the [API](api.md).\n\n## Quickstart\n\n```sh\nx\n```\n")
        self.assertNotIn('class="card"', home)
        self.assertIn('id="only"', home)


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class Checker(unittest.TestCase):
    def test_reports_broken_links_anchors_and_redirects(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / "out", Path(tmp)
            (out / "a").mkdir(parents=True)
            (out / "index.html").write_text('<a href="a/">a</a><a href="a/#here">h</a><a href="a/#gone">g</a>'
                                            '<a href="b/">b</a><p id="x"></p><p id="x"></p>', encoding="utf-8")
            (out / "a/index.html").write_text('<meta http-equiv="refresh" content="0; url=../c/"><p id="here"></p>',
                                              encoding="utf-8")
            (out / "a.md").write_text(f"> {SITE}llms.txt\n\n[b]({SITE}b.md) `{SITE}code.md`\n"
                                      f"```\n[c]({SITE}fenced.md)\n```\n[h]({SITE}a.md#here) [n]({SITE}a.md#none)\n",
                                      encoding="utf-8")
            (out / "llms.txt").write_text(f"- [a]({SITE}a.md)\n- [r]({REPO}/blob/main/nothing.md)\n", encoding="utf-8")
            errors = check.check(out, root)["errors"]
        joined = "\n".join(errors)
        for needle in ("#gone is not an id", "b/ does not exist", "id 'x' appears 2 times", "../c/ does not exist",
                       "a.md:3: b.md does not exist", "nothing.md is not a file", "#none is not an id on a/index.html"):
            self.assertIn(needle, joined)
        self.assertNotIn("code.md", joined)
        self.assertNotIn("fenced.md", joined)
        self.assertEqual(len(errors), 7, joined)


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class RealSite(unittest.TestCase):
    """The repository's own site builds, every link in it resolves, and no placeholder is left."""

    def test_build_and_check(self):
        root = HERE.parent
        if not (root / "docs" / "index.md").exists() or not (root / "space" / "results.py").exists():
            self.skipTest("not inside the repository")
        for page in build.PAGES:
            self.assertNotRegex((root / page.source).read_text(encoding="utf-8"), r"\{\{[A-Z_]+\}\}", page.source)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            report = build.Site(root, out).build()
            self.assertEqual(report["pages"], len(build.PAGES))
            result = check.check(out, root)
            self.assertEqual(result["errors"], [])
            for path in out.rglob("*"):
                if path.suffix in (".html", ".md", ".txt"):
                    self.assertNotIn("{{", path.read_text(encoding="utf-8"), path)
            home = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn('class="blk-fig"', home)
            self.assertGreaterEqual(home.count('<a class="card"'), 2)


if __name__ == "__main__":
    unittest.main()
