"""Tests for the docs site builder and link checker.

    uv run --no-project --with-requirements site/requirements.txt python -m unittest discover -s site -p 'test_*.py'
"""

from __future__ import annotations

import json
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


class Slugs(unittest.TestCase):
    def test_github_slugs(self):
        cases = {
            "Part 1: An LLM from zero": "part-1-an-llm-from-zero",
            "Part 12: Serving, Docker, and the Space": "part-12-serving-docker-and-the-space",
            "Decision Index 0.2 (local run)": "decision-index-02-local-run",
            "request-mixed.json": "request-mixedjson",
            "Full attention, step by step": "full-attention-step-by-step",
            "Kullback–Leibler divergence": "kullbackleibler-divergence",
            "snake_case stays": "snake_case-stays",
        }
        for text, want in cases.items():
            self.assertEqual(build.github_slug(text), want, text)

    def test_repeated_headings_are_numbered(self):
        s = build.Slugger()
        self.assertEqual([s.slug("Usage"), s.slug("Usage"), s.slug("Usage")], ["usage", "usage-1", "usage-2"])

    def test_relative_urls(self):
        self.assertEqual(build.rel_url("", ""), "./")
        self.assertEqual(build.rel_url("", "wire-format/"), "wire-format/")
        self.assertEqual(build.rel_url("overview", ""), "../")
        self.assertEqual(build.rel_url("models/blink-4b", "wire-format/"), "../../wire-format/")


class Strings(unittest.TestCase):
    def test_every_string_is_used(self):
        data = json.loads((HERE / "strings.json").read_text(encoding="utf-8"))
        keys = set(data["draft"]) | set(data["reused"])
        code = (HERE / "build.py").read_text(encoding="utf-8")
        for key in keys - {"readme.docs_line"}:
            used = f'"{key}"' in code or key.startswith("nav.group.")
            self.assertTrue(used, f"{key} is never used")
        self.assertFalse(set(data["draft"]) & set(data["reused"]))

    def test_nav_groups_have_labels(self):
        s = build.load_strings()
        for group in build.GROUPS:
            self.assertIn(f"nav.group.{group}", s)


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class Links(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        for rel, text in {
            "README.md": "# blink\n\n**One line.**\n",
            "LICENSE": "x",
            "docs/WIRE_FORMAT.md": "# Wire format\n",
            "docs/RESULTS.md": "# Results\n",
            "docs/models/blink-4b.md": "# blink-4b\n",
            "docs/models/blink-4b/eval/a.json": "{}",
            "docs/facts/FACTS.md": "# Facts\n",
            "serve.py": "",
            "examples/request-mixed.json": "{}",
        }.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text, encoding="utf-8")
        (root / "docs/assets/blink-4b").mkdir(parents=True)
        (root / "docs/assets/blink-4b/r.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
        (root / "EXPORT_MANIFEST.json").write_text(json.dumps([
            {"src": "release/out/blink-4b/README.md", "dest": "docs/models/blink-4b.md"},
            {"src": "release/serve.py", "dest": "serve.py"},
            {"src": "<generated>", "dest": "LICENSE"},
        ]), encoding="utf-8")
        self.site = build.Site(root, root / "_site")
        self.readme = build.PAGES[0]
        self.card = next(p for p in build.PAGES if p.slug == "models/blink-4b")
        self.guide = next(p for p in build.PAGES if p.slug == "first-principles")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_become_site_links(self):
        self.assertEqual(self.site.resolve(self.readme, "docs/RESULTS.md"), "../results/")
        self.assertEqual(self.site.resolve(self.card, "#licence"), "#licence")
        self.assertEqual(self.site.resolve(self.guide, "../release/out/blink-4b/README.md#use"),
                         "../models/blink-4b/#use")

    def test_repository_files_go_to_github(self):
        repo = f"https://github.com/{build.CONFIG['repo']}"
        self.assertEqual(self.site.resolve(self.readme, "LICENSE"), f"{repo}/blob/main/LICENSE")
        self.assertEqual(self.site.resolve(self.readme, "docs/models"), f"{repo}/tree/main/docs/models")
        self.assertEqual(self.site.resolve(self.guide, "../release/serve.py"), f"{repo}/blob/main/serve.py")
        self.assertEqual(self.site.resolve(self.guide, "facts/FACTS.md"), f"{repo}/blob/main/docs/facts/FACTS.md")
        self.assertEqual(self.site.resolve(self.card, "eval/a.json"), f"{repo}/blob/main/docs/models/blink-4b/eval/a.json")

    def test_card_images_resolve_through_the_alias(self):
        self.assertEqual(self.site.resolve(self.card, "assets/r.png", image=True), "../../docs/assets/blink-4b/r.png")
        self.assertIn("docs/assets/blink-4b/r.png", self.site.assets)

    def test_examples_and_missing_targets(self):
        self.assertEqual(self.site.resolve(self.readme, "examples/request-mixed.json"), "../examples/#request-mixedjson")
        self.assertIsNone(self.site.resolve(self.guide, "../review/private.md"))
        self.assertIsNone(self.site.resolve(self.guide, "../../outside.md"))

    def test_own_address_becomes_relative(self):
        site = build.CONFIG["site_url"]
        self.assertEqual(self.site.resolve(self.readme, site), "../")
        self.assertEqual(self.site.resolve(self.readme, site + "wire-format/#x"), "../wire-format/#x")
        self.assertEqual(self.site.resolve(self.readme, "https://example.org/a"), "https://example.org/a")

    def test_money_is_not_math_and_unlinked_text_stays(self):
        html, env = self.site.render("It costs $5 and $10, not $x^2$. See [notes](../review/private.md).\n", self.guide)
        self.assertEqual(html.count('class="math inline"'), 1)
        self.assertIn("$5 and $10", html)
        self.assertIn("See notes.", html)
        self.assertTrue(env["math"])

    def test_diagrams_fall_back_to_the_browser(self):
        html, env = self.site.render("```mermaid\nflowchart TB\n  A --> B\n```\n", self.guide)
        self.assertIn('<pre class="mermaid">', html)
        self.assertTrue(env["mermaid"])
        self.assertIn("mermaid-config", self.site.head_extra(env))


@unittest.skipUnless(HAVE_MD, "needs site/requirements.txt")
class RealSite(unittest.TestCase):
    """The repository's own site builds and every link in it resolves."""

    def test_build_and_check(self):
        root = HERE.parent
        if not (root / "README.md").exists() or not (root / "space" / "results.py").exists():
            self.skipTest("not inside the repository")
        with tempfile.TemporaryDirectory() as tmp:
            report = build.Site(root, Path(tmp) / "out").build()
            self.assertEqual(report["pages"], len(build.PAGES) + 1)
            result = check.check(Path(tmp) / "out", root)
            self.assertEqual(result["errors"], [])
            home = (Path(tmp) / "out" / "index.html").read_text(encoding="utf-8")
            self.assertIn("blk-answers", home)
            self.assertIn('class="blk-fig"', home)
            self.assertNotIn("{{", home)


if __name__ == "__main__":
    unittest.main()
