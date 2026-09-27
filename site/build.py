#!/usr/bin/env python3
"""Build blink's docs site from the Markdown in docs/.

    uv run --no-project --with-requirements site/requirements.txt python site/build.py --out _site
    python3 site/check.py _site
    python3 -m http.server --bind 127.0.0.1 --directory _site 8000

Five pages: docs/index.md (home), api.md, agents.md, models.md and computer-use.md. Each is also served as
Markdown at its path plus .md, for agents; docs/llms.txt is served at the root and
skills/blink/SKILL.md beside it. Addresses that moved get redirect stubs. Every string the site
adds is in site/strings.json; text it reuses verbatim is in site/strings-reused.json; links and
pins are in site/config.json. The home page draws the Space's own one-pass figure
(space/results.py).
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import shutil
import struct
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


@dataclass(frozen=True)
class Page:
    slug: str            # "" is the home page
    source: str          # the Markdown it renders
    nav: str | None      # strings key for its top-bar label


HOME = Page("", "docs/index.md", None)
PAGES = (
    HOME,
    Page("api", "docs/api.md", "nav.api"),
    Page("agents", "docs/agents.md", "nav.agents"),
    Page("models", "docs/models.md", "nav.models"),
    Page("computer-use", "docs/computer-use.md", "nav.cua"),
)
# addresses that moved: old page -> the page that has its content now
REDIRECTS = {
    "wire-format": "api",
    "models/blink-4b": "models",
    "models/blink-mimo-9b": "models",
    "models/blink-27b": "models",
    "overview": "",
    "examples": "",
    "results": "",
    "evaluation": "",
    "history": "",
    "first-principles": "",
    "changelog": "",
    "contributing": "",
    "security": "",
}
# a page gets an on-page contents list when it has this many sections
TOC_MIN = 4
# repository files served as they are, at these site paths
SITE_FILES = {"docs/llms.txt": "llms.txt", "skills/blink/SKILL.md": "skills/blink/SKILL.md"}
# of those, the ones a link should reach on the site rather than on GitHub
LINKED_ON_SITE = {"docs/llms.txt"}


class BuildError(RuntimeError):
    """The sources cannot produce a site worth publishing."""


# --- strings and small helpers -----------------------------------------------------------


def load_strings() -> dict[str, str]:
    """strings.json: every string the site adds. strings-reused.json: text copied verbatim from its source."""
    own = json.loads((HERE / "strings.json").read_text(encoding="utf-8"))
    reused = json.loads((HERE / "strings-reused.json").read_text(encoding="utf-8"))["strings"]
    out = dict(own)
    for key, item in reused.items():
        if key in out:
            raise BuildError(f"{key} is in both strings.json and strings-reused.json")
        out[key] = item["text"]
    return out


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def github_slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation and symbols dropped, spaces to hyphens."""
    kept = [ch for ch in text.lower()
            if ch in " -_" or ch.isalnum() or unicodedata.category(ch).startswith("M")]
    return "".join(kept).replace(" ", "-")


class Slugger:
    """GitHub's duplicate rule: the second "Usage" is "usage-1"."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = slug = github_slug(text)
        while slug in self.seen:
            self.seen[base] += 1
            slug = f"{base}-{self.seen[base]}"
        self.seen[slug] = 0
        return slug


def rel_url(from_slug: str, target: str) -> str:
    """A link from the page at <from_slug>/index.html to a path relative to the site root."""
    depth = len([p for p in from_slug.split("/") if p])
    return ("../" * depth + target) or "./"


def page_path(slug: str) -> str:
    return slug + "/" if slug else ""


def markdown_path(slug: str) -> str:
    """Where a page's Markdown copy is served: /api.md for /api/, /index.md for the home page."""
    return (slug or "index") + ".md"


def image_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()[:4096]
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if path.suffix.lower() == ".svg":
        text = data.decode("utf-8", "replace")
        m = re.search(r"<svg\b[^>]*>", text)
        if m:
            w = re.search(r'\bwidth="([\d.]+)"', m.group(0))
            h = re.search(r'\bheight="([\d.]+)"', m.group(0))
            if w and h:
                return round(float(w.group(1))), round(float(h.group(1)))
    return None


# --- markdown ----------------------------------------------------------------------------


def markdown():
    from markdown_it import MarkdownIt
    from mdit_py_plugins.front_matter import front_matter_plugin

    md = MarkdownIt("commonmark", {"html": True, "typographer": False})
    md.enable(["table", "strikethrough"])
    md.use(front_matter_plugin)
    return md


def inline_text(tok) -> str:
    parts = []
    for c in tok.children or []:
        if c.type in ("text", "code_inline"):
            parts.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif c.type == "image":
            parts.append(c.content)
    return "".join(parts)


def new_env() -> dict:
    return {"toc": [], "slugger": Slugger()}


class Site:
    def __init__(self, root: Path, out: Path, site_dir: Path = HERE) -> None:
        self.root = root.resolve()
        self.out = out.resolve()
        self.site_dir = site_dir
        self.s = load_strings()
        self.md = markdown()
        self.by_source = {p.source: p for p in PAGES}
        self.assets: set[str] = set()
        self.repo_url = f"https://github.com/{CONFIG['repo']}"
        self._rules()

    # --- links ---------------------------------------------------------------------------

    def github(self, path: str, directory: bool = False) -> str:
        kind = "tree" if directory else "blob"
        return f"{self.repo_url}/{kind}/{CONFIG['branch']}/{urllib.parse.quote(path)}"

    def site_target(self, path: str) -> str | None:
        """A site address named without the repository (e.g. "/api/"), as a path from the site root."""
        bare = path.strip("/")
        if bare in {p.slug for p in PAGES} or bare in REDIRECTS:
            return page_path(bare)
        if bare in set(SITE_FILES.values()) or bare in {markdown_path(p.slug) for p in PAGES}:
            return bare
        return None

    def locate(self, page: Page, raw: str) -> tuple[str, str] | None:
        """What a relative link names: ("site", path from the site root), ("file", repo path) or ("dir", repo path)."""
        if raw.startswith("/"):
            found = self.site_target(raw)
            if found is not None:
                return "site", found
            cand = raw.lstrip("/")
        else:
            cand = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), raw))
        cand = cand.rstrip("/")
        if cand in ("", ".") and not raw.startswith("/"):
            cand = ""
        if cand == ".." or cand.startswith("../"):
            return None
        if cand in self.by_source:
            return "site", page_path(self.by_source[cand].slug)
        if cand in LINKED_ON_SITE:
            return "site", SITE_FILES[cand]
        full = self.root / cand
        if cand and full.is_file():
            return "file", cand
        if full.is_dir():
            return "dir", cand
        return None

    def resolve(self, page: Page, href: str, image: bool = False) -> str:
        """A Markdown link as the page at `page` should write it. A link to nothing public is an error."""
        if not href or href.startswith("#"):
            return href
        site = CONFIG["site_url"]
        if href.startswith(site):
            path, _, frag = href[len(site):].partition("#")
            return rel_url(page.slug, path) + (f"#{frag}" if frag else "")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) or href.startswith("//"):
            return href
        raw, _, frag = href.partition("#")
        anchor = f"#{frag}" if frag else ""
        found = self.locate(page, urllib.parse.unquote(raw.split("?", 1)[0]))
        if found is None:
            raise BuildError(f"{page.source}: link {href!r} names nothing in this repository or on this site")
        kind, value = found
        if kind == "site":
            return rel_url(page.slug, value) + anchor
        if kind == "file" and image and Path(value).suffix.lower() in IMAGE_SUFFIXES:
            self.assets.add(value)
            return rel_url(page.slug, value)
        return self.github(value, directory=kind == "dir") + (anchor if kind == "file" else "")

    def absolute(self, page: Page, href: str, image: bool = False) -> str:
        """The same link as a full address, for the Markdown copies, which are read out of context. A link to a
        page, or to an address that moved, goes to that page's Markdown copy."""
        site = CONFIG["site_url"]
        url = urllib.parse.urljoin(site + page_path(page.slug), self.resolve(page, href, image=image))
        if not url.startswith(site):
            return url
        path, mark, frag = url[len(site):].partition("#")
        if path and not path.endswith("/"):
            return url
        slug = path.strip("/")
        slug = REDIRECTS.get(slug, slug)
        if slug in {p.slug for p in PAGES}:
            return site + markdown_path(slug) + mark + frag
        return url

    # --- rendering -----------------------------------------------------------------------

    def _rules(self) -> None:
        md = self.md
        plain_code = md.renderer.rules["code_inline"]

        def fence(renderer, tokens, idx, options, env):
            return code_html(tokens[idx].content, fence_lang(tokens[idx]))

        def code_block(renderer, tokens, idx, options, env):
            return code_html(tokens[idx].content, "")

        def code_inline(renderer, tokens, idx, options, env):
            if tokens[idx].meta.get("cell"):
                return cell_code(tokens[idx].content)
            return plain_code(tokens, idx, options, env)

        def heading_close(renderer, tokens, idx, options, env):
            tag = tokens[idx].tag
            hid = tokens[idx].meta.get("id")
            mark = (f'<a class="anchor" href="#{esc(hid)}" aria-hidden="true" tabindex="-1">#</a>'
                    if hid and tag != "h1" else "")
            return f"{mark}</{tag}>\n"

        def table_open(renderer, tokens, idx, options, env):
            return '<div class="table-wrap"><table>\n'

        def table_close(renderer, tokens, idx, options, env):
            return "</table></div>\n"

        md.add_render_rule("fence", fence)
        md.add_render_rule("code_block", code_block)
        md.add_render_rule("code_inline", code_inline)
        md.add_render_rule("heading_close", heading_close)
        md.add_render_rule("table_open", table_open)
        md.add_render_rule("table_close", table_close)

    def prepare(self, tokens, page: Page, env: dict) -> None:
        in_cell = False
        for i, tok in enumerate(tokens):
            if tok.type in ("td_open", "th_open"):
                in_cell = True
            elif tok.type in ("td_close", "th_close"):
                in_cell = False
            if tok.type == "heading_open":
                text = inline_text(tokens[i + 1]).strip()
                hid = env["slugger"].slug(text)
                tok.attrSet("id", hid)
                tokens[i + 2].meta["id"] = hid
                level = int(tok.tag[1])
                if level == 1 and not env.get("title"):
                    env["title"] = text
                elif level in (2, 3):
                    env["toc"].append((level, text, hid))
            elif tok.type == "paragraph_open" and env.get("title") and "summary" not in env:
                env["summary"] = inline_text(tokens[i + 1]).strip()
            if tok.type != "inline" or not tok.children:
                continue
            for c in tok.children:
                if c.type == "code_inline" and in_cell:
                    c.meta["cell"] = True
                elif c.type == "link_open":
                    c.attrSet("href", self.resolve(page, c.attrGet("href") or ""))
                elif c.type == "image":
                    src = c.attrGet("src") or ""
                    c.attrSet("src", self.resolve(page, src, image=True))
                    c.attrSet("loading", "lazy")
                    c.attrSet("decoding", "async")
                    found = None if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", src) else self.locate(page, src.split("#")[0])
                    size = image_size(self.root / found[1]) if found and found[0] == "file" else None
                    if size:
                        c.attrSet("width", str(size[0]))
                        c.attrSet("height", str(size[1]))

    def parse(self, page: Page) -> tuple[list, dict]:
        env = new_env()
        tokens = self.md.parse(self.page_text(page), env)
        self.prepare(tokens, page, env)
        return group_code(tokens, env), env

    def render(self, tokens, env: dict) -> str:
        return self.md.renderer.render(tokens, self.md.options, env)

    def page_text(self, page: Page) -> str:
        return (self.root / page.source).read_text(encoding="utf-8")

    def copy_assets(self) -> None:
        for rel in sorted(self.assets):
            dest = self.out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, dest)

    # --- the Markdown copies ---------------------------------------------------------------

    def markdown_copy(self, page: Page) -> str:
        """The page's own Markdown, with every link made a full address, under a pointer to llms.txt."""
        text = self.page_text(page)
        normalize = self.md.normalizeLink
        images = {c.attrGet("src") for tok in self.md.parse(text, new_env()) for c in tok.children or []
                  if c.type == "image"}

        def swap(m: re.Match) -> str:
            raw = m.group(2)
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            href = normalize(raw)
            return f"{m.group(1)}{self.absolute(page, href, image=href in images)}{m.group(3)}"

        inline = re.compile(r'(\]\()(<[^>]*>|[^)\s]+)((?:\s+"[^"]*")?\))')
        definition = re.compile(r'^( {0,3}\[[^\]]+\]:\s*)(<[^>]*>|\S+)(.*)$')
        out, fence = [], None
        for line in text.splitlines(keepends=True):
            opener = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence is None and opener:
                fence = opener.group(1)[0] * len(opener.group(1))
            elif fence is not None and line.strip().startswith(fence):
                fence = None
            elif fence is None:
                line = inline.sub(swap, line)
                line = definition.sub(swap, line)
            out.append(line)
        header = f'> {self.s["md.header"]} {CONFIG["site_url"]}llms.txt\n\n'
        return header + "".join(out)

    # --- chrome --------------------------------------------------------------------------

    def top_links(self, current: Page | None) -> str:
        items = []
        for p in PAGES:
            if p.nav:
                on = ' aria-current="page"' if current is not None and p.slug == current.slug else ""
                target = self.link_root(current) + page_path(p.slug)
                items.append(f'<a href="{esc(target)}"{on}>{esc(self.s[p.nav])}</a>')
        items.append(f'<a class="wide" href="{esc(CONFIG["space_url"])}">{esc(self.s["nav.space"])}</a>')
        items.append(f'<a href="{esc(self.repo_url)}">{esc(self.s["site.github"])}</a>')
        return "".join(items)

    def link_root(self, page: Page | None) -> str:
        if page is None:  # the 404 page is served at any depth
            return urllib.parse.urlparse(CONFIG["site_url"]).path or "/"
        root = rel_url(page.slug, "")
        return "" if root == "./" else root

    def toc(self, entries) -> str:
        items = "".join(f'<li class="l{lvl}"><a href="#{esc(hid)}">{esc(text)}</a></li>'
                        for lvl, text, hid in entries)
        return f"<ul>{items}</ul>"

    def shell(self, page: Page | None, title: str, description: str, body: str, body_class: str) -> str:
        root = self.link_root(page)
        site = CONFIG["site_url"]
        if page is not None:
            canonical = esc(site + page_path(page.slug))
            meta = (f'<link rel="canonical" href="{canonical}">\n'
                    f'<link rel="alternate" type="text/markdown" href="{esc(root + markdown_path(page.slug))}">\n'
                    '<meta property="og:type" content="website">\n'
                    f'<meta property="og:title" content="{esc(title)}">\n'
                    f'<meta property="og:description" content="{esc(description)}">\n'
                    f'<meta property="og:url" content="{canonical}">\n'
                    '<meta name="twitter:card" content="summary">')
        else:
            meta = '<meta name="robots" content="noindex">'
        return fill(template("base.html"), {
            "title": esc(title),
            "description": esc(description),
            "meta": meta,
            "root": root,
            "home": root or "./",
            "fonts_css": esc(CONFIG["fonts_css"]),
            "body_class": body_class,
            "copy": esc(self.s["code.copy"]),
            "copied": esc(self.s["code.copied"]),
            "skip": esc(self.s["site.skip"]),
            "nav_aria": esc(self.s["nav.aria"]),
            "top_links": self.top_links(page),
            "github": esc(self.s["site.github"]),
            "github_url": esc(self.repo_url),
            "space": esc(self.s["nav.space"]),
            "space_url": esc(CONFIG["space_url"]),
            "hub": esc(self.s["home.model_hub"]),
            "hub_url": esc(CONFIG["hub_url"] + CONFIG["repo"].split("/")[0]),
            "footer_note": esc(self.s["footer.note"]),
            "body": body,
        })

    # --- pages ---------------------------------------------------------------------------

    def docs_page(self, page: Page) -> str:
        tokens, env = self.parse(page)
        content = self.render(tokens, env)
        title = env.get("title") or self.s.get(page.nav or "", page.slug)
        toc = self.toc(env["toc"]) if len(env["toc"]) >= TOC_MIN else ""
        if toc:
            mobile = (f'<details class="tocm"><summary>{esc(self.s["toc.title"])}</summary>'
                      f'<nav aria-label="{esc(self.s["toc.title"])}">{toc}</nav></details>')
            content = content.replace("</h1>\n", "</h1>\n" + mobile, 1) if "</h1>" in content else mobile + content
            aside = (f'<aside class="toc"><nav aria-label="{esc(self.s["toc.title"])}">'
                     f'<p class="toc-title">{esc(self.s["toc.title"])}</p>{toc}</nav></aside>')
        else:
            aside = ""
        body = fill(template("docs.html"), {
            "layout": "page" if toc else "page solo",
            "content": content,
            "toc": aside,
            "md_url": esc(rel_url(page.slug, markdown_path(page.slug))),
            "markdown": esc(self.s["page.markdown"]),
            "source_url": esc(self.github(page.source)),
            "source": esc(self.s["page.source"]),
        })
        return self.shell(page, f"{title} · blink", env.get("summary") or self.pitch, body, "docs-page")

    def home_page(self) -> str:
        """docs/index.md, drawn as the home page: its title and first paragraph are the hero, its first list of
        links is the cards, and everything else follows the Space's one-pass figure."""
        tokens, env = self.parse(HOME)
        h1 = next((i for i, t in enumerate(tokens) if t.type == "heading_open" and t.tag == "h1"), None)
        if h1 is None:
            raise BuildError(f"{HOME.source} needs a title")
        # the lede is the paragraph straight after the title, if that is what follows it
        after = next((i for i in range(h1 + 3, len(tokens)) if tokens[i].level == 0 and tokens[i].nesting >= 0), None)
        lede = after if after is not None and tokens[after].type == "paragraph_open" else None
        hero_end = (lede + 2) if lede is not None else (h1 + 2)
        title = inline_text(tokens[h1 + 1]).strip()
        lede_html = (self.md.renderer.renderInline(tokens[lede + 1].children, self.md.options, env)
                     if lede is not None else "")
        cards, span = self.cards(tokens, hero_end + 1, env)
        rest = [t for i, t in enumerate(tokens)
                if not (h1 <= i <= hero_end) and not (span and span[0] <= i <= span[1])]
        pieces = space_pieces(self.root)
        steps, notes = pieces["stages"]
        hub = CONFIG["hub_url"] + CONFIG["repo"].split("/")[0] + "/"
        chips = "".join(f'<a class="chip key" href="{esc(hub + m)}">{esc(CONFIG["repo"].split("/")[0] + "/" + m)}</a>'
                        for m in CONFIG["models"])
        chips += f'<span class="chip flash">{esc(self.s["hero.chip"])}</span>'
        word = ('<span class="word">blink<em class="dot" aria-hidden="true"></em></span>' if title == "blink"
                else esc(title))
        body = fill(template("home.html"), {
            "title": word,
            "lede": lede_html,
            "chips": chips,
            "cards": cards,
            "figure": pieces["figure"],
            "stages": "".join(f'<li><span class="n">{i}</span><b>{esc(t)}</b><span>{esc(sub)}</span></li>'
                              for i, (t, sub) in enumerate(steps, 1)),
            "stage_notes": "".join(f"<p>{esc(n)}</p>" for n in notes),
            "rest": self.render(rest, env),
        })
        return self.shell(HOME, "blink", self.pitch, body, "home-page")

    def cards(self, tokens, start: int, env: dict) -> tuple[str, tuple[int, int] | None]:
        """The home page's cards: the first top-level list whose every item opens with a link, or the first run of
        two or more sections that are a heading and one paragraph with a link, whichever comes first.
        Returns the cards and the token span they replace, or ("", None)."""
        for i in range(start, len(tokens)):
            if tokens[i].level != 0:
                continue
            found = list_cards(tokens, i) or section_cards(tokens, i)
            if found:
                items, span = found
                break
        else:
            return "", None
        render = self.md.renderer.renderInline
        html_cards = []
        for href, title_kids, desc_kids in items:
            title = render(title_kids, self.md.options, env)
            desc = render(desc_kids, self.md.options, env)
            html_cards.append(f'<a class="card" href="{esc(href)}"><b>{title}</b><span>{desc}</span>'
                              '<em aria-hidden="true">\u2192</em></a>')
        return f'<div class="cards">{"".join(html_cards)}</div>', span

    def redirect_stub(self, old: str, new: str, titles: dict[str, str]) -> str:
        target = rel_url(old, page_path(new))
        title = "blink" if not new else f"{titles.get(new, new)} · blink"
        return fill(template("redirect.html"), {
            "title": esc(title),
            "canonical": esc(CONFIG["site_url"] + page_path(new)),
            "target": esc(target),
            "target_js": json.dumps(target),
            "note": esc(self.s["redirect.note"]),
        })

    def not_found(self) -> str:
        body = fill(template("404.html"), {
            "title": esc(self.s["notfound.title"]),
            "home": esc(self.link_root(None)),
            "home_label": esc(self.s["notfound.home"]),
        })
        return self.shell(None, f'{self.s["notfound.title"]} · blink', self.pitch, body, "plain-page")

    # --- output --------------------------------------------------------------------------

    def write(self, rel: str, text: str) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def build(self) -> dict:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        home_tokens, home_env = self.parse(HOME)
        self.pitch = home_env.get("summary") or "blink"
        titles = {}
        for page in PAGES:
            _, env = self.parse(page)
            titles[page.slug] = env.get("title") or page.slug
        for page in PAGES:
            html_page = self.home_page() if page is HOME else self.docs_page(page)
            self.write(f"{page_path(page.slug)}index.html", html_page)
            self.write(markdown_path(page.slug), self.markdown_copy(page))
        for old, new in REDIRECTS.items():
            if old in {p.slug for p in PAGES}:
                raise BuildError(f"{old} is both a page and a redirect")
            self.write(f"{old}/index.html", self.redirect_stub(old, new, titles))
        for src, dest in SITE_FILES.items():
            (self.out / dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / src, self.out / dest)
        self.write("404.html", self.not_found())
        self.copy_assets()
        for name in ("site.css", "site.js", "favicon.svg"):
            dest = self.out / "assets" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.site_dir / "static" / name, dest)
        self.write(".nojekyll", "")
        self.write("sitemap.xml", sitemap([page_path(p.slug) for p in PAGES]))
        return {"pages": len(PAGES), "markdown": len(PAGES), "redirects": len(REDIRECTS),
                "files": len(SITE_FILES), "images": len(self.assets)}


def list_cards(tokens, i: int):
    """A top-level list at tokens[i] whose items each open with a link: ([(href, title, description)], span)."""
    if tokens[i].type != "bullet_list_open":
        return None
    j = next(k for k in range(i + 1, len(tokens)) if tokens[k].type == "bullet_list_close" and tokens[k].level == 0)
    items = [card_parts(t.children or []) for t in tokens[i:j] if t.type == "inline" and t.level == 3]
    if not items or any(item is None for item in items):
        return None
    return items, (i, j)


def section_cards(tokens, i: int):
    """Two or more sections from tokens[i] on, each a heading and one paragraph that holds a link. A card's link
    is the paragraph's first link; the paragraph's links become plain text, since the card is the link."""
    items, k, level = [], i, None
    while (k + 5 < len(tokens) and tokens[k].type == "heading_open" and tokens[k].level == 0
           and tokens[k + 3].type == "paragraph_open" and tokens[k + 5].type == "paragraph_close"
           and (k + 6 == len(tokens) or tokens[k + 6].type == "heading_open")):
        if level is None:
            level = tokens[k].tag
        if tokens[k].tag != level:
            break
        kids = [c for c in tokens[k + 4].children or [] if not (c.type == "text" and not c.content)]
        link = next((c for c in kids if c.type == "link_open"), None)
        if link is None:
            break
        desc = [c for c in kids if c.type not in ("link_open", "link_close")]
        title = [c for c in tokens[k + 1].children or [] if c.type not in ("link_open", "link_close")]
        items.append((link.attrGet("href"), title, desc))
        k += 6
    if len(items) < 2:
        return None
    return items, (i, k - 1)


def card_parts(kids):
    """(href, title children, description children) for a list item that opens with a link, else None."""
    kids = [k for k in kids if not (k.type == "text" and not k.content)]
    if not kids or kids[0].type != "link_open":
        return None
    close = next((i for i, k in enumerate(kids) if k.type == "link_close"), None)
    if close is None:
        return None
    rest = [k for k in kids[close + 1:] if k.type not in ("link_open", "link_close")]
    if rest and rest[0].type == "text":
        rest[0].content = re.sub(r"^\s*(?:[:\u2014\u2013-]\s*)?", "", rest[0].content)
    return kids[0].attrGet("href"), kids[1:close], rest


def space_pieces(root: Path) -> dict:
    """The Space's one-pass figure, drawn by its own code (space/results.py, which needs only the standard
    library), loaded from this checkout under a private name."""
    import importlib.util

    path = (root / "space" / "results.py").resolve()
    if not path.is_file():
        raise BuildError(f"the home page draws its figure with {path}, which is missing")
    name = "_blink_site_results"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        figure = module.pipeline_figure()
    finally:
        sys.modules.pop(name, None)
    return {"figure": figure, "stages": pipeline_stages(figure)}


def pipeline_stages(figure: str) -> tuple[list[tuple[str, str]], list[str]]:
    """The Space figure's stages as (title, line) pairs and its closing notes, for a phone."""
    stages = []
    for g in re.findall(r'<g class="blk-stage".*?</g>', figure, re.S):
        title = re.search(r'class="blk-stage-t"[^>]*>(.*?)</text>', g, re.S)
        lines = re.findall(r'class="blk-stage-s"[^>]*>(.*?)</text>', g, re.S)
        if not title:
            raise BuildError("space/results.py pipeline_figure changed shape")
        stages.append((html.unescape(title.group(1)), " ".join(html.unescape(x) for x in lines)))
    notes = [html.unescape(x) for x in re.findall(r"<text[^>]*>(.*?)</text>", figure[figure.rfind("</g>"):], re.S)]
    if len(stages) < 2:
        raise BuildError("space/results.py pipeline_figure changed shape")
    return stages, notes


# a table cell's code may wrap before a '/' (not one that opens it, or one of '//')
SLASH_BREAK = re.compile(r"(?<=[^/\s])(?=/[^/\s])")


def cell_code(content: str) -> str:
    """Inline code in a table cell: whole segments that wrap only at a space or just before a '/', never inside a
    token; where a segment is still too wide, the table scrolls."""
    words = []
    for word in content.split(" "):
        words.append("<wbr>".join(f"<span>{esc(seg)}</span>" for seg in SLASH_BREAK.split(word) if seg))
    return f'<code class="cell">{" ".join(words)}</code>'


def fence_lang(tok) -> str:
    return tok.info.strip().split()[0] if tok.info.strip() else ""


def group_code(tokens, env: dict) -> list:
    """Adjacent code blocks in different languages (one call in Python, JavaScript and shell, say) become one block
    with a tab for each language; the tabs are named by the section's heading."""
    from markdown_it.token import Token

    out, i, heading = [], 0, None
    while i < len(tokens):
        if tokens[i].type == "heading_open":
            heading = tokens[i].attrGet("id")
        j = i
        while j < len(tokens) and tokens[j].type == "fence" and tokens[j].level == 0:
            j += 1
        if j == i:
            out.append(tokens[i])
            i += 1
            continue
        run = tokens[i:j]
        langs = [fence_lang(tok) for tok in run]
        if len(run) >= 2 and all(langs) and len(set(langs)) == len(langs):
            env["tabs"] = env.get("tabs", 0) + 1
            block = Token("html_block", "", 0)
            block.block = True
            block.content = code_tabs_html([tok.content for tok in run], langs, env["tabs"], heading)
            out.append(block)
        else:
            out.extend(run)
        i = j
    return out


def code_tabs_html(codes: list[str], langs: list[str], n: int, heading: str | None) -> str:
    """One code block, a tab per language. Without scripts every panel shows, each under its language."""
    tabs, panels = [], []
    for k, (code, lang) in enumerate(zip(codes, langs)):
        tab, panel, on = f"code-{n}-{k}", f"code-{n}-{k}-panel", k == 0
        rest = "" if on else ' tabindex="-1"'
        tabs.append(f'<button type="button" role="tab" class="tab" id="{tab}" aria-controls="{panel}" '
                    f'aria-selected="{"true" if on else "false"}"{rest}>{esc(lang)}</button>')
        panels.append(f'<div class="code-panel{" on" if on else ""}" role="tabpanel" id="{panel}" '
                      f'aria-labelledby="{tab}" data-lang="{esc(lang)}">'
                      f'<pre><code class="language-{esc(lang)}">{esc(code)}</code></pre></div>')
    label = f' aria-labelledby="{esc(heading)}"' if heading else ""
    return (f'<div class="code code-tabs"><div class="code-head"><div class="tabs" role="tablist"{label}>'
            f'{"".join(tabs)}</div></div>{"".join(panels)}</div>\n')


def code_html(code: str, lang: str) -> str:
    cls = f' class="language-{esc(lang)}"' if lang else ""
    head = f'<div class="code-head"><span class="code-lang">{esc(lang)}</span></div>'
    return f'<div class="code">{head}<pre><code{cls}>{esc(code)}</code></pre></div>\n'


_TEMPLATES: dict[str, str] = {}


def template(name: str) -> str:
    if name not in _TEMPLATES:
        _TEMPLATES[name] = (HERE / "templates" / name).read_text(encoding="utf-8")
    return _TEMPLATES[name]


def fill(text: str, values: dict[str, str]) -> str:
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise BuildError(f"template placeholder {{{{{key}}}}} has no value")
        return values[key]
    return re.sub(r"\{\{(\w+)\}\}", sub, text)


def sitemap(paths: list[str]) -> str:
    site = CONFIG["site_url"]
    urls = "".join(f"<url><loc>{esc(site + p)}</loc></url>" for p in paths)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root (default: this checkout)")
    ap.add_argument("--out", type=Path, default=DEFAULT_ROOT / "_site")
    a = ap.parse_args(argv)
    try:
        report = Site(a.root, a.out).build()
    except BuildError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2
    print(f"built {report['pages']} pages with {report['markdown']} Markdown copies, {report['redirects']} redirects, "
          f"{report['files']} files and {report['images']} images into {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
