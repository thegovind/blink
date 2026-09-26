#!/usr/bin/env python3
"""Check every link and anchor in the built site.

    python3 site/check.py _site

A link inside the site (relative, or under the site's own address) must reach a file in the build,
and its #fragment must be an id on that page (for a page's Markdown copy, on the page it copies). That covers pages, redirect stubs (their refresh
target too), the Markdown copies and llms.txt. A link to this repository on GitHub must name a file
or folder in this checkout. Other outside links are counted, not fetched. Duplicate ids are errors.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import posixpath
import re
import urllib.parse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
LINK_ATTRS = {("a", "href"), ("link", "href"), ("script", "src"), ("img", "src"), ("source", "src"),
              ("iframe", "src"), ("use", "href"), ("use", "xlink:href"), ("image", "href")}
MD_LINK = re.compile(r"\]\((<[^>]*>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")
BARE_URL = re.compile(r"(?<![(<\w])https?://[^\s)>\]\"'`]+")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")


class Parsed(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: Counter[str] = Counter()
        self.refs: list[tuple[str, str, int]] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (tag == "a" and name == "name"):
                self.ids[value] += 1
            if (tag, name) in LINK_ATTRS:
                self.refs.append((tag, value, self.getpos()[0]))
        if tag == "meta" and (values.get("http-equiv") or "").lower() == "refresh":
            m = re.search(r"url=(.+)$", values.get("content") or "", re.I)
            if m:
                self.refs.append(("meta refresh", m.group(1).strip().strip("'\""), self.getpos()[0]))

    handle_startendtag = handle_starttag


def text_links(text: str) -> list[tuple[str, int]]:
    """Links in Markdown or llms.txt, outside fenced code."""
    out, fence = [], None
    for n, line in enumerate(text.splitlines(), 1):
        opener = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and opener:
            fence = opener.group(1)
            continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            continue
        line = re.sub(r"`[^`]*`", "", line)
        found = [m.group(1).strip("<>") for m in MD_LINK.finditer(line)]
        found += [m.group(1) for m in AUTOLINK.finditer(line)]
        found += [m.group(0).rstrip(".,;:") for m in BARE_URL.finditer(line)]
        out += [(u, n) for u in dict.fromkeys(found)]
    return out


def html_twin(rel: str) -> str:
    """The page a Markdown copy copies (api.md -> api/index.html, index.md -> index.html), whose ids its
    headings share."""
    stem = rel[:-3] if rel.endswith(".md") else rel
    return "index.html" if stem == "index" else f"{stem}/index.html"


def check(out: Path, root: Path) -> dict:
    site = CONFIG["site_url"]
    base_path = urllib.parse.urlparse(site).path or "/"
    repo_prefix = f"https://github.com/{CONFIG['repo']}/"
    pages = {p.relative_to(out).as_posix(): p for p in sorted(out.rglob("*.html"))}
    texts = {p.relative_to(out).as_posix(): p for p in sorted(out.rglob("*")) if p.suffix in (".md", ".txt")}
    parsed = {}
    for rel, path in pages.items():
        doc = Parsed()
        doc.feed(path.read_text(encoding="utf-8"))
        parsed[rel] = doc

    errors: list[str] = []
    counts = Counter()
    hosts: Counter[str] = Counter()

    def check_url(url: str, rel: str, where: str) -> None:
        url = url.strip()
        low = url.lower()
        if not url:
            errors.append(f"{where}: empty link")
            return
        if low.startswith("javascript:"):
            errors.append(f"{where}: script link {url}")
            return
        if low.startswith(("mailto:", "data:")):
            counts["other"] += 1
            return
        own_dir = posixpath.dirname(rel) + "/" if "/" in rel else ""
        if url.startswith(site):
            url, target_dir = url[len(site):] or "./", ""
        elif url.startswith(base_path) and not url.startswith("//"):
            url, target_dir = url[len(base_path):] or "./", ""
        elif low.startswith(("http://", "https://", "//")):
            if url.startswith(repo_prefix):
                counts["repo"] += 1
                kind, _, rest = url[len(repo_prefix):].partition("/")
                branch, _, path = rest.partition("/")
                path = urllib.parse.unquote(path.split("#")[0])
                if kind not in ("blob", "tree") and path:
                    errors.append(f"{where}: unexpected repository link {url}")
                elif branch != CONFIG["branch"] and path:
                    errors.append(f"{where}: repository link not on {CONFIG['branch']}: {url}")
                elif path and kind == "blob" and not (root / path).is_file():
                    errors.append(f"{where}: {path} is not a file in the repository")
                elif path and kind == "tree" and not (root / path).is_dir():
                    errors.append(f"{where}: {path} is not a folder in the repository")
            else:
                counts["external"] += 1
                hosts[urllib.parse.urlparse(url if not url.startswith("//") else "https:" + url).netloc] += 1
            return
        else:
            target_dir = own_dir
        path, _, frag = url.partition("#")
        path = urllib.parse.unquote(path.split("?", 1)[0])
        counts["internal"] += 1
        if path in ("", "./") and target_dir == own_dir and not url.startswith(("./", "../")):
            target = rel
        else:
            joined = posixpath.normpath(posixpath.join(target_dir, path)) if path else target_dir.rstrip("/")
            if joined.startswith(".."):
                errors.append(f"{where}: {url} leaves the site")
                return
            joined = "" if joined == "." else joined
            full = out / joined
            if full.is_dir():
                target = posixpath.join(joined, "index.html") if joined else "index.html"
                if not (out / target).is_file():
                    errors.append(f"{where}: {url} is a folder without a page")
                    return
            elif full.is_file():
                target = joined
            else:
                errors.append(f"{where}: {url} does not exist")
                return
        if frag:
            counts["anchors"] += 1
            ids_from = target if target in parsed or not target.endswith(".md") else html_twin(target)
            doc = parsed.get(ids_from)
            if doc is None:
                errors.append(f"{where}: {url} has a fragment but {target} is not a page")
            elif urllib.parse.unquote(frag) not in doc.ids:
                errors.append(f"{where}: #{frag} is not an id on {ids_from}")

    for rel, doc in parsed.items():
        for ident, n in doc.ids.items():
            if n > 1:
                errors.append(f"{rel}: id {ident!r} appears {n} times")
        for tag, url, line in doc.refs:
            if tag == "meta refresh":
                counts["redirects"] += 1
            check_url(url, rel, f"{rel}:{line}")
    for rel, path in texts.items():
        for url, line in text_links(path.read_text(encoding="utf-8")):
            counts["text"] += 1
            check_url(url, rel, f"{rel}:{line}")
    return {"pages": len(pages), "texts": len(texts), "counts": dict(counts), "hosts": dict(hosts), "errors": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=HERE.parent / "_site")
    ap.add_argument("--root", type=Path, default=HERE.parent, help="checkout the repository links must exist in")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args(argv)
    report = check(a.out.resolve(), a.root.resolve())
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    c = report["counts"]
    print(f"checked {report['pages']} HTML files and {report['texts']} Markdown or text files "
          f"({c.get('text', 0)} links in those): {c.get('internal', 0)} internal links ({c.get('anchors', 0)} with "
          f"anchors, {c.get('redirects', 0)} redirect targets), {c.get('repo', 0)} repository links, "
          f"{c.get('external', 0)} outside links not fetched; {len(report['errors'])} problems")
    for line in report["errors"]:
        print("  ", line)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
