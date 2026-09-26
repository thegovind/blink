#!/usr/bin/env python3
"""Check every link and anchor in the built site.

    python3 site/check.py _site

A link inside the site (relative, or under the site's own address) must reach a file in the
build, and its #fragment must be an id on that page. A link to this repository on GitHub must
name a file or folder that exists in this checkout. Other outside links are counted, not
fetched. Duplicate ids on a page are errors too.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import posixpath
import urllib.parse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
LINK_ATTRS = {("a", "href"), ("link", "href"), ("script", "src"), ("img", "src"), ("source", "src"),
              ("iframe", "src"), ("use", "href"), ("use", "xlink:href"), ("image", "href")}


class Parsed(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: Counter[str] = Counter()
        self.refs: list[tuple[str, str, int]] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (tag == "a" and name == "name"):
                self.ids[value] += 1
            if (tag, name) in LINK_ATTRS:
                self.refs.append((tag, value, self.getpos()[0]))

    handle_startendtag = handle_starttag


def page_url(rel: str) -> str:
    """The directory a page's relative links start from."""
    return posixpath.dirname(rel) + "/" if "/" in rel else ""


def check(out: Path, root: Path) -> dict:
    site = CONFIG["site_url"]
    base_path = urllib.parse.urlparse(site).path or "/"
    repo_prefix = f"https://github.com/{CONFIG['repo']}/"
    pages = {p.relative_to(out).as_posix(): p for p in sorted(out.rglob("*.html"))}
    parsed = {}
    for rel, path in pages.items():
        doc = Parsed()
        doc.feed(path.read_text(encoding="utf-8"))
        parsed[rel] = doc

    errors: list[str] = []
    counts = Counter()
    hosts: Counter[str] = Counter()
    for rel, doc in parsed.items():
        for ident, n in doc.ids.items():
            if n > 1:
                errors.append(f"{rel}: id {ident!r} appears {n} times")
        for tag, url, line in doc.refs:
            where = f"{rel}:{line}"
            url = url.strip()
            if not url:
                errors.append(f"{where}: empty {tag} link")
                continue
            low = url.lower()
            if low.startswith("javascript:"):
                errors.append(f"{where}: script link {url}")
                continue
            if low.startswith(("mailto:", "data:")):
                counts["other"] += 1
                continue
            if url.startswith(site):
                url = url[len(site):] or "./"
                target_dir = ""
            elif url.startswith(base_path) and not url.startswith("//"):
                url = url[len(base_path):] or "./"
                target_dir = ""
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
                continue
            else:
                target_dir = page_url(rel)
            path, _, frag = url.partition("#")
            path = urllib.parse.unquote(path.split("?", 1)[0])
            counts["internal"] += 1
            if path in ("", "./") and target_dir == page_url(rel) and not url.startswith(("./", "../")):
                target = rel
            else:
                joined = posixpath.normpath(posixpath.join(target_dir, path)) if path else target_dir.rstrip("/")
                if joined.startswith(".."):
                    errors.append(f"{where}: {url} leaves the site")
                    continue
                joined = "" if joined == "." else joined
                full = out / joined
                if full.is_dir():
                    target = posixpath.join(joined, "index.html") if joined else "index.html"
                elif full.is_file():
                    target = joined
                else:
                    errors.append(f"{where}: {url} does not exist")
                    continue
            if frag:
                counts["anchors"] += 1
                frag = urllib.parse.unquote(frag)
                doc = parsed.get(target)
                if doc is None:
                    errors.append(f"{where}: {url} has a fragment but {target} is not a page")
                elif frag not in doc.ids:
                    errors.append(f"{where}: #{frag} is not an id on {target}")
    return {"pages": len(pages), "counts": dict(counts), "hosts": dict(hosts), "errors": errors}


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
    print(f"checked {report['pages']} pages: {c.get('internal', 0)} internal links "
          f"({c.get('anchors', 0)} with anchors), {c.get('repo', 0)} repository links, "
          f"{c.get('external', 0)} outside links not fetched; {len(report['errors'])} problems")
    for line in report["errors"]:
        print("  ", line)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
