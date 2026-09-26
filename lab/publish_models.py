"""Publish one staged model repo: upload -> verify -> squash history -> tag -> verify at the tag. Visibility is untouched.

  HF_TOKEN=... uv run --with huggingface_hub python lab/publish_models.py blink-4b [blink-27b blink-mimo-9b] [--tag v1.0]
  ... publish_models.py blink-4b --update-main "card: figures"   # follow-up commit on main; the tag must already exist and stays put

release/out/<name>/ holds everything except the weights (README, licences, runtime, Dockerfile, eval, weights.sha256);
the weights are already on the Hub and are checked against weights.sha256 by their LFS sha256. Refuses to run if the
tag exists, if a weight in the manifest is missing or differs, or if a staged file does not round-trip byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OWNER = "thegovind"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(api: HfApi, repo: str, staged: Path, revision: str) -> list[str]:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    problems = []
    manifest = {}
    for line in (staged / "weights.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            manifest[name.strip()] = digest
    tmp = Path(tempfile.mkdtemp())
    for name, digest in manifest.items():
        s = remote.get(name)
        if s is None:
            problems.append(f"missing on the Hub: {name}")
            continue
        got = s.lfs.sha256 if s.lfs else sha256_file(Path(hf_hub_download(repo, name, revision=info.sha, cache_dir=tmp)))
        if got != digest:
            problems.append(f"differs from weights.sha256: {name}")
    for p in sorted(x for x in staged.rglob("*") if x.is_file()):
        name = p.relative_to(staged).as_posix()
        if name not in remote:
            problems.append(f"staged file missing on the Hub: {name}")
            continue
        got = sha256_file(Path(hf_hub_download(repo, name, revision=info.sha, cache_dir=tmp)))
        if got != sha256_file(p):
            problems.append(f"staged file differs on the Hub: {name}")
    expected = set(manifest) | {x.relative_to(staged).as_posix() for x in staged.rglob("*") if x.is_file()} | {".gitattributes"}
    for extra in sorted(set(remote) - expected):
        problems.append(f"unexpected file on the Hub: {extra}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--verify-only", action="store_true", help="compare the staged folder with the Hub's main; change nothing")
    ap.add_argument("--update-main", metavar="MESSAGE",
                    help="upload the staged folder to main as one follow-up commit (no squash, no tag), then verify main")
    a = ap.parse_args()
    api = HfApi()
    for name in a.names:
        repo = f"{OWNER}/{name}"
        staged = ROOT / "release" / "out" / name
        assert (staged / "README.md").exists() and (staged / "weights.sha256").exists(), f"stage {name} first"
        if a.verify_only:
            problems = verify(api, repo, staged, "main")
            print(f"{repo}: {'matches main' if not problems else 'differs from main:'}", *problems, sep="\n  ")
            continue
        if a.update_main:
            tags = {t.name: t.target_commit for t in api.list_repo_refs(repo).tags}
            assert a.tag in tags, f"{repo}: {a.tag} missing; publish it first"
            info = api.upload_folder(repo_id=repo, folder_path=str(staged), commit_message=a.update_main)
            problems = verify(api, repo, staged, "main")
            after = {t.name: t.target_commit for t in api.list_repo_refs(repo).tags}
            if after.get(a.tag) != tags[a.tag]:
                problems.append(f"{a.tag} moved: {tags[a.tag]} -> {after.get(a.tag)}")
            if problems:
                sys.exit(f"{repo}: main differs after upload:\n  " + "\n  ".join(problems))
            print(f"{repo}: main = {api.model_info(repo).sha} ({info.oid[:12]}), every file verified; {a.tag} unchanged at {tags[a.tag]}")
            continue
        if a.tag in {t.name for t in api.list_repo_refs(repo).tags}:
            sys.exit(f"{repo}: tag {a.tag} already exists; not re-tagging")
        api.upload_folder(repo_id=repo, folder_path=str(staged), commit_message=f"blink {a.tag}: model card, runtime and licences")
        problems = verify(api, repo, staged, "main")
        if problems:
            sys.exit(f"{repo}: verification failed before squashing:\n  " + "\n  ".join(problems))
        api.super_squash_history(repo_id=repo, commit_message=f"blink {a.tag}")
        # super_squash_history returns nothing and model_info can still report the pre-squash head for a moment;
        # tag only once main's history is the single squash commit, so the tag lands on main's history
        for _ in range(60):
            commits = api.list_repo_commits(repo)
            if len(commits) == 1 and commits[0].title == f"blink {a.tag}":
                break
            time.sleep(2)
        else:
            sys.exit(f"{repo}: squash did not show up on main; not tagging")
        head = commits[0].commit_id
        api.create_tag(repo, tag=a.tag, revision=head, tag_message=f"blink {a.tag}")
        if api.model_info(repo, revision=a.tag).sha != head:
            sys.exit(f"{repo}: {a.tag} does not resolve to main's squash commit {head}")
        problems = verify(api, repo, staged, a.tag)
        if problems:
            sys.exit(f"{repo}: verification failed at {a.tag}:\n  " + "\n  ".join(problems))
        commits = api.list_repo_commits(repo)
        print(f"{repo}: {a.tag} = {head} ({len(commits)} commit), every file verified; private={api.model_info(repo).private}")


if __name__ == "__main__":
    main()
