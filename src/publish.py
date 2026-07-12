"""Git/GitHub publishing: secret-scanned commits and private repo creation.

The repo may go public later, so every commit path is scanned: no secrets, no
weights, no datasets. `--create-remote` tries modern gh syntax first and falls
back to the REST API (the installed gh 2.4.0 predates `--source/--push`)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .paths import ROOT

REPO_NAME = "detection-engineering"

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),          # AWS access key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),       # GitHub PAT
    re.compile(r"(?i)wandb[_-]?api[_-]?key\s*[:=]\s*[0-9a-f]{40}"),
]
FORBIDDEN_STAGED = re.compile(
    r"^(data/|runs/|wandb/|\.venv/|\.env$)|\.(pt|onnx|engine|mlpackage|npy|cache)$"
)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def _scan_staged() -> None:
    """Scan what is actually staged (index blobs, not the worktree — they can differ)."""
    staged = [l for l in _git("diff", "--cached", "--name-only", "--diff-filter=d")
              .stdout.splitlines() if l]
    bad_paths = [p for p in staged if FORBIDDEN_STAGED.search(p)]
    if bad_paths:
        raise SystemExit(f"REFUSING to commit forbidden paths (weights/data/env): {bad_paths}")
    hits = []
    for p in staged:
        raw = subprocess.run(["git", "show", f":{p}"], cwd=ROOT, capture_output=True).stdout
        if len(raw) > 1_000_000:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(f"{p}: {m.group(0)[:24]}…")
    if hits:
        raise SystemExit("REFUSING to commit possible secrets:\n  " + "\n  ".join(hits))
    print(f"secret scan clean ({len(staged)} staged files)")


def _outgoing_blobs() -> list[tuple[str, str]]:
    """Every (blob_sha, path) that pushing main would publish — ALL history in
    origin/main..main, not the endpoint diff. A secret committed and later
    deleted is still in the pushed pack; a clean working copy can hide a dirty
    committed blob, so scanning must read blob contents, never the worktree."""
    rng = "origin/main..main" \
        if _git("rev-parse", "--verify", "origin/main", check=False).returncode == 0 else "main"
    objs = _git("rev-list", "--objects", rng).stdout.splitlines()
    sha_path = [l.split(" ", 1) for l in objs if l.strip()]
    sha_path = [(sp[0], sp[1] if len(sp) > 1 else "") for sp in sha_path]

    batch = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        input="\n".join(sha for sha, _ in sha_path),
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    types = {l.split()[0]: (l.split()[1], int(l.split()[2]))
             for l in batch if len(l.split()) == 3}
    return [(sha, path) for sha, path in sha_path
            if path and types.get(sha, ("", 0))[0] == "blob"
            and types[sha][1] <= 1_000_000]


def _blob_text(sha: str) -> str:
    raw = subprocess.run(["git", "cat-file", "blob", sha],
                         cwd=ROOT, capture_output=True, check=True).stdout
    return raw.decode("utf-8", errors="ignore")


def _scan_outgoing_history() -> None:
    blobs = _outgoing_blobs()
    bad_paths = sorted({p for _, p in blobs if FORBIDDEN_STAGED.search(p)})
    if bad_paths:
        raise SystemExit(f"REFUSING to push history containing forbidden paths: {bad_paths}")
    hits = []
    for sha, path in blobs:
        text = _blob_text(sha)
        for pat in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(f"{path}@{sha[:8]}: {m.group(0)[:24]}…")
    if hits:
        raise SystemExit("REFUSING to push history with possible secrets "
                         "(rewrite history to remove them):\n  " + "\n  ".join(sorted(set(hits))))
    print(f"secret scan clean ({len(blobs)} blobs across outgoing history)")


def _commit(message: str) -> None:
    _git("add", "-A")
    if not _git("diff", "--cached", "--name-only").stdout.strip():
        print("nothing to commit")
        return
    _scan_staged()
    _git("commit", "-m", message + "\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
    print(f"committed: {message}")


def _gh_login() -> str:
    out = subprocess.run(["gh", "api", "user"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["login"]


def _create_remote() -> None:
    if "origin" in _git("remote").stdout.split():
        print(f"remote origin already set: {_git('remote', 'get-url', 'origin').stdout.strip()}")
        return
    login = _gh_login()
    url = f"https://github.com/{login}/{REPO_NAME}.git"
    # no --push here: publishing history is _push's job, behind its secret scan
    modern = subprocess.run(
        ["gh", "repo", "create", REPO_NAME, "--private", "--source=.",
         "--remote=origin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if modern.returncode == 0:
        print(f"created private repo via gh: {url}")
        return
    print("modern gh syntax unavailable, using REST API fallback")
    api = subprocess.run(
        ["gh", "api", "-X", "POST", "/user/repos",
         "-f", f"name={REPO_NAME}", "-F", "private=true"],
        capture_output=True, text=True,
    )
    if api.returncode != 0 and "already exists" not in api.stderr + api.stdout:
        raise SystemExit(f"gh api repo creation failed: {api.stderr.strip()}")
    _git("remote", "add", "origin", url)
    print(f"created private repo: {url}")


def _push() -> None:
    _git("fetch", "origin", check=False)  # so the outgoing range is accurate
    _scan_outgoing_history()
    r = _git("push", "-u", "origin", "main", check=False)
    if r.returncode != 0:
        raise SystemExit(f"push failed: {r.stderr.strip()}")
    print("pushed to origin/main")


def publish(init: bool = False, create_remote: bool = False, push: bool = False,
            message: str | None = None) -> None:
    if not any((init, create_remote, push)):
        raise SystemExit("nothing to do: pass --init, --create-remote, and/or --push")
    if init:
        if not (ROOT / ".git").exists():
            _git("init", "-b", "main")
        _commit(message or "Update experiments")
    if create_remote:
        _create_remote()
    if push:
        _push()
