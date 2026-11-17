"""Compact copies of big external sources, kept in the owner's private Hugging Face dataset repo.

    .venv/bin/python -m src.hf_cache ls [name]                  # what the cache holds
    .venv/bin/python -m src.hf_cache push <local_dir> <name>     # upload a folder as <name>/
    .venv/bin/python -m src.hf_cache pull <name> <local_dir>     # download <name>/ into a folder

Why (backlog G, A17): SoccerTrack v2's game state is ~54 GB of JSON; the compact per-half files built
from it are ~100 MB. A session pulls those instead of re-parsing. The repo is `$MTAH_HF_CACHE`
(`<user>/<repo>`, a private dataset repo), read and written with `$HF_TOKEN`. Both come from the
environment only: never put them in the repo, a run file or a PR body.

**External data only.** push() refuses, before anything is sent:
  - any file under data/mtah2027/, outputs/ or submissions/ (checked on the resolved path, so a
    symlink into them is refused too), or a folder that contains them;
  - any file or name with "bar_ath" in it (also "bar-ath", "barath", any case);
  - a repo that isn't private.
It can't tell what numbers a file holds, so the rule stays: never push anything derived from BAR-ATH.

pull() is idempotent: a file already present with the same size is skipped. Both take a `client`
(default huggingface_hub.HfApi with the token) so the tests run against a fake.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ENV = "MTAH_HF_CACHE"
REPO_TYPE = "dataset"
FORBIDDEN_DIRS = (os.path.join("data", "mtah2027"), "outputs", "submissions")
FORBIDDEN_NAME = re.compile(r"bar[\s_\-]?ath", re.IGNORECASE)
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(/[A-Za-z0-9][A-Za-z0-9._\-]*)*$")


class Refused(ValueError):
    """push/pull refused by the cache's rules (match data, a bad name, a public repo)."""


def repo_id() -> str:
    repo = os.environ.get(REPO_ENV, "").strip()
    if not repo:
        raise Refused("{} is not set (team.md H31)".format(REPO_ENV))
    return repo


def default_client():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


# -- refusals ----------------------------------------------------------------------------------------

def _forbidden_roots(root: str | None = None) -> list[str]:
    root = root or ROOT
    return [os.path.realpath(os.path.join(root, d)) for d in FORBIDDEN_DIRS]


def _inside(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


def check_name(name: str) -> str:
    """A cache folder name like 'soccertrack/compact-v1'. Refuses '..', absolute paths, BAR-ATH."""
    if not NAME_OK.match(name or "") or ".." in name.split("/"):
        raise Refused("bad cache name {!r}: use letters, digits, '._-' and '/'".format(name))
    if FORBIDDEN_NAME.search(name):
        raise Refused("cache name {!r} mentions BAR-ATH: external data only".format(name))
    return name


def check_path(path: str, root: str | None = None) -> str:
    """Refuse a local file or folder that is, contains or points into match data."""
    root = root or ROOT
    real = os.path.realpath(path)
    for bad in _forbidden_roots(root):
        if _inside(real, bad) or _inside(bad, real):
            raise Refused("{} is or holds match data / outputs ({}): external data only".format(
                path, os.path.relpath(bad, os.path.realpath(root))))
    if FORBIDDEN_NAME.search(os.path.basename(os.path.normpath(path))):
        raise Refused("{} mentions BAR-ATH: external data only".format(path))
    return real


def local_files(local_dir: str, root: str | None = None) -> list[str]:
    """Relative paths of the files to push, after every refusal check. Hidden files are skipped."""
    check_path(local_dir, root)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(local_dir)
    out = []
    for folder, dirs, files in os.walk(local_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for d in dirs:  # a symlinked folder is not walked into: check where it points
            if os.path.islink(os.path.join(folder, d)):
                check_path(os.path.join(folder, d), root)
        for f in sorted(files):
            if f.startswith("."):
                continue
            full = os.path.join(folder, f)
            rel = os.path.relpath(full, local_dir)
            check_path(full, root)
            if FORBIDDEN_NAME.search(rel):
                raise Refused("{} mentions BAR-ATH: external data only".format(rel))
            out.append(rel.replace(os.sep, "/"))
    return out


# -- operations --------------------------------------------------------------------------------------

def remote_files(name: str, client=None) -> dict[str, int]:
    """{path relative to name/: size in bytes} of what the cache holds under name/ ({} if nothing)."""
    check_name(name)
    client = client or default_client()
    out = {}
    try:
        entries = client.list_repo_tree(repo_id(), path_in_repo=name, recursive=True,
                                        repo_type=REPO_TYPE)
        for entry in entries:
            size = getattr(entry, "size", None)
            if size is None or not entry.path.startswith(name + "/"):
                continue  # folders, or something outside name/
            out[entry.path[len(name) + 1:]] = int(size)
    except Exception as error:  # huggingface_hub raises EntryNotFoundError (404) for a missing folder
        if type(error).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError") or \
                "404" in str(error):
            return {}
        raise
    return out


def push(local_dir: str, name: str, client=None, message: str | None = None) -> int:
    """Upload every file under local_dir to <name>/ in the cache repo. Returns the number of files.
    Refuses match data, BAR-ATH names and a non-private repo (module docstring)."""
    check_name(name)
    files = local_files(local_dir)
    if not files:
        raise Refused("{} has no files to push".format(local_dir))
    client = client or default_client()
    repo = repo_id()
    if not getattr(client.repo_info(repo, repo_type=REPO_TYPE), "private", False):
        raise Refused("{} is not private: the cache must be a private dataset repo".format(repo))
    client.upload_folder(repo_id=repo, repo_type=REPO_TYPE, folder_path=local_dir, path_in_repo=name,
                         allow_patterns=files, commit_message=message or "cache: {}".format(name))
    return len(files)


def pull(name: str, local_dir: str, client=None) -> int:
    """Download <name>/ from the cache into local_dir (files land at local_dir/<path under name/>).
    Skips files already there with the same size. Returns the number of files downloaded; raises
    FileNotFoundError if the cache holds nothing under name/."""
    check_name(name)
    client = client or default_client()
    wanted = remote_files(name, client)
    if not wanted:
        raise FileNotFoundError("cache has nothing under {}/".format(name))
    os.makedirs(local_dir, exist_ok=True)
    got = 0
    with tempfile.TemporaryDirectory(dir=local_dir, prefix=".pull-") as tmp:
        for rel, size in sorted(wanted.items()):
            dest = os.path.join(local_dir, rel)
            if os.path.isfile(dest) and os.path.getsize(dest) == size:
                continue
            path = client.hf_hub_download(repo_id=repo_id(), repo_type=REPO_TYPE,
                                          filename="{}/{}".format(name, rel), local_dir=tmp)
            if os.path.getsize(path) != size:
                raise IOError("{}: downloaded {} bytes, expected {}".format(rel, os.path.getsize(path),
                                                                            size))
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.move(path, dest)
            got += 1
    return got


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ls")
    p.add_argument("name", nargs="?", default="")
    p = sub.add_parser("push")
    p.add_argument("local_dir")
    p.add_argument("name")
    p = sub.add_parser("pull")
    p.add_argument("name")
    p.add_argument("local_dir")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "ls":
            if args.name:
                files = remote_files(args.name)
                total = sum(files.values())
                print("{}/: {} files, {:.1f} MB".format(args.name, len(files), total / 1e6))
            else:
                client = default_client()
                tops = sorted({e.path.split("/")[0] for e in client.list_repo_tree(
                    repo_id(), repo_type=REPO_TYPE) if not e.path.startswith(".")})
                print("cache folders: {}".format(", ".join(tops) or "(none)"))
        elif args.cmd == "push":
            print("pushed {} files to {}/".format(push(args.local_dir, args.name), args.name))
        else:
            print("pulled {} files from {}/".format(pull(args.name, args.local_dir), args.name))
    except (Refused, FileNotFoundError) as error:
        print("hf_cache: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
