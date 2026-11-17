"""src/hf_cache.py against a fake Hugging Face client: round trip, idempotent pull, every refusal."""

import os
import shutil
from types import SimpleNamespace

import pytest

from src import hf_cache
from src.hf_cache import Refused


class FakeClient:
    """Just enough of huggingface_hub.HfApi: an in-memory dataset repo {path: bytes}."""

    def __init__(self, private=True):
        self.files: dict[str, bytes] = {}
        self.private = private
        self.uploads = 0
        self.downloads = 0

    def repo_info(self, repo_id, repo_type=None):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        self.uploads += 1
        for rel in allow_patterns:
            with open(os.path.join(folder_path, rel), "rb") as handle:
                self.files["{}/{}".format(path_in_repo, rel)] = handle.read()

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, repo_type=None):
        prefix = (path_in_repo + "/") if path_in_repo else ""
        hits = [SimpleNamespace(path=p, size=len(b)) for p, b in self.files.items() if p.startswith(prefix)]
        if path_in_repo and not hits:
            raise type("EntryNotFoundError", (Exception,), {})("404 " + path_in_repo)
        return hits

    def hf_hub_download(self, repo_id, repo_type, filename, local_dir):
        self.downloads += 1
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(self.files[filename])
        return path


@pytest.fixture
def repo(monkeypatch, tmp_path):
    monkeypatch.setenv(hf_cache.REPO_ENV, "someone/cache")
    monkeypatch.setattr(hf_cache, "ROOT", str(tmp_path / "repo"))
    for d in ("data/mtah2027/tactical", "outputs", "submissions", "data/raw/external/x"):
        os.makedirs(tmp_path / "repo" / d)
    return tmp_path / "repo"


def write(path, text="abc"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def test_round_trip_and_idempotent_pull(repo, tmp_path):
    src = repo / "data/raw/external/x"
    write(src / "a.npz", "one")
    write(src / "sub/b.json", "two!")
    write(src / ".hidden", "skip")
    client = FakeClient()
    assert hf_cache.push(str(src), "src/v1", client) == 2
    assert set(client.files) == {"src/v1/a.npz", "src/v1/sub/b.json"}
    assert hf_cache.remote_files("src/v1", client) == {"a.npz": 3, "sub/b.json": 4}

    dest = tmp_path / "out"
    assert hf_cache.pull("src/v1", str(dest), client) == 2
    assert (dest / "sub/b.json").read_text() == "two!"
    assert hf_cache.pull("src/v1", str(dest), client) == 0  # same sizes: nothing re-downloaded
    (dest / "a.npz").write_text("changed size")
    assert hf_cache.pull("src/v1", str(dest), client) == 1
    assert not [p for p in os.listdir(dest) if p.startswith(".pull-")]  # temp folder cleaned up


def test_pull_missing_name(repo, tmp_path):
    with pytest.raises(FileNotFoundError):
        hf_cache.pull("nothing/here", str(tmp_path / "out"), FakeClient())
    assert hf_cache.remote_files("nothing/here", FakeClient()) == {}


@pytest.mark.parametrize("rel", ["data/mtah2027", "data/mtah2027/tactical", "outputs", "submissions",
                                 "data", "."])
def test_refuses_match_data_folders(repo, rel):
    write(repo / rel / "f.txt")
    with pytest.raises(Refused):
        hf_cache.push(str(repo / rel), "x/v1", FakeClient())


def test_refuses_symlink_into_match_data(repo):
    src = repo / "data/raw/external/x"
    write(src / "ok.txt")
    write(repo / "data/mtah2027/tactical/secret.txt")
    os.symlink(repo / "data/mtah2027/tactical/secret.txt", src / "link.txt")
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", FakeClient())
    os.remove(src / "link.txt")
    os.symlink(repo / "data/mtah2027", src / "linkdir")
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", FakeClient())


@pytest.mark.parametrize("bad", ["bar_ath_rows.npz", "BAR-ATH.csv", "sub/barath/x.npz"])
def test_refuses_bar_ath_file_names(repo, bad):
    src = repo / "data/raw/external/x"
    write(src / "ok.txt")
    write(src / bad)
    client = FakeClient()
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", client)
    assert client.uploads == 0


@pytest.mark.parametrize("name", ["bar_ath/v1", "x/Bar-Ath", "../x", "/abs", "a//b", "", "a b"])
def test_refuses_bad_names(repo, name):
    src = repo / "data/raw/external/x"
    write(src / "ok.txt")
    with pytest.raises(Refused):
        hf_cache.push(str(src), name, FakeClient())


def test_refuses_public_repo(repo):
    src = repo / "data/raw/external/x"
    write(src / "ok.txt")
    client = FakeClient(private=False)
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", client)
    assert client.uploads == 0


def test_needs_repo_env(repo, monkeypatch):
    monkeypatch.delenv(hf_cache.REPO_ENV)
    src = repo / "data/raw/external/x"
    write(src / "ok.txt")
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", FakeClient())


def test_empty_folder_refused(repo):
    src = repo / "data/raw/external/x"
    shutil.rmtree(src)
    os.makedirs(src)
    with pytest.raises(Refused):
        hf_cache.push(str(src), "x/v1", FakeClient())
