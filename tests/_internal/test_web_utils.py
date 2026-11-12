# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the dataset download utilities (web_utils).

neuraloperator has no dedicated test for these helpers (they are borrowed from
torchvision). Coverage exercises everything reachable without network access:

* md5 / integrity helpers against real temp files,
* ``download_from_zenodo_record`` short-circuiting on an empty file list,
* ``download_from_url`` reusing an already-present, integrity-verified file.

Network-dependent branches (actual HTTP downloads) are driven through a
monkeypatched ``requests`` so no real traffic occurs; the genuinely
network-only path is covered by a gated, skipped test.
"""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from scirex._internal import web_utils
from scirex._internal.web_utils import (
    UnsafeTarMemberError,
    _extract_tar_safely,
    calculate_md5,
    check_integrity,
    check_md5,
    download_from_url,
    download_from_zenodo_record,
)

# ---------------------------------------------------------------------------
# md5 / integrity helpers
# ---------------------------------------------------------------------------


def _write(path, content: bytes):
    path.write_bytes(content)
    return path


def test_calculate_md5_matches_hashlib(tmp_path) -> None:
    """calculate_md5 equals hashlib's md5 over the same bytes."""
    content = b"ScireX mesh data"
    f = _write(tmp_path / "f.bin", content)
    expected = hashlib.md5(content).hexdigest()
    assert calculate_md5(f) == expected


def test_check_md5_true_and_false(tmp_path) -> None:
    """check_md5 accepts the correct digest and rejects a wrong one."""
    content = b"hello"
    f = _write(tmp_path / "f.bin", content)
    good = hashlib.md5(content).hexdigest()
    assert check_md5(f, good)
    assert not check_md5(f, "0" * 32)


def test_check_integrity_missing_file(tmp_path) -> None:
    """A missing file fails the integrity check."""
    assert not check_integrity(tmp_path / "nope.bin")


def test_check_integrity_no_md5_only_existence(tmp_path) -> None:
    """With md5=None, integrity is just file existence."""
    f = _write(tmp_path / "f.bin", b"x")
    assert check_integrity(f)
    assert check_integrity(f, md5=None)


def test_check_integrity_with_matching_md5(tmp_path) -> None:
    """Integrity passes when the provided md5 matches the file."""
    content = b"payload"
    f = _write(tmp_path / "f.bin", content)
    assert check_integrity(f, hashlib.md5(content).hexdigest())


def test_check_integrity_accepts_str_path(tmp_path) -> None:
    """A str fpath is converted to Path before the existence check."""
    f = _write(tmp_path / "f.bin", b"x")
    assert check_integrity(str(f)) is True
    assert check_integrity(str(tmp_path / "missing.bin")) is False


# ---------------------------------------------------------------------------
# download_from_zenodo_record - offline branches
# ---------------------------------------------------------------------------


def test_zenodo_empty_file_list_is_noop(monkeypatch) -> None:
    """An empty files_to_download list returns before any network call."""

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("requests.get should not be called for empty list")

    monkeypatch.setattr(web_utils.requests, "get", _boom)
    # Should simply return without raising.
    download_from_zenodo_record(record_id="123", root="/tmp", files_to_download=[])


def test_zenodo_downloads_only_requested_files(monkeypatch, tmp_path) -> None:
    """Only files named in files_to_download are passed to the downloader."""
    record = {
        "files": [
            {
                "key": "darcy_16.tgz",
                "links": {"self": "http://example/darcy_16.tgz"},
                "checksum": "md5:" + "a" * 32,
                "size": 10,
            },
            {
                "key": "darcy_32.tgz",
                "links": {"self": "http://example/darcy_32.tgz"},
                "checksum": "md5:" + "b" * 32,
                "size": 20,
            },
        ]
    }

    class _Resp:
        status_code = 200

        def json(self):
            return record

    monkeypatch.setattr(web_utils.requests, "get", lambda url, **kw: _Resp())

    downloaded = []
    monkeypatch.setattr(
        web_utils,
        "download_from_url",
        lambda **kwargs: downloaded.append(kwargs["filename"]),
    )

    download_from_zenodo_record(
        record_id="123", root=tmp_path, files_to_download=["darcy_16.tgz"]
    )
    assert downloaded == ["darcy_16.tgz"]


def test_zenodo_raises_on_bad_status(monkeypatch, tmp_path) -> None:
    """A non-200 response from the Zenodo API raises RuntimeError."""

    class _Resp:
        status_code = 404

        def json(self):  # pragma: no cover
            return {}

    monkeypatch.setattr(web_utils.requests, "get", lambda url, **kw: _Resp())
    with pytest.raises(RuntimeError):
        download_from_zenodo_record(
            record_id="123", root=tmp_path, files_to_download=["x"]
        )


# ---------------------------------------------------------------------------
# download_from_url - already-present file short-circuit
# ---------------------------------------------------------------------------


def test_download_from_url_uses_existing_verified_file(monkeypatch, tmp_path) -> None:
    """If the file already exists with a matching md5, no HTTP request is made."""
    content = b"already here"
    fname = "data.bin"
    (tmp_path / fname).write_bytes(content)
    md5 = hashlib.md5(content).hexdigest()

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("requests.get should not be called for cached file")

    monkeypatch.setattr(web_utils.requests, "get", _boom)
    download_from_url(
        url="http://example/data.bin", root=tmp_path, filename=fname, md5=md5
    )
    # File is left intact.
    assert (tmp_path / fname).read_bytes() == content


# ---------------------------------------------------------------------------
# Tar extraction safety (path traversal / link escape)
# ---------------------------------------------------------------------------


@pytest.fixture
def tar_impl() -> None:
    """Skip on interpreters without the stdlib ``data`` filter."""
    if not hasattr(tarfile, "data_filter"):
        pytest.skip("interpreter has no tarfile.data_filter")


def _make_tar(path, members):
    """Write a tar at ``path``; members are (TarInfo-kwargs, payload-or-None)."""
    with tarfile.open(path, "w") as tar:
        for kwargs, payload in members:
            info = tarfile.TarInfo(name=kwargs.pop("name"))
            for attr, value in kwargs.items():
                setattr(info, attr, value)
            if payload is not None:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                tar.addfile(info)


def _extract(tar_path, dest):
    dest.mkdir()
    with tarfile.open(tar_path) as archive:
        _extract_tar_safely(archive, dest)


def test_extract_benign_archive(tar_impl, tmp_path) -> None:
    """Regular nested files and in-tree symlinks still extract."""
    tar_path = tmp_path / "ok.tar"
    _make_tar(
        tar_path,
        [
            ({"name": "a/b.txt"}, b"data"),
            ({"name": "a/link", "type": tarfile.SYMTYPE, "linkname": "b.txt"}, None),
        ],
    )
    dest = tmp_path / "out"
    _extract(tar_path, dest)
    assert (dest / "a" / "b.txt").read_bytes() == b"data"
    assert (dest / "a" / "link").is_symlink()


def test_extract_rejects_dotdot_member(tar_impl, tmp_path) -> None:
    """A ``../`` member is refused and nothing is written outside the target."""
    tar_path = tmp_path / "evil.tar"
    _make_tar(tar_path, [({"name": "../evil.txt"}, b"pwned")])
    with pytest.raises(tarfile.TarError):
        _extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_sibling_prefix_escape(tar_impl, tmp_path) -> None:
    """``out_evil`` shares a string prefix with ``out`` but is outside it."""
    tar_path = tmp_path / "evil.tar"
    _make_tar(tar_path, [({"name": "../out_evil/x.txt"}, b"pwned")])
    with pytest.raises(tarfile.TarError):
        _extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "out_evil").exists()


def test_extract_absolute_member_stays_inside(tar_impl, tmp_path) -> None:
    """An absolute member name never lands at its absolute location."""
    outside = tmp_path / "abs_target.txt"
    tar_path = tmp_path / "evil.tar"
    _make_tar(tar_path, [({"name": str(outside)}, b"pwned")])
    try:
        _extract(tar_path, tmp_path / "out")
    except tarfile.TarError:
        pass  # refusing outright is also acceptable
    assert not outside.exists()


def test_extract_rejects_symlink_escape(tar_impl, tmp_path) -> None:
    """A symlink pointing outside the target directory is refused."""
    tar_path = tmp_path / "evil.tar"
    _make_tar(
        tar_path,
        [({"name": "link", "type": tarfile.SYMTYPE, "linkname": "../../etc"}, None)],
    )
    with pytest.raises(tarfile.TarError):
        _extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "out" / "link").exists()


def test_extract_rejects_device_member(tar_impl, tmp_path) -> None:
    """Special files (devices) are never extracted."""
    tar_path = tmp_path / "evil.tar"
    _make_tar(tar_path, [({"name": "dev", "type": tarfile.CHRTYPE}, None)])
    with pytest.raises(tarfile.TarError):
        _extract(tar_path, tmp_path / "out")


def test_extract_rejects_chained_symlink_escape(tar_impl: None, tmp_path: Path) -> None:
    """Links that only escape once earlier links exist (x -> ., y -> x/..)."""
    tar_path = tmp_path / "evil.tar"
    _make_tar(
        tar_path,
        [
            ({"name": "x", "type": tarfile.SYMTYPE, "linkname": "."}, None),
            ({"name": "y", "type": tarfile.SYMTYPE, "linkname": "x/.."}, None),
            ({"name": "y/escaped.txt"}, b"pwned"),
        ],
    )
    # Python < 3.12 raises; newer data filters keep the link inside the target.
    # Windows raises FileExistsError (an OSError) while creating the second link.
    try:
        _extract(tar_path, tmp_path / "out")
    except (tarfile.TarError, OSError):
        pass
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_refuses_without_data_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interpreters lacking the stdlib filter fail closed."""
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    tar_path = tmp_path / "ok.tar"
    _make_tar(tar_path, [({"name": "a.txt"}, b"data")])
    with pytest.raises(UnsafeTarMemberError):
        _extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "out" / "a.txt").exists()


# ---------------------------------------------------------------------------
# Real network download (gated)
# ---------------------------------------------------------------------------


def test_download_from_url_streams_and_extracts_tar(tmp_path, monkeypatch) -> None:
    """A mocked streaming download writes the file, checks size, and extracts a tar."""
    import io
    import tarfile

    # build a small in-memory tar archive containing one file
    member_bytes = b"hello mesh"
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        info = tarfile.TarInfo(name="inner.txt")
        info.size = len(member_bytes)
        tar.addfile(info, io.BytesIO(member_bytes))
    tar_bytes = tar_buf.getvalue()

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for i in range(0, len(tar_bytes), chunk_size):
                yield tar_bytes[i : i + chunk_size]

    monkeypatch.setattr(
        web_utils.requests, "get", lambda url, stream=False, **kw: _FakeResp()
    )

    # filename omitted -> derived from the url basename; root as str -> Path branch
    download_from_url(
        url="http://example/archive.tgz",
        root=str(tmp_path),
        size=len(tar_bytes),
        chunk_size=4,
        extract_tars=True,
    )
    # the archive was written and its contents extracted alongside it
    assert (tmp_path / "archive.tgz").exists()
    assert (tmp_path / "inner.txt").read_bytes() == member_bytes


def test_download_from_url_size_mismatch_raises(tmp_path, monkeypatch) -> None:
    """A downloaded file whose size disagrees with `size` raises RuntimeError."""

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"short"

    monkeypatch.setattr(
        web_utils.requests, "get", lambda url, stream=False, **kw: _FakeResp()
    )
    with pytest.raises(RuntimeError):
        download_from_url(
            url="http://example/data.bin",
            root=tmp_path,
            filename="data.bin",
            size=9999,  # deliberately wrong
            chunk_size=4,
            extract_tars=False,
        )


def test_download_from_url_corrupt_file_raises(tmp_path, monkeypatch) -> None:
    """A downloaded file failing its md5 check raises RuntimeError."""
    payload = b"abcd"  # 4 bytes, matches size; md5 will not match the wrong digest

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield payload

    monkeypatch.setattr(
        web_utils.requests, "get", lambda url, stream=False, **kw: _FakeResp()
    )
    with pytest.raises(RuntimeError):
        download_from_url(
            url="http://example/data.bin",
            root=tmp_path,
            filename="data.bin",
            md5="0" * 32,  # wrong md5 -> integrity check fails after download
            size=len(payload),
            chunk_size=4,
            extract_tars=False,
        )


@pytest.mark.skip(
    reason="Performs a real HTTP download; requires network access and is not run "
    "in CI. Offline branches are covered by the monkeypatched tests above."
)
def test_download_from_url_real(tmp_path) -> None:
    download_from_url(
        url="https://example.com/somefile.bin",
        root=tmp_path,
        filename="somefile.bin",
        size=None,
    )


def _failing_resp(payload: bytes):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield payload
            raise ConnectionError("dropped")

    return _Resp()


def test_interrupted_download_leaves_no_file(tmp_path, monkeypatch) -> None:
    """A connection that drops mid-stream leaves neither the file nor a .part."""
    monkeypatch.setattr(
        web_utils.requests, "get", lambda url, **kw: _failing_resp(b"abc")
    )
    with pytest.raises(ConnectionError):
        download_from_url(
            url="http://example/data.bin",
            root=tmp_path,
            filename="data.bin",
            extract_tars=False,
        )
    assert list(tmp_path.iterdir()) == []


def test_download_passes_a_timeout(tmp_path, monkeypatch) -> None:
    seen = {}

    def _get(url, **kw):
        seen.update(kw)
        return _failing_resp(b"abc")

    monkeypatch.setattr(web_utils.requests, "get", _get)
    with pytest.raises(ConnectionError):
        download_from_url("http://example/x.bin", root=tmp_path, extract_tars=False)
    assert seen.get("timeout")


@pytest.mark.parametrize("name", ["../x", "a/b", "..", "a\\b"])
def test_download_rejects_non_plain_file_names(tmp_path, name) -> None:
    with pytest.raises(ValueError):
        download_from_url(
            "http://example/x.bin", root=tmp_path, filename=name, extract_tars=False
        )
