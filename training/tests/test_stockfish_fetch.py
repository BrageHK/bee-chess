"""Unit tests for stockfish_fetch. Mocks all HTTP I/O -- no real network
access, matching CI's default fast/offline `uv run pytest`.
"""

from __future__ import annotations

from typing import Self

import httpx
import pytest

from bee_training.dataset import stockfish_fetch as sf


class _FakeGetResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            raise httpx.HTTPStatusError("error", request=request, response=httpx.Response(self.status_code, request=request))

    def json(self) -> dict:
        return self._payload


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self):
        yield from self._chunks


class _FakeClient:
    def __init__(self, get_response=None, stream_response=None):
        self._get_response = get_response
        self._stream_response = stream_response

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def get(self, url: str):
        return self._get_response

    def stream(self, method: str, url: str):
        return self._stream_response


def test_resolve_release_latest(monkeypatch) -> None:
    payload = {
        "tag_name": "sf_18",
        "assets": [
            {"name": "stockfish-ubuntu-x86-64-avx2.tar", "browser_download_url": "https://x/1"},
        ],
    }
    monkeypatch.setattr(sf, "_client", lambda token: _FakeClient(get_response=_FakeGetResponse(200, payload)))

    release = sf.resolve_release()
    assert release.tag == "sf_18"
    assert release.assets[0].name == "stockfish-ubuntu-x86-64-avx2.tar"


def test_resolve_release_missing_tag_raises(monkeypatch) -> None:
    monkeypatch.setattr(sf, "_client", lambda token: _FakeClient(get_response=_FakeGetResponse(404)))
    with pytest.raises(sf.StockfishFetchError):
        sf.resolve_release("does-not-exist")


def _release(*names: str) -> sf.Release:
    return sf.Release(tag="sf_18", assets=[sf.Asset(name=n, download_url=f"https://x/{n}") for n in names])


def test_select_asset_macos_arm(monkeypatch) -> None:
    monkeypatch.setattr(sf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sf.platform, "machine", lambda: "arm64")
    release = _release("stockfish-macos-m1-apple-silicon.tar", "stockfish-ubuntu-x86-64-avx2.tar")
    asset = sf.select_asset(release)
    assert asset.name == "stockfish-macos-m1-apple-silicon.tar"


def test_select_asset_linux_prefers_portable_isa(monkeypatch) -> None:
    monkeypatch.setattr(sf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sf.platform, "machine", lambda: "x86_64")
    release = _release(
        "stockfish-ubuntu-x86-64.tar",
        "stockfish-ubuntu-x86-64-avx512.tar",
        "stockfish-ubuntu-x86-64-avx2.tar",
    )
    asset = sf.select_asset(release)
    assert asset.name == "stockfish-ubuntu-x86-64.tar"


def test_select_asset_no_match_raises(monkeypatch) -> None:
    monkeypatch.setattr(sf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sf.platform, "machine", lambda: "x86_64")
    release = _release("stockfish-windows-x86-64.zip")
    with pytest.raises(sf.StockfishFetchError):
        sf.select_asset(release)


def test_select_asset_explicit_isa(monkeypatch) -> None:
    monkeypatch.setattr(sf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sf.platform, "machine", lambda: "x86_64")
    release = _release("stockfish-ubuntu-x86-64-avx2.tar", "stockfish-ubuntu-x86-64-bmi2.tar")
    asset = sf.select_asset(release, prefer_isa="bmi2")
    assert asset.name == "stockfish-ubuntu-x86-64-bmi2.tar"


def test_select_asset_explicit_isa_no_match_raises(monkeypatch) -> None:
    monkeypatch.setattr(sf.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sf.platform, "machine", lambda: "x86_64")
    release = _release("stockfish-ubuntu-x86-64-avx2.tar")
    with pytest.raises(sf.StockfishFetchError):
        sf.select_asset(release, prefer_isa="avx512")


def test_download_asset_streams_and_caches(monkeypatch, tmp_path) -> None:
    asset = sf.Asset(name="stockfish-ubuntu-x86-64.tar", download_url="https://x/1")
    monkeypatch.setattr(
        sf, "_client", lambda token: _FakeClient(stream_response=_FakeStreamResponse([b"abc", b"def"]))
    )

    path = sf.download_asset(asset, tmp_path)
    assert path.read_bytes() == b"abcdef"
    assert not (tmp_path / f"{asset.name}.part").exists()

    # Second call is a cache hit and must not touch the network at all.
    def _fail_if_called(token):
        raise AssertionError("should not fetch again on a cache hit")

    monkeypatch.setattr(sf, "_client", _fail_if_called)
    path_again = sf.download_asset(asset, tmp_path)
    assert path_again == path


def test_find_binary_prefers_extensionless_executable(tmp_path) -> None:
    (tmp_path / "stockfish-readme.md").write_text("docs", encoding="utf-8")
    (tmp_path / "stockfish-license.txt").write_text("license", encoding="utf-8")
    binary = tmp_path / "stockfish"
    binary.write_bytes(b"\x7fELF")

    found = sf._find_binary(tmp_path)
    assert found == binary


def test_find_binary_raises_when_none_found(tmp_path) -> None:
    (tmp_path / "readme.md").write_text("docs", encoding="utf-8")
    with pytest.raises(sf.StockfishFetchError):
        sf._find_binary(tmp_path)
