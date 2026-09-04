"""Download, verify, and cache the official Stockfish release binary.

No cryptographic checksums are published alongside Stockfish GitHub release
assets, so "verify" here means: (a) the asset came from the official
`official-stockfish/Stockfish` repo's release API over HTTPS, and (b) a
functional smoke test after extraction (spawn the binary, send `uci`, expect
`uciok`).
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

GITHUB_API = "https://api.github.com/repos/official-stockfish/Stockfish"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "stockfish"


class StockfishFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    download_url: str


@dataclass(frozen=True)
class Release:
    tag: str
    assets: list[Asset]


def _client(token: str | None) -> httpx.Client:
    headers = {"User-Agent": "bee-training-stockfish-fetch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, timeout=30.0, follow_redirects=True)


def resolve_release(version: str | None = None, token: str | None = None) -> Release:
    """Resolve "latest" or a specific tag (e.g. "sf_18") to a `Release`."""
    url = f"{GITHUB_API}/releases/latest" if version in (None, "latest") else f"{GITHUB_API}/releases/tags/{version}"
    with _client(token) as client:
        resp = client.get(url)
    if resp.status_code == 404:
        raise StockfishFetchError(
            f"No Stockfish release found for tag {version!r}. "
            "Check https://github.com/official-stockfish/Stockfish/releases for valid tags."
        )
    resp.raise_for_status()
    data = resp.json()
    assets = [Asset(name=a["name"], download_url=a["browser_download_url"]) for a in data["assets"]]
    return Release(tag=data["tag_name"], assets=assets)


def _platform_key() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        os_key = "macos"
    elif system == "linux":
        os_key = "ubuntu"
    elif system == "windows":
        os_key = "windows"
    else:
        raise StockfishFetchError(f"Unsupported OS for Stockfish release assets: {system}")
    return os_key, machine


# Ordered from most-portable to most-specific; select_asset prefers the first match
# unless `prefer_isa` narrows the search.
_ISA_PRIORITY = ["x86-64", "x86-64-sse41-popcnt", "x86-64-avx2", "x86-64-bmi2", "x86-64-avx512", "x86-64-vnni256"]


def select_asset(release: Release, prefer_isa: str | None = None) -> Asset:
    os_key, machine = _platform_key()

    if os_key == "macos" and machine in ("arm64", "aarch64"):
        candidates = [a for a in release.assets if "macos-m1" in a.name or "macos-arm" in a.name]
    else:
        arch = "x86-64" if machine in ("x86_64", "amd64") else machine
        candidates = [a for a in release.assets if a.name.startswith(f"stockfish-{os_key}-{arch}")]

    if not candidates:
        available = ", ".join(a.name for a in release.assets)
        raise StockfishFetchError(
            f"No Stockfish asset matched platform ({os_key}, {machine}) in release {release.tag}. "
            f"Available assets: {available}"
        )

    if prefer_isa:
        matches = [a for a in candidates if prefer_isa in a.name]
        if matches:
            return matches[0]
        raise StockfishFetchError(
            f"Requested ISA variant {prefer_isa!r} not found for {os_key}/{machine} in release {release.tag}."
        )

    if len(candidates) == 1:
        return candidates[0]

    for isa in _ISA_PRIORITY:
        for a in candidates:
            if a.name.endswith(f"{isa}.tar") or a.name.endswith(f"{isa}.zip") or a.name == f"stockfish-{os_key}-{isa}":
                return a
    # Fall back to the shortest name, which is typically the least-specialized variant.
    return min(candidates, key=lambda a: len(a.name))


def download_asset(asset: Asset, dest_dir: Path, token: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / asset.name
    if final_path.exists():
        return final_path
    part_path = dest_dir / f"{asset.name}.part"
    with _client(token) as client, client.stream("GET", asset.download_url) as resp:
        resp.raise_for_status()
        with part_path.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    os.replace(part_path, final_path)
    return final_path


def _extract(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)


def _find_binary(root: Path) -> Path:
    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and "stockfish" in p.name.lower()
        and p.suffix.lower() not in (".md", ".txt", ".nnue")
    ]
    if not candidates:
        raise StockfishFetchError(f"Could not locate a Stockfish executable under {root}")
    # Prefer the one with no extension (or .exe) over any stray non-binary matches.
    candidates.sort(key=lambda p: (p.suffix.lower() not in ("", ".exe"), len(str(p))))
    return candidates[0]


def _smoke_test(binary_path: Path) -> None:
    try:
        proc = subprocess.run(
            [str(binary_path)],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StockfishFetchError(f"Stockfish binary at {binary_path} failed to run: {exc}") from exc
    if "uciok" not in proc.stdout:
        raise StockfishFetchError(
            f"Stockfish binary at {binary_path} did not respond with 'uciok' to the UCI handshake. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def verify_and_extract(archive_path: Path, extract_dir: Path) -> Path:
    _extract(archive_path, extract_dir)
    binary_path = _find_binary(extract_dir)
    mode = binary_path.stat().st_mode
    binary_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _smoke_test(binary_path)
    return binary_path


def ensure_stockfish(
    version: str | None = None,
    cache_dir: Path | None = None,
    prefer_isa: str | None = None,
    token: str | None = None,
) -> tuple[Path, str]:
    """Return (binary_path, resolved_tag), fetching only on a cache miss."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    release = resolve_release(version, token=token)
    asset = select_asset(release, prefer_isa=prefer_isa)

    run_dir = cache_dir / release.tag
    extract_dir = run_dir / "extracted"
    if extract_dir.exists():
        try:
            binary_path = _find_binary(extract_dir)
            _smoke_test(binary_path)
            return binary_path, release.tag
        except StockfishFetchError:
            pass  # Cached extraction is broken/incomplete; re-fetch below.

    archive_path = download_asset(asset, run_dir, token=token)
    binary_path = verify_and_extract(archive_path, extract_dir)
    return binary_path, release.tag
