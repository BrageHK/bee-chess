from dataclasses import replace
from itertools import pairwise

import pytest

from bee_training.dataset import manifest as manifest_mod
from bee_training.dataset.config import SelfPlayConfig


def _config(tmp_path, **overrides) -> SelfPlayConfig:
    defaults = {
        "games": 10,
        "workers": 3,
        "limit_kind": "nodes",
        "limit_value": 5000,
        "opening_book": None,
        "opening_plies": 8,
        "resign_cp": 1000,
        "resign_plies": 8,
        "draw_cp": 10,
        "draw_plies": 8,
        "draw_min_ply": 40,
        "max_plies": 300,
        "stockfish_version": "sf_18",
        "output_dir": str(tmp_path),
        "run_id": "test-run",
        "seed": 0,
    }
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def test_build_fresh_manifest_partitions_all_games_disjointly(tmp_path) -> None:
    config = _config(tmp_path, games=10, workers=3)
    manifest = manifest_mod.build_fresh_manifest(config)
    ranges = [set(range(w.start, w.end)) for w in manifest.workers.values()]
    union = set().union(*ranges)
    assert union == set(range(10))
    for a, b in pairwise(ranges):
        assert not (a & b)


def test_load_or_create_creates_and_persists_manifest(tmp_path) -> None:
    config = _config(tmp_path)
    manifest = manifest_mod.load_or_create(config)
    assert manifest_mod.manifest_path(config).exists()
    reloaded = manifest_mod.load(config)
    assert reloaded == manifest


def test_load_or_create_resumes_matching_config(tmp_path) -> None:
    config = _config(tmp_path)
    manifest_mod.load_or_create(config)
    manifest_mod.mark_completed(config, "0", 0)

    resumed = manifest_mod.load_or_create(config)
    assert resumed.workers["0"].last_completed_index == 0


def test_load_or_create_rejects_mismatched_config(tmp_path) -> None:
    config = _config(tmp_path)
    manifest_mod.load_or_create(config)

    changed = replace(config, limit_value=99_999)
    with pytest.raises(manifest_mod.ManifestMismatchError):
        manifest_mod.load_or_create(changed)


def test_load_or_create_rejects_mismatched_shape(tmp_path) -> None:
    config = _config(tmp_path)
    manifest_mod.load_or_create(config)

    changed = replace(config, workers=5)
    with pytest.raises(manifest_mod.ManifestMismatchError):
        manifest_mod.load_or_create(changed)


def test_fresh_discards_existing_progress(tmp_path) -> None:
    config = _config(tmp_path)
    manifest_mod.load_or_create(config)
    manifest_mod.mark_completed(config, "0", 0)

    fresh = manifest_mod.load_or_create(config, force_fresh=True)
    assert fresh.workers["0"].last_completed_index == -1


def test_worker_state_remaining_excludes_completed() -> None:
    state = manifest_mod.WorkerState(start=0, end=5, last_completed_index=1)
    assert list(state.remaining()) == [2, 3, 4]


def test_completed_count_sums_across_workers(tmp_path) -> None:
    config = _config(tmp_path, games=9, workers=3)
    manifest_mod.load_or_create(config)
    manifest_mod.mark_completed(config, "0", 2)
    manifest_mod.mark_completed(config, "1", 4)

    manifest = manifest_mod.load(config)
    assert manifest.completed_count() == 3 + 2  # worker 0: indices 0-2 (3 games); worker 1: 3-4 (2 games)
