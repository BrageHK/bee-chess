import threading
import time

import chess
import pytest
import torch

from bee_training.chess_mamba.train import (
    TrainConfig,
    build_model,
    load_checkpoint_if_compatible,
    save_checkpoint,
    wait_for_data,
)
from bee_training.dataset.schema import PositionRecord, append_jsonl

N_RECORDS = 20


def _write_records(shard_path, n, start=0):
    lines = []
    for i in range(start, start + n):
        record = PositionRecord(
            schema_version=1, game_id="test-game", ply=i, fen=chess.STARTING_FEN,
            side_to_move="w", eval_cp=(i * 10) - 100, eval_mate=None, depth=1,
            best_move="e2e4", pv=["e2e4"], game_result="1-0", stockfish_version="test",
        )
        lines.append(record.to_json())
    append_jsonl(shard_path, lines)


def _make_shard(tmp_path):
    _write_records(tmp_path / "worker-0.positions.jsonl", N_RECORDS)
    return tmp_path / "*.positions.jsonl"


def _tiny_config(tmp_path, **overrides):
    defaults = {
        "data_glob": str(_make_shard(tmp_path)),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "d_model": 16, "n_layers": 2, "n_ssm": 1, "d_state": 4, "expand": 1.0,
        "batch_size": 4, "val_fraction": 0.2, "total_steps": 4,
        "save_every": 2, "val_every": 2, "log_every": 1, "num_workers": 0,
        "compile": False, "device": "cpu", "seed": 0,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_checkpoint_round_trip(tmp_path):
    config = _tiny_config(tmp_path)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(ckpt_path, model, optimizer, global_step=7, config=config)

    loaded = load_checkpoint_if_compatible(ckpt_path, config)
    assert loaded is not None
    assert loaded["global_step"] == 7

    model2 = build_model(config)
    model2.load_state_dict(loaded["model_state"])
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_no_checkpoint_returns_none(tmp_path):
    config = _tiny_config(tmp_path)
    result = load_checkpoint_if_compatible(tmp_path / "ckpt" / "latest.pt", config)
    assert result is None


def test_mismatched_architecture_refuses_resume(tmp_path):
    config = _tiny_config(tmp_path)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(ckpt_path, model, optimizer, global_step=1, config=config)

    different_config = _tiny_config(tmp_path, d_model=32)  # different architecture
    with pytest.raises(RuntimeError, match="different architecture"):
        load_checkpoint_if_compatible(ckpt_path, different_config)


def test_non_architecture_field_change_does_not_block_resume(tmp_path):
    config = _tiny_config(tmp_path)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(ckpt_path, model, optimizer, global_step=1, config=config)

    # lr, total_steps, scan_backend etc. aren't architecture fields -- changing them
    # (e.g. to extend a run, or switch scan backend) must not block resuming.
    different_config = _tiny_config(tmp_path, lr=1e-5, total_steps=100, scan_backend="pscan")
    loaded = load_checkpoint_if_compatible(ckpt_path, different_config)
    assert loaded is not None


def test_wait_for_data_blocks_until_enough_records_then_returns(tmp_path):
    """The generator writes a shard file only once its first game finishes
    (see worker.py), which can happen well after training starts when the
    two run side by side -- wait_for_data must poll, not crash, and must
    actually pick up data that arrives late."""
    shard_path = tmp_path / "worker-0.positions.jsonl"
    config = TrainConfig(
        data_glob=str(tmp_path / "*.positions.jsonl"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        batch_size=4,  # min_records = batch_size * 2 = 8
    )

    delay_s = 0.3

    def write_late():
        time.sleep(delay_s)
        _write_records(shard_path, 20)

    thread = threading.Thread(target=write_late)
    thread.start()
    t0 = time.perf_counter()
    shard_paths, records = wait_for_data(config, poll_s=0.05)
    elapsed = time.perf_counter() - t0
    thread.join()

    assert elapsed >= delay_s, "returned before the data actually existed"
    assert len(records) == 20
    assert shard_paths == [str(shard_path)]


def test_train_runs_and_checkpoints_then_resumes(tmp_path):
    from bee_training.chess_mamba.train import run

    config = _tiny_config(tmp_path, total_steps=2)
    run(config)
    ckpt_path = tmp_path / "ckpt" / "latest.pt"
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["global_step"] == 2

    # resume and train further -- should pick up from step 2, not restart from 0
    config2 = _tiny_config(tmp_path, total_steps=4)
    run(config2)
    loaded2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded2["global_step"] == 4
