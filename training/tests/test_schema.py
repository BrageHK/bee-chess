import json

from bee_training.dataset.schema import (
    SCHEMA_VERSION,
    GameRecord,
    PositionRecord,
    append_jsonl,
    read_jsonl,
)


def _sample_position() -> PositionRecord:
    return PositionRecord(
        schema_version=SCHEMA_VERSION,
        game_id="run-0",
        ply=3,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        side_to_move="b",
        eval_cp=25,
        eval_mate=None,
        depth=12,
        best_move="e7e5",
        pv=["e7e5", "g1f3"],
        game_result="1-0",
        stockfish_version="sf_18",
    )


def test_position_record_round_trips_through_json() -> None:
    record = _sample_position()
    restored = PositionRecord.from_dict(json.loads(record.to_json()))
    assert restored == record


def test_position_record_carries_schema_version() -> None:
    record = _sample_position()
    assert record.schema_version == SCHEMA_VERSION


def test_game_record_round_trips_through_json() -> None:
    record = GameRecord(
        schema_version=SCHEMA_VERSION,
        game_id="run-0",
        result="1-0",
        termination="checkmate",
        ply_count=42,
        opening_source="book",
        stockfish_version="sf_18",
        node_limit=25_000,
        time_limit_s=None,
        depth_limit=None,
        seed=0,
    )
    restored = GameRecord.from_dict(json.loads(record.to_json()))
    assert restored == record


def test_append_and_read_jsonl_round_trips(tmp_path) -> None:
    path = tmp_path / "shard.jsonl"
    records = [_sample_position(), _sample_position()]
    append_jsonl(path, [r.to_json() for r in records])
    append_jsonl(path, [records[0].to_json()])

    loaded = read_jsonl(path)
    assert len(loaded) == 3
    assert PositionRecord.from_dict(loaded[0]) == records[0]


def test_append_jsonl_no_op_on_empty_list(tmp_path) -> None:
    path = tmp_path / "shard.jsonl"
    append_jsonl(path, [])
    assert not path.exists()


def test_read_jsonl_missing_file_returns_empty(tmp_path) -> None:
    assert read_jsonl(tmp_path / "missing.jsonl") == []
