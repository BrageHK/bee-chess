"""
Real training loop for `ChessMamba` on `data/main-dawg/`, with crash-safe
checkpoint/resume: a periodic checkpoint (model + optimizer + step count)
written atomically, so a crash between writes never leaves a corrupt
checkpoint, and a mismatched-architecture resume is refused with a clear
error rather than silently loading garbage.

Run as:
  python -m bee_training.chess_mamba.train
  python -m bee_training.chess_mamba.train --checkpoint-dir checkpoints/run2 --total-steps 50000

Re-running the exact same command after a crash (or Ctrl+C) resumes from
`<checkpoint-dir>/latest.pt` automatically -- no separate --resume flag
needed, since "the checkpoint exists" already is the signal.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bee_training.chess_mamba.encode import IN_DIM, PositionDataset, load_all_records
from bee_training.chess_mamba.model import ChessMamba, hybrid_layer_types

# Config fields that determine parameter *shapes* -- if any of these
# differ from a checkpoint's saved config, loading its state_dict would
# fail (or silently do the wrong thing), so a resume must refuse instead.
_ARCHITECTURE_FIELDS = ("d_model", "n_layers", "n_ssm", "d_state", "expand", "n_value_bins")


@dataclass
class TrainConfig:
    data_glob: str = "data/main-dawg/shards/*.positions.jsonl"
    checkpoint_dir: str = "checkpoints/default"
    d_model: int = 192
    n_layers: int = 8
    n_ssm: int = 1
    d_state: int = 8
    expand: float = 1.0
    scan_backend: str = "pscan"  # not an architecture field -- see _ARCHITECTURE_FIELDS
    n_value_bins: int = 128
    batch_size: int = 64
    lr: float = 3e-4
    total_steps: int = 20_000
    val_fraction: float = 0.05
    max_records: int = 0  # 0 = no limit; caps in-memory positions loaded (see load_all_records)
    save_every: int = 200
    val_every: int = 500
    log_every: int = 20
    num_workers: int = 2
    compile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def architecture_key(self) -> tuple:
        return tuple(getattr(self, f) for f in _ARCHITECTURE_FIELDS)


def build_model(config: TrainConfig) -> ChessMamba:
    layer_types = hybrid_layer_types(config.n_layers, n_ssm=config.n_ssm)
    return ChessMamba(
        d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state,
        expand=config.expand, scan_backend=config.scan_backend, layer_types=layer_types,
        n_history=0, n_value_bins=config.n_value_bins,
    )


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     global_step: int, config: TrainConfig, best_val_loss: float) -> None:
    """Writes to a temp file then atomically renames into place (`os.replace`
    is atomic on the same filesystem), so a crash mid-write can never leave
    `latest.pt` itself truncated/corrupt -- worst case you lose the
    in-progress write and keep the previous good checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "config": config.to_dict(),
        "best_val_loss": best_val_loss,
    }, tmp_path)
    os.replace(tmp_path, path)


def append_history(path: Path, record: dict) -> None:
    """Appends one JSON line per call -- history.jsonl is never rewritten in
    place, so it survives resumes (unlike latest.pt) and can be tailed live
    while training runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_checkpoint_if_compatible(path: Path, config: TrainConfig) -> dict | None:
    """Returns the loaded checkpoint dict if `path` exists and its saved
    config's architecture-affecting fields match `config`'s; returns None
    if no checkpoint exists. Raises if a checkpoint exists but was trained
    with an incompatible architecture -- refuse rather than silently
    loading a state_dict that doesn't fit (or worse, one that happens to
    fit but means something different)."""
    if not path.exists():
        return None
    # weights_only=False: this checkpoint is always one this project's own
    # train.py wrote (a plain dict of tensors + str/int/float config), never
    # untrusted input, so torch's newer default pickle-safety restriction
    # (meant for loading other people's model files) doesn't apply here.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # merge onto *current* config (not TrainConfig()'s bare defaults) so a
    # TrainConfig field added after this checkpoint was written falls back
    # to what the current run asked for, instead of erroring on a missing key
    saved_config = TrainConfig(**{**config.to_dict(), **ckpt["config"]})
    if saved_config.architecture_key() != config.architecture_key():
        raise RuntimeError(
            f"checkpoint at {path} was trained with a different architecture "
            f"({dict(zip(_ARCHITECTURE_FIELDS, saved_config.architecture_key()))}) "
            f"than the current config ({dict(zip(_ARCHITECTURE_FIELDS, config.architecture_key()))}) "
            f"-- refusing to resume into it. Use a different --checkpoint-dir, or delete it "
            f"if you really mean to discard it."
        )
    return ckpt


def _collate(batch):
    planes, moves, bins = zip(*batch)
    x = torch.stack(planes)
    target_move = torch.tensor(moves, dtype=torch.long)
    target_bin = torch.tensor(bins, dtype=torch.long)
    return x, target_move, target_bin


def train_step(model, optimizer, x, target_move, target_bin) -> float:
    policy_logits, value_logits = model(x)
    loss = F.cross_entropy(policy_logits.reshape(x.shape[0], -1), target_move) \
        + F.cross_entropy(value_logits, target_bin)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for x, target_move, target_bin in loader:
        x, target_move, target_bin = x.to(device), target_move.to(device), target_bin.to(device)
        policy_logits, value_logits = model(x)
        loss = F.cross_entropy(policy_logits.reshape(x.shape[0], -1), target_move) \
            + F.cross_entropy(value_logits, target_bin)
        total_loss += loss.item() * x.shape[0]
        n += x.shape[0]
    model.train()
    return total_loss / max(1, n)


def split_records(records, val_fraction: float, seed: int):
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(records) * val_fraction))
    val_idx = set(indices[:n_val])
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]
    return train_records, val_records


def wait_for_data(config: TrainConfig, poll_s: float = 10.0) -> tuple[list, list]:
    """Polls `config.data_glob` until enough positions exist to train on,
    returning (shard_paths, records).

    Positions only appear on disk once a self-play *game* finishes (the
    generator writes a whole game's worth of positions at once, not one
    at a time -- see worker.py), which can take a while at a real node
    budget. Running the generator and trainer side by side (this
    project's train-mamba.sh) means the trainer would otherwise start and
    immediately crash before the generator has produced its first game.

    `config.max_records`, if set, caps how many positions get loaded into
    memory -- see `load_all_records`'s docstring for why an unbounded load
    against this project's current data scale can exhaust host memory.
    """
    min_records = config.batch_size * 2
    logged_waiting = False
    while True:
        shard_paths = sorted(glob.glob(config.data_glob))
        max_records = config.max_records or None
        records = load_all_records(shard_paths, max_records=max_records) if shard_paths else []
        if len(records) >= min_records:
            if logged_waiting:
                print(f"found {len(records)} positions, proceeding")
            return shard_paths, records
        if not logged_waiting:
            print(f"waiting for at least {min_records} positions at {config.data_glob} "
                  f"(found {len(records)}) -- is the dataset generator running?")
            logged_waiting = True
        time.sleep(poll_s)


def run(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)

    shard_paths, records = wait_for_data(config)
    print(f"loaded {len(records)} positions from {len(shard_paths)} shard(s)")

    train_records, val_records = split_records(records, config.val_fraction, config.seed)
    train_loader = DataLoader(
        PositionDataset(train_records, n_value_bins=config.n_value_bins),
        batch_size=config.batch_size, shuffle=True, drop_last=True,
        collate_fn=_collate, num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        PositionDataset(val_records, n_value_bins=config.n_value_bins),
        batch_size=config.batch_size, shuffle=False, drop_last=False,
        collate_fn=_collate, num_workers=config.num_workers,
    )

    model = build_model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    checkpoint_path = Path(config.checkpoint_dir) / "latest.pt"
    best_path = Path(config.checkpoint_dir) / "best.pt"
    history_path = Path(config.checkpoint_dir) / "history.jsonl"
    ckpt = load_checkpoint_if_compatible(checkpoint_path, config)
    global_step = 0
    best_val_loss = float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"resumed from {checkpoint_path} at step {global_step} (best_val_loss={best_val_loss:.4f})")
    else:
        print(f"no checkpoint at {checkpoint_path}, starting fresh")

    # Compile the *forward path* only -- always save/load `model`'s own
    # (uncompiled) state_dict, never the compiled wrapper's, since
    # torch.compile's wrapper isn't guaranteed to use the same state_dict
    # key names.
    compute_model = model
    if config.compile:
        try:
            compute_model = torch.compile(model)
        except Exception as e:  # noqa: BLE001 - best-effort, fall back to eager
            print(f"torch.compile failed ({type(e).__name__}: {e}); continuing uncompiled")

    model.train()
    t_start = time.time()
    step_at_start = global_step
    data_iter = iter(train_loader)

    try:
        while global_step < config.total_steps:
            try:
                x, target_move, target_bin = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)  # new shuffled epoch
                x, target_move, target_bin = next(data_iter)

            x = x.to(config.device)
            target_move = target_move.to(config.device)
            target_bin = target_bin.to(config.device)

            loss = train_step(compute_model, optimizer, x, target_move, target_bin)
            global_step += 1

            if global_step % config.log_every == 0:
                steps_per_sec = (global_step - step_at_start) / (time.time() - t_start)
                print(f"step {global_step:7d}  loss {loss:.4f}  ({steps_per_sec:.2f} steps/sec)")
                append_history(history_path, {"step": global_step, "train_loss": loss})

            if global_step % config.val_every == 0:
                val_loss = evaluate(compute_model, val_loader, config.device)
                print(f"step {global_step:7d}  val_loss {val_loss:.4f}")
                append_history(history_path, {"step": global_step, "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(best_path, model, optimizer, global_step, config, best_val_loss)
                    print(f"step {global_step:7d}  new best val_loss {best_val_loss:.4f}, saved -> {best_path}")

            if global_step % config.save_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer, global_step, config, best_val_loss)
                print(f"step {global_step:7d}  checkpoint saved -> {checkpoint_path}")
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {global_step}, saving checkpoint before exit...")
        save_checkpoint(checkpoint_path, model, optimizer, global_step, config, best_val_loss)
        print(f"checkpoint saved -> {checkpoint_path}")
        raise

    save_checkpoint(checkpoint_path, model, optimizer, global_step, config, best_val_loss)
    print(f"training complete at step {global_step}, final checkpoint -> {checkpoint_path}")


def _parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    for field_name, default in asdict(defaults).items():
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(arg_name, dest=field_name, action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(arg_name, dest=field_name, type=type(default), default=default)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = _parse_args()
    assert IN_DIM == 20  # sanity: encode.py and ChessMamba's fixed n_history=0 in_dim must agree
    run(cfg)
