#!/usr/bin/env python3
"""WebSocket <-> UCI bridge for bee-mamba (the trained ChessMamba
checkpoint, played move-by-move with no search -- see
training/src/bee_training/chess_mamba/play.py) on :8767.

This is what's left of the original bridge (see #89): Stockfish and
Bee both moved to Bee Lab (`lab/`, see #67/#69), which is now the only
way the frontend reaches either of them -- authoritative game state,
not just a raw relay. Bee-Mamba stays here because it has no Lab-side
equivalent yet (#66/#70); once it does, this file goes away entirely.

The browser cannot spawn a process, so this exposes one WebSocket and
pipes lines straight to the engine's stdin/stdout. Each browser
connection gets its own engine process. Bee-Mamba is optional here
too: it needs a trained checkpoint that most clones won't have (it's
produced by ./scripts/train-mamba.sh, not checked in), so this process
exits with a clear message rather than silently listening on nothing
if it's missing.
"""
import asyncio
import contextlib
import os
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
MAMBA_PYTHON = TRAINING / ".venv" / "bin" / "python3"
MAMBA_CHECKPOINT = Path(os.environ["MAMBA_CHECKPOINT"]) if "MAMBA_CHECKPOINT" in os.environ \
    else TRAINING / "checkpoints" / "main-dawg" / "latest.pt"


def make_handler(argv, cwd):
    async def handle(ws):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )

        async def pump():
            while line := await proc.stdout.readline():
                await ws.send(line.decode().rstrip())

        async def watch_for_exit():
            """If the process dies (e.g. a missing Python module, a bad
            checkpoint path) -- especially right away, before ever
            printing a UCI reply -- neither `pump()` (its readline()
            just returns empty) nor the `async for msg in ws` loop
            below notices on their own: the browser is left waiting
            forever for a reply that will never come, and the socket
            never closes to tell it otherwise. This actively closes
            the socket once the process exits, after relaying whatever
            it printed to stderr as one diagnostic line so the reason
            is visible instead of a silent disconnect."""
            returncode = await proc.wait()
            stderr = (await proc.stderr.read()).decode(errors="replace").strip()
            reason = stderr.splitlines()[-1] if stderr else f"exited with code {returncode}"
            with contextlib.suppress(Exception):
                await ws.send(f"info string engine process exited: {reason}")
            with contextlib.suppress(Exception):
                await ws.close()

        pump_task = asyncio.create_task(pump())
        watch_task = asyncio.create_task(watch_for_exit())
        try:
            async for msg in ws:
                proc.stdin.write((msg + "\n").encode())
                await proc.stdin.drain()
        finally:
            pump_task.cancel()
            watch_task.cancel()
            proc.kill()

    return handle


def mamba_unavailable_reason():
    """None if Bee-Mamba can be started; otherwise why not."""
    if not MAMBA_PYTHON.exists():
        return f"missing {MAMBA_PYTHON} (run: cd training && uv sync)"
    if not MAMBA_CHECKPOINT.exists():
        return (
            f"missing checkpoint {MAMBA_CHECKPOINT} "
            "(run ./scripts/train-mamba.sh, or point MAMBA_CHECKPOINT elsewhere)"
        )
    return None


async def main():
    unavailable = mamba_unavailable_reason()
    if unavailable is not None:
        sys.exit(f"bee-mamba unavailable, nothing for this bridge to serve ({unavailable})")

    mamba_argv = [
        str(MAMBA_PYTHON), "-m", "bee_training.chess_mamba.play",
        "--checkpoint", str(MAMBA_CHECKPOINT), "--device", "cpu",
    ]
    async with websockets.serve(make_handler(mamba_argv, TRAINING), "localhost", 8767):
        print("bee-mamba :8767")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
