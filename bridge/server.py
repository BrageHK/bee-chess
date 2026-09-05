#!/usr/bin/env python3
"""WebSocket <-> UCI bridges: stockfish on :8765, bee on :8766, bee-mamba
(the trained ChessMamba checkpoint, played move-by-move with no search --
see training/src/bee_training/chess_mamba/play.py) on :8767.

The browser cannot spawn a process, so this exposes one WebSocket per
engine and pipes lines straight to the engine's stdin/stdout. Each browser
connection gets its own engine process.

Stockfish and Bee are required -- the bridge refuses to start without
them. Bee-Mamba is optional: it needs a trained checkpoint that most
clones won't have (it's produced by ./scripts/train-mamba.sh, not
checked in), so its absence only disables :8767 rather than the whole
bridge. Without it, the frontend's "Play vs Bee-Mamba" mode will fail
to connect, but "Spectate: Stockfish vs Bee" (the other mode, and the
only one that existed before Bee-Mamba was added) keeps working.
"""
import asyncio
import contextlib
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
STOCKFISH = ROOT / "external" / "stockfish" / "src" / "stockfish"
BEE = ROOT / "engine" / "target" / "release" / "bee"
TRAINING = ROOT / "training"
MAMBA_PYTHON = TRAINING / ".venv" / "bin" / "python3"
MAMBA_CHECKPOINT = TRAINING / "checkpoints" / "main-dawg" / "latest.pt"


def make_handler(argv, cwd):
    async def handle(ws):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )

        async def pump():
            while line := await proc.stdout.readline():
                await ws.send(line.decode().rstrip())

        task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                proc.stdin.write((msg + "\n").encode())
                await proc.stdin.drain()
        finally:
            task.cancel()
            proc.kill()

    return handle


def require(path, build_cmd):
    """Stockfish/Bee are mandatory: without them there's nothing for
    the bridge to serve at all, so refuse to start."""
    if not path.exists():
        sys.exit(f"missing engine binary: {path}\nbuild it with: {build_cmd}")


def mamba_unavailable_reason():
    """None if Bee-Mamba can be started; otherwise why not, so main()
    can print a clear warning instead of silently not listening on
    :8767. Optional, unlike Stockfish/Bee -- see the module docstring."""
    if not MAMBA_PYTHON.exists():
        return f"missing {MAMBA_PYTHON} (run: cd training && uv sync)"
    if not MAMBA_CHECKPOINT.exists():
        return (
            f"missing checkpoint {MAMBA_CHECKPOINT} "
            "(run ./scripts/train-mamba.sh, or point MAMBA_CHECKPOINT elsewhere)"
        )
    return None


async def main():
    require(STOCKFISH, "./scripts/build-stockfish.sh")
    require(BEE, "./scripts/build-bee.sh")

    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(
            websockets.serve(make_handler([str(STOCKFISH)], STOCKFISH.parent), "localhost", 8765)
        )
        await stack.enter_async_context(
            websockets.serve(make_handler([str(BEE)], BEE.parent), "localhost", 8766)
        )

        status = "stockfish :8765  |  bee :8766"
        unavailable = mamba_unavailable_reason()
        if unavailable is None:
            mamba_argv = [
                str(MAMBA_PYTHON), "-m", "bee_training.chess_mamba.play",
                "--checkpoint", str(MAMBA_CHECKPOINT), "--device", "cpu",
            ]
            await stack.enter_async_context(
                websockets.serve(make_handler(mamba_argv, TRAINING), "localhost", 8767)
            )
            status += "  |  bee-mamba :8767"
        else:
            print(f"bee-mamba unavailable, skipping :8767 ({unavailable})", file=sys.stderr)
            print("'Play vs Bee-Mamba' will fail to connect; 'Spectate' is unaffected.", file=sys.stderr)

        print(status)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
