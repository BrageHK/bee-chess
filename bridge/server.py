#!/usr/bin/env python3
"""WebSocket <-> UCI bridges: stockfish on :8765, bee on :8766, bee-mamba
(the trained ChessMamba checkpoint, played move-by-move with no search --
see training/src/bee_training/chess_mamba/play.py) on :8767.

The browser cannot spawn a process, so this exposes one WebSocket per
engine and pipes lines straight to the engine's stdin/stdout. Each browser
connection gets its own engine process.
"""
import asyncio
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


def check(path, build_cmd):
    if not path.exists():
        sys.exit(f"missing engine binary: {path}\nbuild it with: {build_cmd}")


async def main():
    check(STOCKFISH, "./scripts/build-stockfish.sh")
    check(BEE, "./scripts/build-bee.sh")
    check(MAMBA_PYTHON, "cd training && uv sync")
    check(MAMBA_CHECKPOINT, "./scripts/train-mamba.sh (or point MAMBA_CHECKPOINT elsewhere)")
    mamba_argv = [
        str(MAMBA_PYTHON), "-m", "bee_training.chess_mamba.play",
        "--checkpoint", str(MAMBA_CHECKPOINT), "--device", "cpu",
    ]
    async with (
        websockets.serve(make_handler([str(STOCKFISH)], STOCKFISH.parent), "localhost", 8765),
        websockets.serve(make_handler([str(BEE)], BEE.parent), "localhost", 8766),
        websockets.serve(make_handler(mamba_argv, TRAINING), "localhost", 8767),
    ):
        print("stockfish :8765  |  bee :8766  |  bee-mamba :8767")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
