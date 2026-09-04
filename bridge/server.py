#!/usr/bin/env python3
"""WebSocket <-> UCI bridges: stockfish on :8765, bee on :8766.

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


def make_handler(engine_path):
    async def handle(ws):
        proc = await asyncio.create_subprocess_exec(
            str(engine_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            cwd=str(engine_path.parent),
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
    async with (
        websockets.serve(make_handler(STOCKFISH), "localhost", 8765),
        websockets.serve(make_handler(BEE), "localhost", 8766),
    ):
        print("stockfish :8765  |  bee :8766")
        await asyncio.Future()


asyncio.run(main())
