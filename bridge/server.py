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
# Not engine/target/ -- engine/ is a member of the root Cargo workspace
# (see /Cargo.toml, introduced alongside lab/ by #68), so Cargo shares
# one target/ directory across all workspace members by default.
BEE = ROOT / "target" / "release" / "bee"
TRAINING = ROOT / "training"
MAMBA_PYTHON = TRAINING / ".venv" / "bin" / "python3"
MAMBA_CHECKPOINT = TRAINING / "checkpoints" / "main-dawg" / "latest.pt"


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
