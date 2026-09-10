# Testing transposition-table reuse

Build and start Bee Lab with `./scripts/dev.sh`. In the experiment setup,
set variant A to `TTReuse = PerSearch` and variant B to `TTReuse = PerGame`.
Keep `UseTT = true` and every other option, opening, and time control identical.
The controls are discovered from Bee's UCI handshake; reopen the setup after
rebuilding if an already-open form does not show `TTReuse`.

`PerGame` is the default. It keeps the table across searches in the same
engine process, including completed subtrees from a search interrupted by
`stop`. Select `PerSearch` to start every search with an empty table and
keep the previous depth-based replacement/flush behavior. Killers, history heuristics,
root move ordering state, and search statistics still reset for each search.
`UseTT = false` disables TT probing and storage with either policy.

For a direct UCI check, build and run:

```sh
cargo build --release -p bee-engine
./target/release/bee
```

Enter these commands, waiting for each `bestmove` before sending the next
position/search:

```text
uci
setoption name TTReuse value PerGame
ucinewgame
position startpos
go depth 5
position startpos moves b1c3 b8c6
go depth 3
quit
```

Repeat with `setoption name TTReuse value PerSearch` for the baseline.
Both modes also work with `go movetime ...` and clock-based `go wtime ...
btime ...`. No pondering commands were added.

The persistent table clears on `ucinewgame`, a change to `TTReuse`, any
search/evaluator option change, or an opening-book change. Resending the
same option value, changing debug output, or changing `MoveOverhead` keeps
the cache. A new engine process starts empty. A rewind or a position whose
history does not extend the previous searched game's history clears it at
the next search. Send the full move history (`position startpos moves ...`
or `position fen <base> moves ...`), as Bee Lab already does. Sending a new
standalone FEN each turn conservatively starts a fresh session because the
engine cannot establish continuity. Send `ucinewgame` between games even
when they share an opening.

The existing limit of 1,048,576 entries remains; Bee has no configurable
`Hash` size. Persistent replacement samples eight entries at capacity and
prefers recent/deep work instead of flushing the table. Entries retain the
Zobrist key, depth, mate-normalized score, bound, move, and generation.
Halfmove clock, repetition count, and a fingerprint of reversible history
guard score reuse. This extra history check can make even a cold `PerGame`
search visit more nodes than `PerSearch`. A history mismatch still permits
using a cached move for ordering, but cannot supply a score cutoff.

Local release-build measurements (five runs, default evaluator/options,
2026-09-10) for the second search in each sequence:

| Sequence | PerSearch nodes | PerGame nodes | Median time, PerSearch / PerGame |
| --- | ---: | ---: | ---: |
| Start position, depth 5 twice | 11,265 | 21 | 29 / <1 ms |
| Start position depth 5, then `b1c3 b8c6` at depth 3 | 2,278 | 1,208 | 5 / 2 ms |
| After `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6`, depth 4 twice | 3,475 | 33 | 11 / <1 ms |

The compared second searches returned the same score and best move. These
measure cache reuse, not playing strength; use paired games in Lab to judge
the time/memory tradeoff. TT cutoffs may produce shorter reported PVs.
