# Optional Syzygy endgame knowledge

Bee advertises `SyzygyPath` (empty by default) and `SyzygyProbeLimit`
(default 6, range 0–7). Set the limit to 7 when supplying seven-piece
tables; 0 disables probing. Piece counts include both kings.

```text
uci
setoption name SyzygyPath value /path/to/syzygy
setoption name SyzygyProbeLimit value 6
isready
position fen k7/P7/2K5/8/8/8/8/8 w - - 0 1
go depth 4
```

Use `.rtbw` files for WDL, plus optional `.rtbz` files for root DTZ and
move selection. Supply the smaller material tables needed after captures
and promotions, too. Paths may contain spaces; separate multiple directories
with `:` on Unix or `;` on Windows. No downloads happen inside Bee. The
MIT-licensed Fathom probe library is compiled into the binary; building it
requires a C compiler. Running the binary requires neither a separate
probe library nor tablebase files.

An empty path, empty directory, missing files, or unsuccessful probes use
normal search. `setoption name SyzygyPath value` clears the path;
`<empty>` and `""` are also accepted. Changing path or probe limit clears
the transposition table so cached scores cannot cross A/B configurations.
Re-setting a path rescans it. `ucinewgame` retains the configured path.
Use complete, verified Syzygy files; Fathom memory-maps files while probing.

Search uses exact WDL scores before static evaluation and quiescence stand-pat.
Draws, cursed wins, and blessed losses score zero under the fifty-move rule.
Decisive results score ±20000cp, below Bee's mate-score range: they do not
claim a mate distance. WDL tables assume the halfmove counter has just reset.
Bee only uses decisive WDL cutoffs at counter zero; with a nonzero counter,
it continues search unless a root DTZ probe supplies the clock-aware result.
The [Syzygy format documentation](https://github.com/syzygy1/tb#tablebase-files)
describes the WDL/DTZ distinction.

At the root, Bee checks a DTZ move against its own legal moves before using
it. Without DTZ, a proven draw can still return early after a child WDL probe
identifies a drawing move. A root WDL win/loss without a usable DTZ move
continues normal search; WDL alone does not provide a conversion strategy.
Repeated reversible history bypasses the history-free DTZ shortcut. Existing
terminal and claimable-draw handling takes precedence. Positions with castling
rights or invalid probe metadata are skipped.

## Telemetry and Lab A/B

Every search info line includes standard `tbhits`. With probes enabled,
completed iterations also emit diagnostics, even with `debug off`:

```text
info string tb hit pieces=3 wdl=draw exact=true root=true eval_cp=...
info string tb probes=... hits=... draw_hits=... root_resolved=true
```

Successful DTZ probes add `dtz=N` (absolute plies to zeroing, as reported by
Fathom). `wdl` is `win`, `draw`, `loss`, `cursed_win`, or `blessed_loss`, from
the probed position's side-to-move perspective. `exact=false` marks a decisive
WDL result whose nonzero halfmove clock prevents an exact cutoff.
`eval_cp` is Bee's static evaluator on that same position, before tablebase
correction. `root=true` means the sample describes the root; otherwise it is
the first successful interior probe. Only this bounded sample is emitted,
not one log line per node. `root_resolved` means probing supplied both the
score and the legal move, avoiding further iterative deepening.

Counters are cumulative within a search, through the last completed iteration,
and reset on the next search. `probes` counts backend WDL/DTZ calls, including
misses; `hits` counts successful calls; `draw_hits` counts draw-valued hits.
These are probe counts, not unique positions, games, or adjudicated results.
An aborted iteration's extra probes are not included in its preceding result.

In Lab's discovered engine options, compare A with an empty `SyzygyPath` to
B with a local path, keeping other options and time controls identical. The
path is local to the machine running the engine. Raw UCI logs retain the new
diagnostics. Game-level aggregates, moves saved, and evaluation-error buckets
are follow-up Lab analysis work. WDL provides categorical truth, not a target
centipawn value for averaging error in won/lost positions; drawn samples can
compare `abs(eval_cp)` against zero.

## Regressions and measurements

`cargo test -p bee-engine` runs offline against small three-piece WDL/DTZ
fixtures. Coverage includes actual hit/miss/fallback behavior, both score
perspectives, promotion decoding, invalid metadata, repeated history,
fifty-move handling, option changes, UCI telemetry, and search cancellation.
Full six- and seven-piece sets are not bundled or exercised in CI.

The reported [Lichess game VNfytGnY](https://lichess.org/VNfytGnY) never reached
tablebase range: after 156.Bxd2 it had 11 pieces, and no further captures
occurred. Its final position after 206.Be5 has halfmove clock 100. Tests keep
both positions as out-of-range/fifty-move regressions. Syzygy cannot certify
the earlier 11-piece position as a theoretical draw, and the exported PGN
does not contain Bee's original +1.77 search trace.

```text
After 156.Bxd2: 8/2k5/b3p1p1/5p1p/5P1P/4KP2/3B4/8 b - - 0 156
After 206.Be5:  8/8/2b1p1p1/3kBp1p/5P1P/4KP2/8/8 b - - 100 206
```

Local release measurements on 2026-09-12, 11 alternating runs per binary,
fresh processes, `TTReuse=PerSearch`, default evaluator/search options:

| Position | Depth | Main / disabled Syzygy median ms | Nodes (both) |
| --- | ---: | ---: | ---: |
| Start position | 6 | 89 / 86 | 35,094 |
| After `e4 e5 Nf3 Nc6 Bb5 a6` | 5 | 71 / 64 | 18,694 |
| VNfytGnY after 156.Bxd2 | 6 | 22 / 22 | 11,339 |

Scores and best moves also matched in every run. These short timings are a
disabled-path regression check, not evidence of a speed or playing-strength
improvement. The game sample scored +165cp for Black at depth 6 in both
builds, illustrating that this out-of-range evaluation remains unchanged.
