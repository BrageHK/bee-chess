import { useEffect, useState } from "react";
import {
  getAnalysisMove, getAnalysisReport, listAnalysisRuns,
  type AnalysisReport, type AnalysisRun, type GamePhase, type ReviewMove,
} from "./analysisClient";
import { moveSan } from "./analysisPosition";
import { PositionReview } from "./PositionReview";
import { Button, Panel, PanelBody, PanelHeader, Select } from "./components/ui";

const PAGE_SIZE = 50;

export function Analysis({ runId, moveId, onNavigate }: {
  runId: number | null;
  moveId: number | null;
  onNavigate: (run: number | null, move: number | null) => void;
}) {
  const [runs, setRuns] = useState<AnalysisRun[] | null>(null);
  const [reportResult, setReportResult] = useState<{ key: string; value: AnalysisReport | null; error: string | null } | null>(null);
  const [reviewResult, setReviewResult] = useState<{ key: string; value: ReviewMove | null; error: string | null } | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [phase, setPhase] = useState<GamePhase | "">("");
  const [over, setOver] = useState("");
  const [lossesOnly, setLossesOnly] = useState(false);
  const [offset, setOffset] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const selectedRun = runId ?? runs?.[0]?.id ?? null;
  const reportKey = JSON.stringify([selectedRun, phase, over, lossesOnly, offset, refresh]);
  const reviewKey = JSON.stringify([selectedRun, moveId, refresh]);
  const report = reportResult?.value?.run.id === selectedRun ? reportResult.value : null;
  const loading = reportResult?.key !== reportKey;
  const error = reportResult?.key === reportKey ? reportResult.error : null;
  const review = moveId !== null && reviewResult?.key === reviewKey ? reviewResult.value : null;
  const reviewError = reviewResult?.key === reviewKey ? reviewResult.error : null;

  useEffect(() => {
    let cancelled = false;
    listAnalysisRuns().then(
      (value) => { if (!cancelled) { setRuns(value); setListError(null); } },
      (e: unknown) => { if (!cancelled) setListError(String(e)); },
    );
    return () => { cancelled = true; };
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    if (selectedRun !== null) {
      getAnalysisReport(selectedRun, { phase: phase || undefined, over_cp: over ? Number(over) : undefined,
        losses_only: lossesOnly, offset, limit: PAGE_SIZE }).then(
        (value) => { if (!cancelled) setReportResult({ key: reportKey, value, error: null }); },
        (e: unknown) => { if (!cancelled) setReportResult({ key: reportKey, value: null, error: String(e) }); },
      );
    }
    return () => { cancelled = true; };
  }, [selectedRun, phase, over, lossesOnly, offset, refresh, reportKey]);

  useEffect(() => {
    let cancelled = false;
    if (selectedRun !== null && moveId !== null) {
      getAnalysisMove(selectedRun, moveId).then(
        (value) => { if (!cancelled) setReviewResult({ key: reviewKey, value, error: null }); },
        (e: unknown) => { if (!cancelled) setReviewResult({ key: reviewKey, value: null, error: String(e) }); },
      );
    }
    return () => { cancelled = true; };
  }, [selectedRun, moveId, refresh, reviewKey]);

  const worstPhase = report?.summary.phases.filter((p) => p.avg_cpl !== null)
    .sort((a, b) => b.avg_cpl! - a.avg_cpl!)[0];
  const open = (move: ReviewMove) => onNavigate(move.analysis_run_id, move.id);
  return (
    <div className="grid min-w-0 w-full grid-cols-1 gap-4 text-left">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div><h1 className="text-2xl">Analysis</h1><p className="m-0 mt-1 text-sm text-muted">Find the positions where Bee loses the most.</p></div>
        <Button onClick={() => setRefresh((n) => n + 1)}>Refresh analysis</Button>
      </div>
      {listError && <p role="alert" className="m-0 text-danger">{listError}</p>}
      {!runs && !listError && <p role="status">Loading analysis runs…</p>}
      {runs?.length === 0 && (
        <Panel><PanelBody className="grid gap-2">
          <h2>No analysis runs yet</h2>
          <p className="m-0 text-sm text-muted">Analyze downloaded games with bee-games, then refresh this page. Lab reads the catalog selected by BEE_GAMES_DB.</p>
          <code className="select-text break-all text-xs">bee-games analyze --player beechessjohan --nodes 100000</code>
        </PanelBody></Panel>
      )}
      {runs && runs.length > 0 && (
        <label className="grid min-w-0 max-w-xl grid-cols-1 gap-1 text-sm">Analysis run
          <Select aria-label="Analysis run" value={selectedRun ?? ""} onChange={(e) => { setOffset(0); onNavigate(Number(e.target.value), null); }}>
            {runId !== null && !runs.some((r) => r.id === runId) && <option value={runId}>Run {runId} (unavailable)</option>}
            {runs.map((run) => <option key={run.id} value={run.id}>Run {run.id} · {run.engine} · {run.nodes_per_position?.toLocaleString() ?? "unspecified"} nodes · {new Date(run.created_at).toLocaleDateString()}</option>)}
          </Select>
        </label>
      )}
      {error && <p role="alert" className="m-0 text-danger">{error}</p>}
      {selectedRun !== null && !report && !error && <p role="status">Loading analysis…</p>}
      {report && <>
        <Panel><PanelBody className="grid min-w-0 grid-cols-1 gap-4">
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Metric name="Games analyzed" value={report.summary.games_analyzed.toLocaleString()} />
            <Metric name="Bee moves" value={report.summary.bee_moves.toLocaleString()} />
            <Metric name="Avg CPL" value={average(report.summary.avg_cpl)} />
            <Metric name="Mate-scored moves" value={String(report.summary.mate_moves)} />
          </dl>
          <div className="grid gap-3 sm:grid-cols-3" aria-label="Average CPL by phase">
            {report.summary.phases.map((p) => <div key={p.phase} className="rounded-md border border-border p-3">
              <p className="m-0 text-sm capitalize text-muted">{p.phase}</p>
              <p className="m-0 mt-1 text-xl">{average(p.avg_cpl)} <span className="text-xs text-muted">ACPL · {p.cp_moves} scored moves</span></p>
            </div>)}
          </div>
          {worstPhase && <p className="m-0 text-sm">Highest average loss: <strong className="capitalize">{worstPhase.phase}</strong> ({average(worstPhase.avg_cpl)}cp).</p>}
          <dl className="grid grid-cols-3 gap-4" aria-label="Large mistakes">
            <Metric name=">100cp mistakes" value={String(report.summary.over_100)} />
            <Metric name=">200cp mistakes" value={String(report.summary.over_200)} />
            <Metric name=">400cp blunders" value={String(report.summary.over_400)} />
          </dl>
          <p className="m-0 text-xs text-muted">Averages use {report.summary.cp_moves.toLocaleString()} moves with centipawn scores. Mate transitions are excluded. Counts are cumulative and thresholds are strict.</p>
          {report.summary.score_disagreements > 0 && <p className="m-0 text-xs text-warning">{report.summary.score_disagreements} scored moves match Stockfish’s best move but have a positive score drop between searches. Raw totals include these disagreements; inspect them before classifying errors.</p>}
          {report.summary.games_analyzed === 0 && <p className="m-0 text-sm text-muted">No completed games in this run yet. Refresh after offline analysis completes a game.</p>}
          <details className="text-xs text-muted"><summary className="cursor-pointer">Run configuration</summary>
            <pre className="mt-2 max-w-full overflow-x-auto whitespace-pre-wrap break-all select-text">{report.run.configuration ?? "Legacy run; configuration unavailable."}</pre>
          </details>
        </PanelBody></Panel>
        <Panel id="worst-positions" aria-busy={loading}>
          <PanelHeader><h2>Worst Bee moves</h2></PanelHeader>
          <PanelBody className="grid min-w-0 grid-cols-1 gap-3">
            <div className="flex flex-wrap items-end gap-3">
              <label className="grid gap-1 text-sm">Phase<Select aria-label="Phase" value={phase} onChange={(e) => { setPhase(e.target.value as GamePhase | ""); setOffset(0); }}>
                <option value="">All phases</option><option value="opening">Opening</option><option value="middlegame">Middlegame</option><option value="endgame">Endgame</option>
              </Select></label>
              <label className="grid gap-1 text-sm">Loss threshold<Select aria-label="Loss threshold" value={over} onChange={(e) => { setOver(e.target.value); setOffset(0); }}>
                <option value="">All scored moves</option><option value="100">Over 100cp</option><option value="200">Over 200cp</option><option value="400">Over 400cp</option>
              </Select></label>
              <label className="flex h-9 items-center gap-2 text-sm"><input type="checkbox" checked={lossesOnly} onChange={(e) => { setLossesOnly(e.target.checked); setOffset(0); }} />Only lost games</label>
            </div>
            <p className="m-0 text-xs text-muted">{report.matching_moves.toLocaleString()} matching moves · ply numbers start at 0</p>
            {loading && <p role="status" className="m-0 text-xs text-muted">Updating moves…</p>}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm" aria-label="Worst Bee moves">
                <thead className="text-muted"><tr>{["Game", "Ply", "Phase", "Bee", "Best", "CPL"].map((label) => <th key={label} scope="col" className="border-b border-border px-2 py-2 font-medium">{label}</th>)}</tr></thead>
                <tbody>{report.moves.map((move) => <tr key={move.id} className={move.id === moveId ? "bg-surface-hover" : ""}>
                  <td className="border-b border-border px-2 py-2"><Button disabled={loading} onClick={() => open(move)} aria-label={`Review ${move.game_id} ply ${move.ply}`}>{move.game_id}</Button></td>
                  <td className="border-b border-border px-2 py-2">{move.ply}</td>
                  <td className="border-b border-border px-2 py-2 capitalize">{move.phase}</td>
                  <td className="border-b border-border px-2 py-2">{moveSan(move.fen_before, move.played_move)}</td>
                  <td className="border-b border-border px-2 py-2">{moveSan(move.fen_before, move.best_move)}</td>
                  <td className="border-b border-border px-2 py-2 font-medium text-danger">{move.centipawn_loss}
                    {move.centipawn_loss !== null && move.centipawn_loss > 0 && move.best_move === move.played_move && <span className="block text-xs font-normal text-warning">Score disagreement</span>}
                  </td>
                </tr>)}</tbody>
              </table>
            </div>
            {report.moves.length === 0 && <p className="m-0 text-sm text-muted">No moves match these filters.</p>}
            <div className="flex items-center gap-3">
              <Button disabled={loading || offset === 0} onClick={() => setOffset((n) => Math.max(0, n - PAGE_SIZE))}>Previous 50</Button>
              <span className="text-xs text-muted">{report.matching_moves === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE_SIZE, report.matching_moves)}</span>
              <Button disabled={loading || offset + PAGE_SIZE >= report.matching_moves} onClick={() => setOffset((n) => n + PAGE_SIZE)}>Next 50</Button>
            </div>
          </PanelBody>
        </Panel>
      </>}
      {moveId !== null && !review && !reviewError && <p role="status">Loading position…</p>}
      {reviewError && <p role="alert" className="m-0 text-danger">{reviewError}</p>}
      {review && <PositionReview key={`${review.analysis_run_id}-${review.id}`} move={review} onClose={() => onNavigate(selectedRun, null)} />}
      {report && <Panel>
        <PanelHeader><h2>Biggest single mistakes in losses</h2></PanelHeader>
        <PanelBody className="grid gap-2">
          <p className="m-0 text-xs text-muted">One position per lost game across this run. The percentage is its share of that game’s measured CP loss.</p>
          {report.losses.length === 0 && <p className="m-0 text-sm text-muted">No analyzed losses with centipawn scores.</p>}
          {report.losses.map((loss) => <Button key={loss.worst_move.game_id} className="h-auto flex-wrap justify-between py-2 text-left" onClick={() => open(loss.worst_move)}>
            <span>{loss.worst_move.game_id} · ply {loss.worst_move.ply}</span>
            <span>{loss.worst_move.centipawn_loss}cp · {loss.worst_move_share === null ? "No measured loss" : `${Math.round(loss.worst_move_share * 100)}% of game CPL`}</span>
          </Button>)}
        </PanelBody>
      </Panel>}
    </div>
  );
}

function average(value: number | null) { return value === null ? "—" : value.toFixed(1); }
function Metric({ name, value }: { name: string; value: string }) {
  return <div><dt className="text-xs text-muted">{name}</dt><dd className="m-0 mt-1 text-2xl">{value}</dd></div>;
}
