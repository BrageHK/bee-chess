import type { ExperimentSnapshot } from "./labClient";

/** A minimal, valid `ExperimentSnapshot` fixture for tests that don't
 * care about metadata/stats specifically -- `GameSetup.test.tsx`,
 * `ExperimentView.test.tsx`, and `Dashboard.test.tsx` all previously
 * hand-rolled this same boilerplate (and had to update all three when
 * `metadata`/`stats` were added to the real type); one shared fixture
 * means a future field addition only needs updating here. */
export function experimentSnapshotFixture(overrides: Partial<ExperimentSnapshot> = {}): ExperimentSnapshot {
  return {
    id: "exp-1",
    status: "running",
    label_a: "Baseline",
    label_b: "Candidate",
    requested_games: 20,
    concurrency: 2,
    completed_games: 0,
    wins_a: 0,
    draws: 0,
    wins_b: 0,
    score_a: null,
    elo_diff_a: null,
    games: [],
    metadata: {
      lab_git_commit: "abc123",
      opening_seed: 12345,
      variant_a_argv: ["/path/to/bee"],
      variant_b_argv: ["/path/to/bee"],
      time_control: { type: "move_time", move_time_ms: 100 },
      started_at: "2026-01-01T00:00:00Z",
      finished_at: null,
    },
    stats: {
      avg_game_duration_ms: null,
      avg_plies: null,
      runtime_ms: 0,
      games_per_hour: null,
      variant_a_search: emptySearchStats(),
      variant_b_search: emptySearchStats(),
      timeouts: 0,
    },
    ...overrides,
  };
}

function emptySearchStats() {
  return {
    searches: 0,
    total_nodes: 0,
    avg_nodes: null,
    avg_time_ms: null,
    avg_depth: null,
    max_depth: null,
    effective_nps: null,
    avg_eval_cp: null,
    lmr_attempts: 0,
    lmr_fail_lows: 0,
    lmr_researches: 0,
    lmr_research_rate: null,
    nmp_attempts: 0,
    nmp_cutoffs: 0,
    nmp_cutoff_rate: null,
    delta_attempts: 0,
    delta_pruned: 0,
    delta_prune_rate: null,
    see_attempts: 0,
    see_pruned: 0,
    see_prune_rate: null,
    time_management: null,
  };
}
