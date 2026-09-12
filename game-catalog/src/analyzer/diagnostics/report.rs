use super::{FixtureResult, Sample};
use serde_json::{json, Value};

fn acceptable(result: &FixtureResult, sample: &Sample) -> Option<bool> {
    sample
        .best_move
        .as_ref()
        .and_then(|m| result.judgments.get(m))
        .and_then(|j| j.acceptable)
}

/// Counts fixtures, not plies. Exact duplicate FENs are disclosed separately;
/// histories are preserved because they can change repetition behavior.
pub fn summary(results: &[FixtureResult], depths: &[u32], variants: &[String]) -> Value {
    let mut totals = Vec::new();
    let categories: std::collections::BTreeSet<_> = std::iter::once("all")
        .chain(results.iter().map(|r| r.fixture.category.as_str()))
        .collect();
    for bucket in ["preventable", "already_losing"] {
        for &category in &categories {
            let fixtures: Vec<_> = results
                .iter()
                .filter(|r| {
                    r.bucket == bucket && (category == "all" || r.fixture.category == category)
                })
                .collect();
            for variant in variants {
                for &depth in depths {
                    let mut completed = 0;
                    let mut accepted = 0;
                    let mut unassessed = 0;
                    let mut timed_out = 0;
                    let mut nodes = 0u64;
                    for result in &fixtures {
                        if let Some(s) = result
                            .samples
                            .iter()
                            .find(|s| s.variant == *variant && s.requested_depth == depth)
                        {
                            if s.status == "timed_out" {
                                timed_out += 1;
                                continue;
                            }
                            completed += 1;
                            nodes += s.nodes.unwrap_or(0);
                            match acceptable(result, s) {
                                Some(true) => accepted += 1,
                                None => unassessed += 1,
                                _ => {}
                            }
                        }
                    }
                    totals.push(json!({"bucket": bucket, "category": category, "variant": variant, "depth": depth,
                        "fixtures": fixtures.len(), "completed": completed, "acceptable": accepted,
                        "unassessed": unassessed, "timed_out": timed_out, "completed_nodes": nodes}));
                }
            }
        }
    }
    let mut positions = Vec::new();
    for result in results {
        let mut by_variant = Vec::new();
        for variant in variants {
            let mut samples: Vec<_> = result
                .samples
                .iter()
                .filter(|s| s.variant == *variant)
                .collect();
            samples.sort_by_key(|s| s.requested_depth);
            let accepted: Vec<_> = samples
                .iter()
                .filter(|s| acceptable(result, s) == Some(true))
                .map(|s| s.requested_depth)
                .collect();
            let failed: Vec<_> = samples
                .iter()
                .filter(|s| acceptable(result, s) == Some(false))
                .map(|s| s.requested_depth)
                .collect();
            let deepest = samples.iter().rev().find(|s| s.status == "completed");
            by_variant.push(json!({"variant": variant, "first_acceptable_depth": accepted.first(),
                "acceptable_depths": accepted,
                "deepest_completed_depth": deepest.map(|s| s.requested_depth),
                "deepest_acceptable": deepest.and_then(|s| acceptable(result, s)),
                "search_recovery": failed.iter().any(|f| accepted.iter().any(|a| a > f)),
                "later_regression": accepted.iter().any(|a| failed.iter().any(|f| f > a)),
                "timed_out_depths": samples.iter().filter(|s| s.status == "timed_out").map(|s| s.requested_depth).collect::<Vec<_>>(),
            }));
        }
        let mut toggle_recoveries = Vec::new();
        for baseline in result
            .samples
            .iter()
            .filter(|s| s.variant == "baseline" && acceptable(result, s) == Some(false))
        {
            for alternative in result.samples.iter().filter(|s| {
                s.variant != "baseline"
                    && s.requested_depth == baseline.requested_depth
                    && acceptable(result, s) == Some(true)
            }) {
                toggle_recoveries.push(
                    json!({"variant": alternative.variant, "depth": baseline.requested_depth,
                    "baseline_nodes": baseline.nodes, "variant_nodes": alternative.nodes}),
                );
            }
        }
        let historical_move = result.judgments.get(&result.fixture.bee_move);
        let reference_bucket = result.reference.score.cp.map(super::bucket);
        let reference_transition = result
            .reference
            .score
            .cp
            .zip(historical_move.and_then(|j| j.score.cp))
            .map(|(before, after)| format!("{}_to_{}", super::state(before), super::state(after)));
        positions.push(json!({"fixture": result.fixture.id, "category": result.fixture.category,
            "subcategory": result.fixture.review["subcategory"], "bucket": result.bucket,
            "state_transition": result.state_transition, "historical_regret_cp": result.fixture.analysis.regret_cp,
            "reference_score": result.reference.score,
            "reference_bucket": reference_bucket, "reference_state_transition": reference_transition,
            "recorded_candidate": result.fixture.stockfish_candidate, "reference_candidate": result.reference.best_move,
            "reference_changed": result.fixture.stockfish_candidate != result.reference.best_move,
            "historical_move_judgment": historical_move,
            "recorded_candidate_judgment": result.judgments.get(&result.fixture.stockfish_candidate),
            "variants": by_variant, "same_depth_toggle_recoveries": toggle_recoveries,
        }));
    }
    positions.sort_by_key(|p| {
        (
            p["bucket"] != "preventable",
            !matches!(
                p["state_transition"].as_str(),
                Some("equal_to_losing" | "winning_to_equal" | "winning_to_losing")
            ),
            -p["historical_regret_cp"].as_i64().unwrap_or(0),
        )
    });
    json!({"schema_version": 1, "fixtures": results.len(),
        "unique_fens": results.iter().map(|r| &r.fixture.fen).collect::<std::collections::HashSet<_>>().len(),
        "preventable": results.iter().filter(|r| r.bucket == "preventable").count(),
        "already_losing": results.iter().filter(|r| r.bucket == "already_losing").count(),
        "interpretation": "Stockfish regret is an acceptance proxy, not proof that the human expected property is met. Depth ablations spend different nodes. Persistent mismatches and timeouts do not establish an evaluation defect.",
        "totals": totals, "positions": positions})
}
