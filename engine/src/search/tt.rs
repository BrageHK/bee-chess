//! TT storage and lifetime policy. PerSearch retains the original replacement
//! behavior; PerGame evicts individual entries using age and search depth.

use std::collections::HashMap;

use crate::chess::{Move, Position};

use super::Score;

const MAX_TT_ENTRIES: usize = 1 << 20;
type TtKey = (u64, u32, u8);

/// Whether Engine starts each search with an empty TT or retains completed
/// work across moves. Per-game reuse is the engine default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TtReuse {
    PerSearch,
    #[default]
    PerGame,
}

impl TtReuse {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "persearch" => Some(Self::PerSearch),
            "pergame" => Some(Self::PerGame),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
pub(super) struct TtEntry {
    pub depth: u32,
    pub score: Score,
    pub bound: Bound,
    pub best_move: Option<Move>,
    pub history: u64,
    generation: u64,
}

impl TtEntry {
    pub fn new(
        depth: u32,
        score: Score,
        bound: Bound,
        best_move: Option<Move>,
        history: u64,
    ) -> Self {
        Self {
            depth,
            score,
            bound,
            best_move,
            history,
            generation: 0,
        }
    }
}

pub(super) struct TranspositionTable {
    pub policy: TtReuse,
    entries: HashMap<TtKey, TtEntry>,
    generation: u64,
    // PerGame only: a rotating sample of keys gives bounded eviction work
    // without scanning the entire table on a search's critical path.
    keys: Vec<TtKey>,
    eviction_cursor: usize,
    capacity: usize,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self {
            // Stateless search helpers keep their original table behavior.
            // Engine selects its own default policy when it creates a context.
            policy: TtReuse::PerSearch,
            entries: HashMap::new(),
            generation: 0,
            keys: Vec::new(),
            eviction_cursor: 0,
            capacity: MAX_TT_ENTRIES,
        }
    }
}

impl TranspositionTable {
    pub fn clear(&mut self) {
        *self = Self {
            policy: self.policy,
            capacity: self.capacity,
            ..Self::default()
        };
    }

    pub fn begin_search(&mut self) {
        if self.policy == TtReuse::PerSearch || self.generation == u64::MAX {
            self.clear();
        }
        self.generation += 1;
    }

    pub fn probe(&mut self, key: &TtKey) -> Option<TtEntry> {
        let entry = self.entries.get_mut(key)?;
        entry.generation = self.generation;
        Some(*entry)
    }

    /// Positions with identical boards and repetition counts can still have
    /// different *other* repeatable positions in their histories. Persistent
    /// scores require the same reversible history; move ordering does not.
    /// A commutative fingerprint preserves transpositions with the same counts.
    pub fn history_key(&self, position: &Position, path: &[u64]) -> u64 {
        if self.policy == TtReuse::PerSearch {
            return 0;
        }
        path.iter()
            .rev()
            .take(position.halfmove_clock() as usize + 1)
            .fold(0, |sum, &hash| {
                let mut mixed = hash.wrapping_add(0x9e3779b97f4a7c15);
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
                sum.wrapping_add(mixed ^ (mixed >> 31))
            })
    }

    fn priority(&self, entry: &TtEntry) -> i64 {
        // One search of age costs eight plies of replacement priority.
        i64::from(entry.depth)
            - self
                .generation
                .saturating_sub(entry.generation)
                .min(u32::MAX as u64) as i64
                * 8
    }

    pub fn store(&mut self, key: TtKey, mut entry: TtEntry) {
        entry.generation = self.generation;
        if self.policy == TtReuse::PerSearch {
            // Preserve the baseline's depth preference and flush-at-capacity.
            if self
                .entries
                .get(&key)
                .is_none_or(|old| entry.depth >= old.depth)
            {
                if self.entries.len() >= self.capacity {
                    self.entries.clear();
                }
                self.entries.insert(key, entry);
            }
            return;
        }

        if let Some(old) = self.entries.get(&key) {
            if old.history != entry.history || self.priority(&entry) >= self.priority(old) {
                self.entries.insert(key, entry);
            }
            return;
        }
        if self.entries.len() < self.capacity {
            self.keys.push(key);
        } else {
            let mut victim = self.eviction_cursor;
            for _ in 0..8.min(self.keys.len()) {
                let candidate = self.eviction_cursor;
                if self.priority(&self.entries[&self.keys[candidate]])
                    < self.priority(&self.entries[&self.keys[victim]])
                {
                    victim = candidate;
                }
                self.eviction_cursor = (candidate + 1) % self.keys.len();
            }
            if self.priority(&entry) < self.priority(&self.entries[&self.keys[victim]]) {
                return;
            }
            self.entries.remove(&self.keys[victim]);
            self.keys[victim] = key;
        }
        self.entries.insert(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(depth: u32) -> TtEntry {
        TtEntry::new(depth, 123, Bound::Exact, None, 0)
    }

    fn small_table() -> TranspositionTable {
        TranspositionTable {
            policy: TtReuse::PerGame,
            capacity: 2,
            ..Default::default()
        }
    }

    #[test]
    fn persistent_replacement_prefers_depth_and_age_without_flushing() {
        let mut table = small_table();
        table.begin_search();
        table.store((1, 0, 1), entry(8));
        table.store((2, 0, 1), entry(2));
        table.store((3, 0, 1), entry(1));
        assert!(
            table.probe(&(3, 0, 1)).is_none(),
            "retain deeper recent work"
        );

        table.begin_search();
        table.probe(&(1, 0, 1)); // Deep entry is also recently used.
        table.store((3, 0, 1), entry(1));
        assert!(table.probe(&(1, 0, 1)).is_some());
        assert!(table.probe(&(2, 0, 1)).is_none());
        assert!(table.probe(&(3, 0, 1)).is_some());
        assert_eq!(table.entries.len(), 2);
        assert_eq!(table.keys.len(), 2);
    }

    #[test]
    fn updating_an_existing_key_does_not_evict_another_entry() {
        let mut table = small_table();
        table.store((1, 0, 1), entry(4));
        table.store((2, 0, 1), entry(4));
        table.store((1, 0, 1), entry(5));
        assert_eq!(table.probe(&(1, 0, 1)).unwrap().depth, 5);
        assert_eq!(table.probe(&(2, 0, 1)).unwrap().depth, 4);
        assert_eq!(table.keys.len(), 2);
    }

    #[test]
    fn history_fingerprint_tracks_other_repeatable_positions_and_counts() {
        let table = small_table();
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 3 10").unwrap();
        assert_ne!(
            table.history_key(&position, &[1, 2, 3]),
            table.history_key(&position, &[1, 4, 3])
        );
        assert_ne!(
            table.history_key(&position, &[1, 1, 3]),
            table.history_key(&position, &[1, 3])
        );
        assert_eq!(
            table.history_key(&position, &[1, 2, 3]),
            table.history_key(&position, &[2, 1, 3])
        );
        // Pawn moves/captures end the relevant repetition history.
        let reset = Position::startpos();
        assert_eq!(
            table.history_key(&reset, &[1, 2, 3]),
            table.history_key(&reset, &[4, 5, 3])
        );
    }
}
