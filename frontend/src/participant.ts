/**
 * One board slot's participant: a human, or one of the bots, each with
 * its own per-slot config. Independent per slot -- picking the same
 * bot kind for both slots creates two separate Bee Lab-driven engine
 * instances (`Game.tsx`'s `toParticipantRequest`, one per side, see
 * #69/67b), each with its own move-time/Elo/debug settings, not a
 * shared one.
 */

export const MIN_MOVE_TIME_MS = 1;

/** Stockfish's own supported range for `UCI_Elo` (UCI option range). */
export const MIN_STOCKFISH_ELO = 1320;
export const MAX_STOCKFISH_ELO = 3190;

/** A `setoption` value for one discovered UCI option -- see
 * `EngineOption` in `labClient.ts`. Bee's `options` map is keyed by
 * option name (e.g. `"Evaluator"`, `"UseTT"`) and rendered generically
 * from whatever `GET /api/engines/bee/options` reports; unlike
 * Stockfish's `elo` (a first-class Lab/UI concept, not itself a UCI
 * option name this frontend renders generically -- see the
 * design-system milestone's PR 3 scope notes), nothing here is
 * hardcoded to a particular option's name. */
export type EngineOptionValue = string | number | boolean;
export type EngineOptions = Record<string, EngineOptionValue>;

export type Participant =
  | { kind: "human" }
  | { kind: "stockfish"; elo: number; moveTimeMs: number; debug: boolean }
  | { kind: "bee"; moveTimeMs: number; debug: boolean; options: EngineOptions }
  | { kind: "bee-mamba"; moveTimeMs: number };

export type ParticipantKind = Participant["kind"];

export const PARTICIPANT_LABELS: Record<ParticipantKind, string> = {
  human: "Human",
  stockfish: "Stockfish",
  bee: "Bee",
  "bee-mamba": "Bee-Mamba",
};

export function defaultParticipant(kind: ParticipantKind): Participant {
  switch (kind) {
    case "human":
      return { kind: "human" };
    case "stockfish":
      return { kind: "stockfish", elo: 1600, moveTimeMs: 100, debug: false };
    case "bee":
      // No hardcoded `Evaluator: "Positional"` here anymore -- Bee's
      // discovered options (GameSetup's EngineOptionsFields) seed
      // `options` with each option's own reported default once
      // GET /api/engines/bee/options resolves, the same way a form's
      // fields don't have real values before their schema loads.
      return { kind: "bee", moveTimeMs: 100, debug: false, options: {} };
    case "bee-mamba":
      return { kind: "bee-mamba", moveTimeMs: 500 };
  }
}

/**
 * Validates a `Participant`'s own config. Returns an error message for
 * the first invalid field found, or `null` if valid (always valid for
 * "human", which carries no config).
 *
 * Deliberately does not validate the *contents* of a "bee" participant's
 * `options` map (e.g. that a combo's value is one of its advertised
 * `values`) -- those are rendered directly from whatever
 * GET /api/engines/bee/options reports, and the controls generated
 * for each option type (Select for combo, bounded NumberInput for
 * spin) already make an invalid value hard to produce through the UI
 * itself. Lab/the engine remain the authority on whether a given
 * `setoption` value is actually accepted.
 */
export function validateParticipant(participant: Participant): string | null {
  if (participant.kind === "human") return null;

  if (!Number.isFinite(participant.moveTimeMs)) {
    return "Time per move must be a number.";
  }
  if (!Number.isInteger(participant.moveTimeMs)) {
    return "Time per move must be a whole number of milliseconds.";
  }
  if (participant.moveTimeMs < MIN_MOVE_TIME_MS) {
    return "Time per move must be greater than zero.";
  }

  if (participant.kind === "stockfish") {
    if (!Number.isFinite(participant.elo)) {
      return "Stockfish Elo must be a number.";
    }
    if (!Number.isInteger(participant.elo)) {
      return "Stockfish Elo must be a whole number.";
    }
    if (participant.elo < MIN_STOCKFISH_ELO || participant.elo > MAX_STOCKFISH_ELO) {
      return `Stockfish Elo must be between ${MIN_STOCKFISH_ELO} and ${MAX_STOCKFISH_ELO}.`;
    }
  }

  return null;
}
