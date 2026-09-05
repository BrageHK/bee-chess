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

export type EngineSettingValue = string | number | boolean;
export type EngineSettings = Record<string, EngineSettingValue>;

export type EngineSettingDefinition = {
  key: string;
  label: string;
  description: string;
  control:
    | { type: "select"; options: { value: string; label: string }[] }
    | { type: "number"; min?: number; max?: number; step?: number }
    | { type: "boolean" };
};

/** UI metadata for engine-specific UCI options. Adding another Bee variant
 * or parameter only requires adding definitions here; the form renderer and
 * Lab request mapping remain unchanged. */
export const ENGINE_SETTING_DEFINITIONS: Partial<Record<ParticipantKind, EngineSettingDefinition[]>> = {
  bee: [
    {
      key: "Evaluator",
      label: "Evaluator",
      description: "Choose how Bee scores positions at the leaves of its search.",
      control: {
        type: "select",
        options: [
          { value: "Positional", label: "Positional (recommended)" },
          { value: "Material", label: "Material only" },
        ],
      },
    },
  ],
};

export type Participant =
  | { kind: "human" }
  | { kind: "stockfish"; elo: number; moveTimeMs: number; debug: boolean }
  | { kind: "bee"; moveTimeMs: number; debug: boolean; settings: EngineSettings }
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
      return { kind: "bee", moveTimeMs: 100, debug: false, settings: { Evaluator: "Positional" } };
    case "bee-mamba":
      return { kind: "bee-mamba", moveTimeMs: 500 };
  }
}

/**
 * Validates a `Participant`'s own config. Returns an error message for
 * the first invalid field found, or `null` if valid (always valid for
 * "human", which carries no config).
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

  if (participant.kind === "bee" && !["Positional", "Material"].includes(String(participant.settings.Evaluator))) {
    return "Choose a valid Bee evaluator.";
  }

  return null;
}
