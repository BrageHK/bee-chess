import { useEffect, useState } from "react";
import { checkLabAvailable } from "./labClient";
import {
  MAX_STOCKFISH_ELO,
  MIN_MOVE_TIME_MS,
  MIN_STOCKFISH_ELO,
  PARTICIPANT_LABELS,
  defaultParticipant,
  validateParticipant,
  type Participant,
  type ParticipantKind,
} from "./participant";

const PARTICIPANT_KINDS: ParticipantKind[] = ["human", "stockfish", "bee", "bee-mamba"];

/**
 * Pick a participant (human or one of the bots) for each side, like
 * choosing teams before a match: two independent slot pickers side by
 * side, each showing that bot's own config (Elo, move time, debug)
 * once selected.
 *
 * Per #69/67b: every engine-driven game goes through Bee Lab now, so
 * "is this bot available" reduces to "is Lab itself reachable" --
 * Lab refuses to start without both Stockfish and Bee (see
 * `checkLabAvailable`'s docs), so a single reachability check on
 * mount covers both. Bee-Mamba has no Lab-side engine yet (#66/#70)
 * and is always shown unavailable, regardless of Lab's own
 * reachability -- picking it still isn't blocked here (the same "warn,
 * don't block" philosophy as before Lab existed), but `Game.tsx`
 * refuses to actually start a game with it.
 */
export function GameSetup({
  onStart,
}: {
  onStart: (white: Participant, black: Participant) => void;
}) {
  const [white, setWhite] = useState<Participant>(defaultParticipant("bee"));
  const [black, setBlack] = useState<Participant>(defaultParticipant("stockfish"));
  const [labUnavailable, setLabUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void checkLabAvailable().then((available) => {
      if (!cancelled) setLabUnavailable(!available);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const whiteError = validateParticipant(white);
  const blackError = validateParticipant(black);
  const error = whiteError ?? blackError;

  return (
    <form
      style={{ display: "grid", gap: 16, justifyItems: "center" }}
      onSubmit={(e) => {
        e.preventDefault();
        if (!error) onStart(white, black);
      }}
    >
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center" }}>
        <SlotPicker label="White" participant={white} onChange={setWhite} labUnavailable={labUnavailable} />
        <SlotPicker label="Black" participant={black} onChange={setBlack} labUnavailable={labUnavailable} />
      </div>
      {error && <p style={{ color: "crimson", margin: 0 }}>{error}</p>}
      <button type="submit" disabled={error !== null}>
        Start Game
      </button>
    </form>
  );
}

/** Whether `kind` is currently unavailable, given whether Lab itself
 * responded -- see the component doc comment above. */
function isUnavailable(kind: ParticipantKind, labUnavailable: boolean): boolean {
  if (kind === "human") return false;
  if (kind === "bee-mamba") return true;
  return labUnavailable;
}

function SlotPicker({
  label,
  participant,
  onChange,
  labUnavailable,
}: {
  label: string;
  participant: Participant;
  onChange: (participant: Participant) => void;
  labUnavailable: boolean;
}) {
  const unavailable = isUnavailable(participant.kind, labUnavailable);

  return (
    <fieldset
      style={{
        display: "grid",
        gap: 8,
        justifyItems: "start",
        padding: 12,
        borderRadius: 8,
        minWidth: 220,
      }}
    >
      <legend>{label}</legend>
      <select
        value={participant.kind}
        onChange={(e) => onChange(defaultParticipant(e.target.value as ParticipantKind))}
      >
        {PARTICIPANT_KINDS.map((kind) => (
          <option key={kind} value={kind}>
            {PARTICIPANT_LABELS[kind]}
            {isUnavailable(kind, labUnavailable) ? " (unavailable?)" : ""}
          </option>
        ))}
      </select>
      {unavailable && (
        <p style={{ color: "#b45309", margin: 0, fontSize: 13 }}>
          {participant.kind === "bee-mamba"
            ? "Bee-Mamba isn't available yet during the Bee Lab migration (see #66/#70)."
            : "Bee Lab doesn't seem to be running (see lab/README.md)."}{" "}
          You can still try to start the game.
        </p>
      )}
      <ParticipantFields participant={participant} onChange={onChange} />
    </fieldset>
  );
}

function ParticipantFields({
  participant,
  onChange,
}: {
  participant: Participant;
  onChange: (participant: Participant) => void;
}) {
  if (participant.kind === "human") return null;

  return (
    <div style={{ display: "grid", gap: 8, width: "100%" }}>
      {participant.kind === "stockfish" && (
        <label style={{ display: "grid", gap: 4 }}>
          Elo
          <input
            type="number"
            value={participant.elo}
            min={MIN_STOCKFISH_ELO}
            max={MAX_STOCKFISH_ELO}
            step={1}
            onChange={(e) => onChange({ ...participant, elo: Number(e.target.value) })}
          />
        </label>
      )}
      <label style={{ display: "grid", gap: 4 }}>
        Time per move (ms)
        <input
          type="number"
          value={participant.moveTimeMs}
          min={MIN_MOVE_TIME_MS}
          step={1}
          onChange={(e) => onChange({ ...participant, moveTimeMs: Number(e.target.value) })}
        />
      </label>
      {"debug" in participant && (
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={participant.debug}
            onChange={(e) => onChange({ ...participant, debug: e.target.checked })}
          />
          Debug logging
        </label>
      )}
    </div>
  );
}
