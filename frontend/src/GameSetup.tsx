import { useEffect, useState } from "react";
import { checkBotAvailable, type BotKind } from "./engine";
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
const BOT_KINDS: BotKind[] = ["stockfish", "bee", "bee-mamba"];

/**
 * Pick a participant (human or one of the bots) for each side, like
 * choosing teams before a match: two independent slot pickers side by
 * side, each showing that bot's own config (Elo, move time, debug)
 * once selected. A bot that fails a quick reachability check on
 * mount is still selectable (the bridge might start it by the time
 * Start is clicked) but shown with a warning, rather than only
 * failing after the user has already configured a whole game around
 * it -- see `checkBotAvailable`.
 */
export function GameSetup({
  onStart,
}: {
  onStart: (white: Participant, black: Participant) => void;
}) {
  const [white, setWhite] = useState<Participant>(defaultParticipant("bee"));
  const [black, setBlack] = useState<Participant>(defaultParticipant("stockfish"));
  const [unavailable, setUnavailable] = useState<Partial<Record<BotKind, boolean>>>({});

  useEffect(() => {
    let cancelled = false;
    for (const kind of BOT_KINDS) {
      void checkBotAvailable(kind).then((available) => {
        if (!cancelled) setUnavailable((prev) => ({ ...prev, [kind]: !available }));
      });
    }
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
        <SlotPicker
          label="White"
          participant={white}
          onChange={setWhite}
          unavailable={unavailable}
        />
        <SlotPicker
          label="Black"
          participant={black}
          onChange={setBlack}
          unavailable={unavailable}
        />
      </div>
      {error && <p style={{ color: "crimson", margin: 0 }}>{error}</p>}
      <button type="submit" disabled={error !== null}>
        Start Game
      </button>
    </form>
  );
}

function SlotPicker({
  label,
  participant,
  onChange,
  unavailable,
}: {
  label: string;
  participant: Participant;
  onChange: (participant: Participant) => void;
  unavailable: Partial<Record<BotKind, boolean>>;
}) {
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
            {kind !== "human" && unavailable[kind] ? " (unavailable?)" : ""}
          </option>
        ))}
      </select>
      {participant.kind !== "human" && unavailable[participant.kind] && (
        <p style={{ color: "#b45309", margin: 0, fontSize: 13 }}>
          {PARTICIPANT_LABELS[participant.kind]} doesn't seem to be running on the bridge.
          {participant.kind === "bee-mamba" &&
            " It needs a trained checkpoint (see bridge/server.py)."}{" "}
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
