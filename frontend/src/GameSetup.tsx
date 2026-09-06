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
import { EngineOptionsFields } from "./EngineOptionsFields";
import { Button } from "./components/ui/Button";
import { Checkbox } from "./components/ui/Checkbox";
import { Field } from "./components/ui/Field";
import { NumberInput } from "./components/ui/NumberInput";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";
import { Select } from "./components/ui/Select";

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
      className="grid justify-items-center gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (!error) onStart(white, black);
      }}
    >
      <div className="flex flex-wrap justify-center gap-6">
        <SlotPicker label="White" participant={white} onChange={setWhite} labUnavailable={labUnavailable} />
        <SlotPicker label="Black" participant={black} onChange={setBlack} labUnavailable={labUnavailable} />
      </div>
      {error && <p className="m-0 text-sm text-danger">{error}</p>}
      <Button type="submit" variant="primary" disabled={error !== null}>
        Start Game
      </Button>
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
    <Panel className="min-w-[220px] text-left">
      <PanelHeader>{label}</PanelHeader>
      <PanelBody className="grid gap-3">
        <Select
          value={participant.kind}
          onChange={(e) => onChange(defaultParticipant(e.target.value as ParticipantKind))}
        >
          {PARTICIPANT_KINDS.map((kind) => (
            <option key={kind} value={kind}>
              {PARTICIPANT_LABELS[kind]}
              {isUnavailable(kind, labUnavailable) ? " (unavailable?)" : ""}
            </option>
          ))}
        </Select>
        {unavailable && (
          <p className="m-0 text-xs leading-snug text-warning">
            {participant.kind === "bee-mamba"
              ? "Bee-Mamba isn't available yet during the Bee Lab migration (see #66/#70)."
              : "Bee Lab doesn't seem to be running (see lab/README.md)."}{" "}
            You can still try to start the game.
          </p>
        )}
        <ParticipantFields participant={participant} onChange={onChange} />
      </PanelBody>
    </Panel>
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
    <div className="grid w-full gap-3">
      {participant.kind === "stockfish" && (
        <Field label="Elo">
          <NumberInput
            value={participant.elo}
            min={MIN_STOCKFISH_ELO}
            max={MAX_STOCKFISH_ELO}
            step={1}
            onChange={(value) => onChange({ ...participant, elo: value })}
          />
        </Field>
      )}
      <Field label="Time per move (ms)">
        <NumberInput
          value={participant.moveTimeMs}
          min={MIN_MOVE_TIME_MS}
          step={1}
          onChange={(value) => onChange({ ...participant, moveTimeMs: value })}
        />
      </Field>
      {"debug" in participant && (
        <Checkbox
          label="Debug logging"
          checked={participant.debug}
          onChange={(e) => onChange({ ...participant, debug: e.target.checked })}
        />
      )}
      {participant.kind === "bee" && (
        <EngineOptionsFields
          engineName="bee"
          values={participant.options}
          onChange={(options) => onChange({ ...participant, options })}
        />
      )}
    </div>
  );
}
