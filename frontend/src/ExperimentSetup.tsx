import { useState } from "react";
import { createExperiment, type CreateExperimentRequest, type TimeControl } from "./labClient";
import type { EngineOptions } from "./participant";
import { EngineOptionsFields } from "./EngineOptionsFields";
import { Button } from "./components/ui/Button";
import { Field } from "./components/ui/Field";
import { Input } from "./components/ui/Input";
import { NumberInput } from "./components/ui/NumberInput";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";
import { Select } from "./components/ui/Select";

const DEFAULT_GAMES = 20;
const DEFAULT_CONCURRENCY = 2;
const DEFAULT_MOVE_TIME_MS = 100;
const DEFAULT_FISCHER_INITIAL_S = 60;
const DEFAULT_FISCHER_INCREMENT_S = 1;

type VariantForm = {
  label: string;
  options: EngineOptions;
};

/** Local editable form state for the time-control picker below --
 * kept as plain numbers in whichever unit the field displays (minutes/
 * seconds shown to the user, converted to milliseconds only when
 * building the actual `TimeControl` sent to Lab), rather than storing
 * a `TimeControl` directly and needing to convert back and forth on
 * every keystroke. */
type TimeControlForm =
  | { kind: "move_time"; moveTimeMs: number }
  | { kind: "fischer"; initialMinutes: number; incrementSeconds: number };

function toTimeControl(form: TimeControlForm): TimeControl {
  return form.kind === "move_time"
    ? { type: "move_time", move_time_ms: form.moveTimeMs }
    : {
        type: "fischer",
        initial_ms: Math.round(form.initialMinutes * 60_000),
        increment_ms: Math.round(form.incrementSeconds * 1000),
      };
}

/**
 * Configure and start a Bee-vs-Bee A/B experiment (see `lab::
 * experiment`'s module docs -- v1 is deliberately Bee-only, not a
 * general engine picker): two labeled variants, each with Bee's own
 * discovered UCI options rendered generically (same
 * `EngineOptionsFields` `GameSetup` uses for its Bee slot), plus how
 * many paired games to run and the shared move-time budget.
 */
export function ExperimentSetup({ onStarted }: { onStarted: (experimentId: string) => void }) {
  const [variantA, setVariantA] = useState<VariantForm>({ label: "Baseline", options: {} });
  const [variantB, setVariantB] = useState<VariantForm>({ label: "Candidate", options: {} });
  const [games, setGames] = useState(DEFAULT_GAMES);
  const [concurrency, setConcurrency] = useState(DEFAULT_CONCURRENCY);
  const [timeControl, setTimeControl] = useState<TimeControlForm>({
    kind: "move_time",
    moveTimeMs: DEFAULT_MOVE_TIME_MS,
  });
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const timeControlError =
    timeControl.kind === "move_time"
      ? !Number.isInteger(timeControl.moveTimeMs) || timeControl.moveTimeMs < 1
        ? "Move time must be a whole number of milliseconds, at least 1."
        : null
      : !Number.isFinite(timeControl.initialMinutes) || timeControl.initialMinutes <= 0
        ? "Initial time must be greater than zero."
        : !Number.isFinite(timeControl.incrementSeconds) || timeControl.incrementSeconds < 0
          ? "Increment can't be negative."
          : null;

  const validationError =
    variantA.label.trim() === "" || variantB.label.trim() === ""
      ? "Both variants need a label."
      : !Number.isInteger(games) || games < 2 || games % 2 !== 0
        ? "Games must be a positive even number so every game has a color-swapped partner."
        : !Number.isInteger(concurrency) || concurrency < 1 || concurrency > games
          ? "Concurrency must be a whole number between 1 and the number of games."
          : timeControlError;

  return (
    <form
      className="grid w-full max-w-2xl justify-items-center gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (validationError) return;

        setStarting(true);
        setError(null);
        const request: CreateExperimentRequest = {
          variantA: { label: variantA.label, options: variantA.options },
          variantB: { label: variantB.label, options: variantB.options },
          games,
          concurrency,
          timeControl: toTimeControl(timeControl),
        };
        createExperiment(request).then(
          (snapshot) => onStarted(snapshot.id),
          (err: unknown) => {
            setStarting(false);
            setError(err instanceof Error ? err.message : String(err));
          },
        );
      }}
    >
      <div className="flex w-full flex-wrap justify-center gap-6">
        <VariantPanel label="Variant A" variant={variantA} onChange={setVariantA} />
        <VariantPanel label="Variant B" variant={variantB} onChange={setVariantB} />
      </div>
      <div className="flex flex-wrap justify-center gap-4">
        <Field label="Games">
          <NumberInput value={games} min={2} step={2} onChange={setGames} />
        </Field>
        <Field label="Concurrent games">
          <NumberInput value={concurrency} min={1} max={games} step={1} onChange={setConcurrency} />
        </Field>
      </div>
      <TimeControlFields value={timeControl} onChange={setTimeControl} />
      {(validationError ?? error) && <p className="m-0 text-sm text-danger">{validationError ?? error}</p>}
      <Button type="submit" variant="primary" disabled={validationError !== null || starting}>
        {starting ? "Starting…" : "Run experiment"}
      </Button>
    </form>
  );
}

/** Both variants play under the same clock -- see `TimeControl`'s
 * docs on why time control is the experiment's own configuration, not
 * either variant's. Deliberately simple (a type dropdown plus the
 * one or two numbers each type needs), matching every other numeric
 * field on this form rather than a fancier picker. */
function TimeControlFields({
  value,
  onChange,
}: {
  value: TimeControlForm;
  onChange: (value: TimeControlForm) => void;
}) {
  return (
    <div className="flex flex-wrap items-end justify-center gap-4">
      <Field label="Time control">
        <Select
          value={value.kind}
          onChange={(e) =>
            onChange(
              e.target.value === "fischer"
                ? {
                    kind: "fischer",
                    initialMinutes: DEFAULT_FISCHER_INITIAL_S / 60,
                    incrementSeconds: DEFAULT_FISCHER_INCREMENT_S,
                  }
                : { kind: "move_time", moveTimeMs: DEFAULT_MOVE_TIME_MS },
            )
          }
        >
          <option value="move_time">Fixed move time</option>
          <option value="fischer">Fischer (initial + increment)</option>
        </Select>
      </Field>
      {value.kind === "move_time" ? (
        <Field label="Move time (ms)">
          <NumberInput
            value={value.moveTimeMs}
            min={1}
            step={1}
            onChange={(moveTimeMs) => onChange({ kind: "move_time", moveTimeMs })}
          />
        </Field>
      ) : (
        <>
          <Field label="Initial (minutes)">
            <NumberInput
              value={value.initialMinutes}
              min={0.1}
              step={0.5}
              onChange={(initialMinutes) => onChange({ ...value, initialMinutes })}
            />
          </Field>
          <Field label="Increment (seconds)">
            <NumberInput
              value={value.incrementSeconds}
              min={0}
              step={1}
              onChange={(incrementSeconds) => onChange({ ...value, incrementSeconds })}
            />
          </Field>
        </>
      )}
    </div>
  );
}

function VariantPanel({
  label,
  variant,
  onChange,
}: {
  label: string;
  variant: VariantForm;
  onChange: (variant: VariantForm) => void;
}) {
  return (
    <Panel className="min-w-[240px] flex-1 text-left">
      <PanelHeader>{label}</PanelHeader>
      <PanelBody className="grid gap-3">
        <Field label="Label">
          <Input
            value={variant.label}
            onChange={(e) => onChange({ ...variant, label: e.target.value })}
          />
        </Field>
        <EngineOptionsFields
          engineName="bee"
          values={variant.options}
          onChange={(options) => onChange({ ...variant, options })}
        />
      </PanelBody>
    </Panel>
  );
}
