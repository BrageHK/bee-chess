import { useState } from "react";
import { createExperiment, type CreateExperimentRequest } from "./labClient";
import type { EngineOptions } from "./participant";
import { EngineOptionsFields } from "./EngineOptionsFields";
import { Button } from "./components/ui/Button";
import { Field } from "./components/ui/Field";
import { Input } from "./components/ui/Input";
import { NumberInput } from "./components/ui/NumberInput";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";

const DEFAULT_GAMES = 20;
const DEFAULT_CONCURRENCY = 2;
const DEFAULT_MOVE_TIME_MS = 100;

type VariantForm = {
  label: string;
  options: EngineOptions;
};

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
  const [moveTimeMs, setMoveTimeMs] = useState(DEFAULT_MOVE_TIME_MS);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const validationError =
    variantA.label.trim() === "" || variantB.label.trim() === ""
      ? "Both variants need a label."
      : !Number.isInteger(games) || games < 2 || games % 2 !== 0
        ? "Games must be a positive even number so every game has a color-swapped partner."
        : !Number.isInteger(concurrency) || concurrency < 1 || concurrency > games
          ? "Concurrency must be a whole number between 1 and the number of games."
          : !Number.isInteger(moveTimeMs) || moveTimeMs < 1
          ? "Move time must be a whole number of milliseconds, at least 1."
          : null;

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
          moveTimeMs,
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
        <Field label="Move time (ms)">
          <NumberInput value={moveTimeMs} min={1} step={1} onChange={setMoveTimeMs} />
        </Field>
        <Field label="Concurrent games">
          <NumberInput value={concurrency} min={1} max={games} step={1} onChange={setConcurrency} />
        </Field>
      </div>
      {(validationError ?? error) && <p className="m-0 text-sm text-danger">{validationError ?? error}</p>}
      <Button type="submit" variant="primary" disabled={validationError !== null || starting}>
        {starting ? "Starting…" : "Run experiment"}
      </Button>
    </form>
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
