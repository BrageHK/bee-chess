import type { EngineOptions } from "./participant";
import { Field } from "./components/ui/Field";
import { NumberInput } from "./components/ui/NumberInput";

// Mirrors mamba_mcts_batched_uci's own DEFAULT_SIMULATIONS/DEFAULT_BATCH_SIZE
// and their `option ... spin` min/max (main.rs). Hardcoded here instead of
// discovered via GET /api/engines/bee-mamba/options -- that endpoint spawns
// the engine fresh (loading the Torch model) just to read its option list,
// which is slow enough to stall a form that renders it.
export const MIN_MAMBA_SIMULATIONS = 1;
export const MAX_MAMBA_SIMULATIONS = 100_000;
export const MIN_MAMBA_BATCH_SIZE = 1;
export const MAX_MAMBA_BATCH_SIZE = 512;
export const DEFAULT_MAMBA_SIMULATIONS = 800;
export const DEFAULT_MAMBA_BATCH_SIZE = 64;

/** Static Simulations/BatchSize fields for Bee-Mamba -- see the constants
 * above for why these aren't discovered generically like EngineOptionsFields
 * does for "bee". Shared between `GameSetup` (one Bee-Mamba participant) and
 * `ExperimentSetup` (a Bee-Mamba variant compared against any other bot). */
export function MambaOptionsFields({
  values,
  onChange,
}: {
  values: EngineOptions;
  onChange: (values: EngineOptions) => void;
}) {
  const simulations =
    typeof values.Simulations === "number" ? values.Simulations : DEFAULT_MAMBA_SIMULATIONS;
  const batchSize = typeof values.BatchSize === "number" ? values.BatchSize : DEFAULT_MAMBA_BATCH_SIZE;

  return (
    <div className="grid w-full gap-3">
      <Field label="Simulations">
        <NumberInput
          value={simulations}
          min={MIN_MAMBA_SIMULATIONS}
          max={MAX_MAMBA_SIMULATIONS}
          step={1}
          onChange={(value) => onChange({ ...values, Simulations: value })}
        />
      </Field>
      <Field label="Batch size">
        <NumberInput
          value={batchSize}
          min={MIN_MAMBA_BATCH_SIZE}
          max={MAX_MAMBA_BATCH_SIZE}
          step={1}
          onChange={(value) => onChange({ ...values, BatchSize: value })}
        />
      </Field>
    </div>
  );
}
