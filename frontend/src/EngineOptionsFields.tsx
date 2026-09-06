import { useEffect, useState } from "react";
import { getEngineOptions, type EngineOption } from "./labClient";
import type { EngineOptionValue, EngineOptions } from "./participant";
import { Checkbox } from "./components/ui/Checkbox";
import { Field } from "./components/ui/Field";
import { Input } from "./components/ui/Input";
import { NumberInput } from "./components/ui/NumberInput";
import { Select } from "./components/ui/Select";

/**
 * Renders whichever UCI options an engine happens to advertise
 * (`GET /api/engines/:name/options`, see `labClient.ts`'s
 * `EngineOption`) as form controls -- `check` -> `Checkbox`, `spin` ->
 * a bounded `NumberInput`, `combo` -> `Select`, `string` -> a bare
 * text `Input`. Nothing here is hardcoded to a particular option's
 * name: adding `option name UseLMR type check default true` to Bee
 * makes a `UseLMR` checkbox appear here with no change to this
 * component, which is the entire point (see the design-system
 * milestone's UCI-option-discovery plan). Shared between `GameSetup`
 * (one Bee participant) and `ExperimentSetup` (two Bee variants to
 * compare) -- both need the identical generic rendering, just against
 * different `values`/`onChange`.
 *
 * Also seeds `values` with each discovered option's own reported
 * default the first time they're seen (via `onChange`, once the fetch
 * resolves) -- callers start with an empty `options` map rather than
 * guessing at option names/defaults themselves.
 */
export function EngineOptionsFields({
  engineName,
  values,
  onChange,
}: {
  engineName: string;
  values: EngineOptions;
  onChange: (values: EngineOptions) => void;
}) {
  const [options, setOptions] = useState<EngineOption[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setOptions(null);
    setError(null);
    getEngineOptions(engineName).then(
      (discovered) => {
        if (cancelled) return;
        setOptions(discovered);
        const seeded = { ...values };
        let changed = false;
        for (const option of discovered) {
          if (!(option.name in seeded)) {
            seeded[option.name] = option.default;
            changed = true;
          }
        }
        if (changed) onChange(seeded);
      },
      () => {
        if (!cancelled) setError(`Couldn't load ${engineName}'s options.`);
      },
    );
    return () => {
      cancelled = true;
    };
    // Deliberately only re-runs when the engine itself changes, not on
    // every `values`/`onChange` change -- this fetches the engine's
    // *schema* once per engine, not on every keystroke in the form it
    // renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineName]);

  if (error) {
    return <p className="m-0 text-xs text-danger">{error}</p>;
  }
  if (!options || options.length === 0) {
    return null;
  }

  return (
    <details open className="w-full rounded-md border border-border p-2.5 text-left">
      <summary className="cursor-pointer font-medium text-text">Advanced settings</summary>
      <div className="mt-3 grid gap-3">
        {options.map((option) => (
          <EngineOptionField
            key={option.name}
            option={option}
            value={values[option.name]}
            onChange={(value) => onChange({ ...values, [option.name]: value })}
          />
        ))}
      </div>
    </details>
  );
}

function EngineOptionField({
  option,
  value,
  onChange,
}: {
  option: EngineOption;
  value: EngineOptionValue | undefined;
  onChange: (value: EngineOptionValue) => void;
}) {
  if (option.type === "check") {
    return (
      <Checkbox
        label={option.name}
        checked={typeof value === "boolean" ? value : option.default}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  }

  return (
    <Field label={option.name}>
      {option.type === "combo" ? (
        <Select value={String(value ?? option.default)} onChange={(e) => onChange(e.target.value)}>
          {option.values.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </Select>
      ) : option.type === "spin" ? (
        <NumberInput
          value={typeof value === "number" ? value : option.default}
          min={option.min}
          max={option.max}
          step={1}
          onChange={onChange}
        />
      ) : (
        <Input
          type="text"
          value={typeof value === "string" ? value : option.default}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </Field>
  );
}
