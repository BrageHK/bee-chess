import { cloneElement, useId, type ReactElement, type ReactNode } from "react";
import type { InputProps } from "./Input";
import type { NumberInputProps } from "./NumberInput";
import type { SelectProps } from "./Select";

type FieldControl = ReactElement<InputProps | NumberInputProps | SelectProps>;

export function Field({
  label,
  description,
  error,
  children,
}: {
  label: ReactNode;
  description?: ReactNode;
  error?: ReactNode;
  children: FieldControl;
}) {
  const id = useId();
  const descriptionId = description ? `${id}-description` : undefined;
  const errorId = error ? `${id}-error` : undefined;

  return (
    <div className="grid gap-1.5">
      <label htmlFor={id} className="text-sm font-medium text-text">
        {label}
      </label>
      {description && (
        <p id={descriptionId} className="text-xs text-muted">
          {description}
        </p>
      )}
      {cloneElement(children, {
        id,
        invalid: Boolean(error) || children.props.invalid,
        "aria-describedby": [descriptionId, errorId].filter(Boolean).join(" ") || undefined,
      })}
      {error && (
        <p id={errorId} className="text-xs text-danger">
          {error}
        </p>
      )}
    </div>
  );
}
