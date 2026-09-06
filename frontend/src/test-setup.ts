import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// React Testing Library normally hooks this into Jest's global
// afterEach; vitest doesn't wire that up automatically, so each
// render would otherwise leak into the next test's DOM.
afterEach(() => {
  cleanup();
});
