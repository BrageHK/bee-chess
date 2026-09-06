import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import * as labClient from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    checkLabAvailable: vi.fn().mockResolvedValue(true),
    getEngineOptions: vi.fn().mockResolvedValue([]),
    listGames: vi.fn().mockResolvedValue([]),
    listExperiments: vi.fn().mockResolvedValue([]),
    // Never resolve in these tests -- they only assert on history
    // state and the "loading"/"connecting" screens that produces, not
    // on real game/experiment data.
    getExperiment: vi.fn().mockReturnValue(new Promise(() => {})),
    getGame: vi.fn().mockReturnValue(new Promise(() => {})),
    subscribeToGameEvents: vi.fn().mockReturnValue(() => {}),
  };
});

/** Each test starts at a clean URL. `pushState`/`replaceState` are
 * spied on rather than asserted via `window.history.length`: jsdom's
 * `history.length` doesn't reset between tests sharing one `window`
 * the way a real browser's per-tab history would, so it isn't a
 * reliable signal here -- spying on the two methods App actually
 * calls is both more direct and immune to that. */
let pushSpy: ReturnType<typeof vi.spyOn>;
let replaceSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  window.history.replaceState(null, "", "/");
  vi.mocked(labClient.listGames).mockResolvedValue([]);
  vi.mocked(labClient.listExperiments).mockResolvedValue([]);
  pushSpy = vi.spyOn(window.history, "pushState");
  replaceSpy = vi.spyOn(window.history, "replaceState");
});

afterEach(() => {
  window.history.replaceState(null, "", "/");
  vi.restoreAllMocks();
});

describe("App navigation history", () => {
  it("starts on the dashboard with no URL params", async () => {
    render(<App />);
    expect(await screen.findByText(/nothing running/i)).toBeInTheDocument();
  });

  it("mounting the app on a resumable URL replaces, not pushes, its own history entry", async () => {
    window.history.replaceState(null, "", "/?game=game-123");
    replaceSpy.mockClear(); // clear the call from the line above, before mount

    render(<App />);
    await screen.findByText(/connecting to bee lab/i);

    expect(pushSpy).not.toHaveBeenCalled();
    expect(replaceSpy).toHaveBeenCalledTimes(1);
  });

  it("going dashboard -> new game pushes a history entry with the right URL", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/nothing running/i);

    await user.click(screen.getByRole("button", { name: /new game/i }));

    expect(screen.getByText(/start game/i)).toBeInTheDocument();
    expect(pushSpy).toHaveBeenCalledTimes(1);
    expect(pushSpy).toHaveBeenCalledWith(expect.anything(), "", expect.stringContaining("?new=game"));
  });

  it("browser back from game setup returns to the dashboard", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/nothing running/i);

    await user.click(screen.getByRole("button", { name: /new game/i }));
    expect(screen.getByText(/start game/i)).toBeInTheDocument();

    window.history.back();
    // jsdom fires popstate asynchronously after history.back().
    expect(await screen.findByText(/nothing running/i)).toBeInTheDocument();
  });

  it("a popstate-driven screen change does not push (or re-push) a history entry", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/nothing running/i);

    await user.click(screen.getByRole("button", { name: /new game/i }));
    pushSpy.mockClear();
    replaceSpy.mockClear();

    window.history.back();
    await screen.findByText(/nothing running/i);

    expect(pushSpy).not.toHaveBeenCalled();
  });

  it("app-driven navigation to two different locations in a row each pushes its own entry", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/nothing running/i);

    await user.click(screen.getByRole("button", { name: /new experiment/i }));
    expect(screen.getByRole("button", { name: /run experiment/i })).toBeInTheDocument();
    expect(pushSpy).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /^dashboard$/i }));
    await screen.findByText(/nothing running/i);
    expect(pushSpy).toHaveBeenCalledTimes(2);
  });

  it("clicking the Bee Chess title returns to the dashboard from anywhere", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/nothing running/i);

    await user.click(screen.getByRole("button", { name: /new experiment/i }));
    expect(screen.getByRole("button", { name: /run experiment/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /bee chess/i }));
    expect(await screen.findByText(/nothing running/i)).toBeInTheDocument();
  });

  it("a game created by an experiment offers a way back to it, and clicking it navigates there", async () => {
    vi.mocked(labClient.getGame).mockResolvedValue({
      id: "game-123",
      fen: "start",
      moves: [],
      uci_log: [{ color: "white", direction: "received", line: "historical line" }],
      status: "running",
      white: { kind: "engine", name: "Baseline", debug: false },
      black: { kind: "engine", name: "Candidate", debug: false },
      experiment_id: "exp-42",
    });
    const user = userEvent.setup();
    window.history.replaceState(null, "", "/?game=game-123");

    render(<App />);

    const backLink = await screen.findByRole("button", { name: /back to experiment/i });
    expect(await screen.findByText(/historical line/)).toHaveTextContent("← historical line");
    await user.click(backLink);

    expect(await screen.findByText(/loading experiment/i)).toBeInTheDocument();
    expect(window.location.search).toBe("?experiment=exp-42");
  });

  it("an ordinary game (no experiment_id) does not offer a way back to an experiment", async () => {
    vi.mocked(labClient.getGame).mockResolvedValue({
      id: "game-456",
      fen: "start",
      moves: [],
      uci_log: [],
      status: "running",
      white: { kind: "human" },
      black: { kind: "human" },
      experiment_id: null,
    });
    window.history.replaceState(null, "", "/?game=game-456");

    render(<App />);
    await screen.findByText(/your move|thinking|connecting/i);

    expect(screen.queryByRole("button", { name: /back to experiment/i })).not.toBeInTheDocument();
  });
});
