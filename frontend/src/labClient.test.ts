import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  checkLabAvailable,
  createGame,
  getGame,
  LabError,
  postMove,
  subscribeToGameEvents,
  type GameEvent,
} from "./labClient";

function jsonResponse(body: unknown, ok = true, status = ok ? 200 : 400): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Bad Request",
    json: () => Promise.resolve(body),
  } as Response;
}

/** A minimal, human-vs-human `GameSnapshot` fixture -- these tests
 * exercise labClient's request/response plumbing, not participant
 * rendering, so `white`/`black` only need to be present and valid,
 * not varied per test. */
function humanSnapshot(overrides: Partial<{ id: string; fen: string; moves: string[]; status: "running" }> = {}) {
  return {
    id: "abc",
    fen: "start",
    moves: [] as string[],
    status: "running" as const,
    white: { kind: "human" as const },
    black: { kind: "human" as const },
    ...overrides,
  };
}

describe("createGame", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => vi.unstubAllGlobals());

  it("POSTs to /api/games with only the fields that were given", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(
      jsonResponse(humanSnapshot()),
    );

    await createGame({ white: "stockfish", moveTimeMs: 250 });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8080/api/games",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ white: "stockfish", move_time_ms: 250 }),
      }),
    );
  });

  it("sends an empty body when called with no request at all", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(
      jsonResponse(humanSnapshot()),
    );

    await createGame();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8080/api/games",
      expect.objectContaining({ body: "{}" }),
    );
  });

  it("returns the parsed snapshot on success", async () => {
    const snapshot = humanSnapshot();
    vi.mocked(fetch).mockResolvedValue(jsonResponse(snapshot));

    await expect(createGame()).resolves.toEqual(snapshot);
  });

  it("throws a LabError carrying the server's error message on failure", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ error: "unknown engine \"nope\"" }, false, 400));

    await expect(createGame({ white: "nope" })).rejects.toThrow(LabError);
    await expect(createGame({ white: "nope" })).rejects.toThrow(/unknown engine/);
  });
});

describe("getGame", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("GETs /api/games/:id", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(
      jsonResponse(humanSnapshot()),
    );

    await getGame("abc");

    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8080/api/games/abc");
  });

  it("rejects with LabError on a 404", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ error: "no such game" }, false, 404));

    await expect(getGame("missing")).rejects.toThrow(LabError);
  });
});

describe("postMove", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("POSTs the uci move to /api/games/:id/moves", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(
      jsonResponse(humanSnapshot({ fen: "after-e4", moves: ["e2e4"] })),
    );

    await postMove("abc", "e2e4");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8080/api/games/abc/moves",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ uci: "e2e4" }) }),
    );
  });

  it("rejects with LabError for an illegal move", async () => {
    vi.mocked(fetch).mockResolvedValue(jsonResponse({ error: "illegal move" }, false, 422));

    await expect(postMove("abc", "e2e5")).rejects.toThrow(LabError);
  });
});

describe("subscribeToGameEvents", () => {
  class FakeWebSocket {
    static instances: FakeWebSocket[] = [];
    onmessage: ((e: { data: string }) => void) | null = null;
    closed = false;
    readonly url: string;

    constructor(url: string) {
      this.url = url;
      FakeWebSocket.instances.push(this);
    }

    close() {
      this.closed = true;
    }

    /** Test helper: simulates the server sending an event frame. */
    receive(data: string) {
      this.onmessage?.({ data });
    }
  }

  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("connects to /ws/games/:id", () => {
    subscribeToGameEvents("abc", () => {});

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).toBe("ws://localhost:8080/ws/games/abc");
  });

  it("parses and forwards each event frame", () => {
    const events: GameEvent[] = [];
    subscribeToGameEvents("abc", (event) => events.push(event));

    const ws = FakeWebSocket.instances[0];
    const snapshot = humanSnapshot({ moves: ["e2e4"] });
    ws.receive(JSON.stringify({ type: "uci", color: "white", direction: "received", line: "uciok" }));
    ws.receive(JSON.stringify({ type: "updated", snapshot }));

    expect(events).toEqual([
      { type: "uci", color: "white", direction: "received", line: "uciok" },
      { type: "updated", snapshot },
    ]);
  });

  it("silently ignores a malformed frame instead of throwing", () => {
    const events: GameEvent[] = [];
    subscribeToGameEvents("abc", (event) => events.push(event));

    expect(() => FakeWebSocket.instances[0].receive("not json")).not.toThrow();
    expect(events).toEqual([]);
  });

  it("the unsubscribe function closes the socket", () => {
    const unsubscribe = subscribeToGameEvents("abc", () => {});

    unsubscribe();

    expect(FakeWebSocket.instances[0].closed).toBe(true);
  });
});

describe("checkLabAvailable", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("resolves true when Lab responds ok", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true } as Response);

    await expect(checkLabAvailable()).resolves.toBe(true);
  });

  it("resolves false when Lab responds with an error status", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: false } as Response);

    await expect(checkLabAvailable()).resolves.toBe(false);
  });

  it("resolves false when the fetch itself rejects (Lab unreachable)", async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(checkLabAvailable()).resolves.toBe(false);
  });

  it("resolves false if nothing happens before the timeout", async () => {
    vi.useFakeTimers();
    // A fetch that never settles -- simulates a request that just
    // hangs (e.g. a firewall silently dropping packets) rather than
    // failing immediately the way a refused connection would.
    vi.mocked(fetch).mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          (init?.signal as AbortSignal | undefined)?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );

    const resultPromise = checkLabAvailable(100);
    await vi.advanceTimersByTimeAsync(100);
    await expect(resultPromise).resolves.toBe(false);

    vi.useRealTimers();
  });
});
