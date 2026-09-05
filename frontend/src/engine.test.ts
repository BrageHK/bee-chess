import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { checkBotAvailable, createBotClient, UciClient, type UciLogLine } from "./engine";

/**
 * A minimal fake WebSocket good enough to drive UciClient's handshake
 * and record what it sent, without a real bridge/engine process.
 */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readonly url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    // Real WebSocket fires `open` asynchronously; queue a microtask so
    // callers that set onopen synchronously after construction still
    // see it fire, matching real behavior closely enough for this
    // client's handshake logic.
    queueMicrotask(() => this.onopen?.());
  }

  send(data: string) {
    this.sent.push(data);
  }

  /** Test helper: simulates the engine sending a line back. */
  receive(line: string) {
    this.onmessage?.({ data: line });
  }
}

describe("UciClient logging", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("logs sent commands with direction 'sent'", async () => {
    const client = new UciClient("ws://test", "engine");
    const logs: UciLogLine[] = [];
    client.onLog((line) => logs.push(line));

    void client.init();
    await Promise.resolve(); // let the queued onopen fire

    expect(logs).toHaveLength(1);
    expect(logs[0]).toMatchObject({ direction: "sent", text: "uci" });
  });

  it("logs received lines with direction 'received'", async () => {
    const client = new UciClient("ws://test", "engine");
    const logs: UciLogLine[] = [];
    client.onLog((line) => logs.push(line));

    void client.init();
    await Promise.resolve();

    const ws = FakeWebSocket.instances[0];
    ws.receive("id name Bee");
    ws.receive("uciok");

    const received = logs.filter((l) => l.direction === "received");
    expect(received.map((l) => l.text)).toEqual(["id name Bee", "uciok"]);
  });

  it("logs every sent/received line in order for a full handshake", async () => {
    const client = new UciClient("ws://test", "engine");
    const logs: UciLogLine[] = [];
    client.onLog((line) => logs.push(line));

    const readyPromise = client.init();
    await Promise.resolve();

    const ws = FakeWebSocket.instances[0];
    ws.receive("id name Bee");
    ws.receive("uciok");
    ws.receive("readyok");
    await readyPromise;

    expect(logs.map((l) => `${l.direction === "sent" ? "→" : "←"} ${l.text}`)).toEqual([
      "→ uci",
      "← id name Bee",
      "← uciok",
      "→ isready",
      "← readyok",
    ]);
  });

  it("unsubscribe stops further log delivery", async () => {
    const client = new UciClient("ws://test", "engine");
    const logs: UciLogLine[] = [];
    const unsubscribe = client.onLog((line) => logs.push(line));

    void client.init();
    await Promise.resolve();
    unsubscribe();

    const ws = FakeWebSocket.instances[0];
    ws.receive("uciok");

    expect(logs).toHaveLength(1); // only the initial "→ uci"
    expect(logs[0].text).toBe("uci");
  });

  it("multiple subscribers each receive every line", async () => {
    const client = new UciClient("ws://test", "engine");
    const a: UciLogLine[] = [];
    const b: UciLogLine[] = [];
    client.onLog((line) => a.push(line));
    client.onLog((line) => b.push(line));

    void client.init();
    await Promise.resolve();

    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
  });
});

/** Completes a client's uci/isready handshake against a FakeWebSocket. */
async function completeHandshake(client: UciClient): Promise<FakeWebSocket> {
  const readyPromise = client.init();
  await Promise.resolve();
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  ws.receive("uciok");
  ws.receive("readyok");
  await readyPromise;
  return ws;
}

/** A fake WebSocket whose `open` never fires on its own -- for
 * distinguishing "the bridge itself isn't running" (connection never
 * opens) from "the bridge is up but the engine process behind it died
 * mid-handshake" (connection opens, then closes). */
class ManualWebSocket {
  static instances: ManualWebSocket[] = [];
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  readonly url: string;

  constructor(url: string) {
    this.url = url;
    ManualWebSocket.instances.push(this);
  }

  send() {}
}

describe("UciClient.init error messages", () => {
  beforeEach(() => {
    ManualWebSocket.instances = [];
    vi.stubGlobal("WebSocket", ManualWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("says the bridge isn't running when the socket never opens", async () => {
    const client = new UciClient("ws://test", "engine");
    const initPromise = client.init();

    const ws = ManualWebSocket.instances[0];
    ws.onclose?.();

    await expect(initPromise).rejects.toThrow(/is bridge\/server\.py running/);
  });

  it("says the handshake was interrupted when the socket opens then closes", async () => {
    const client = new UciClient("ws://test", "engine");
    const initPromise = client.init();

    const ws = ManualWebSocket.instances[0];
    ws.onopen?.();
    ws.onclose?.();

    await expect(initPromise).rejects.toThrow(/disconnected before finishing the UCI handshake/);
  });
});

describe("UciClient.setDebug", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends 'debug on' then 'isready', and resolves once readyok arrives", async () => {
    const client = new UciClient("ws://test", "engine");
    const ws = await completeHandshake(client);

    const setDebugPromise = client.setDebug(true);
    await Promise.resolve(); // let the already-resolved init() await settle
    expect(ws.sent.slice(-2)).toEqual(["debug on", "isready"]);

    ws.receive("readyok");
    await setDebugPromise; // resolves once readyok arrives
  });

  it("sends 'debug off'", async () => {
    const client = new UciClient("ws://test", "engine");
    const ws = await completeHandshake(client);

    const setDebugPromise = client.setDebug(false);
    await Promise.resolve();
    expect(ws.sent.slice(-2)).toEqual(["debug off", "isready"]);

    ws.receive("readyok");
    await setDebugPromise;
  });

  it("only resolves once readyok arrives, not merely after sending", async () => {
    const client = new UciClient("ws://test", "engine");
    const ws = await completeHandshake(client);

    let resolved = false;
    const setDebugPromise = client.setDebug(true).then(() => {
      resolved = true;
    });

    await Promise.resolve();
    expect(resolved).toBe(false); // no readyok yet

    ws.receive("readyok");
    await setDebugPromise;
    expect(resolved).toBe(true);
  });
});

describe("createBotClient", () => {
  it("creates a client pointed at the right URL for each kind", () => {
    expect(createBotClient("stockfish")).toMatchObject({ name: "stockfish" });
    expect(createBotClient("bee")).toMatchObject({ name: "bee" });
    expect(createBotClient("bee-mamba")).toMatchObject({ name: "bee-mamba" });
  });

  it("returns a fresh instance every call, not a shared singleton", () => {
    const a = createBotClient("bee");
    const b = createBotClient("bee");
    expect(a).not.toBe(b);
  });
});

/** A fake WebSocket that can simulate either a successful open or a
 * failed/errored connection, for checkBotAvailable's probe. */
class ProbeWebSocket {
  static instances: ProbeWebSocket[] = [];
  static behavior: "open" | "error" = "open";
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  readonly url: string;

  constructor(url: string) {
    this.url = url;
    ProbeWebSocket.instances.push(this);
    queueMicrotask(() => {
      if (ProbeWebSocket.behavior === "open") this.onopen?.();
      else this.onerror?.();
    });
  }

  close() {
    this.closed = true;
  }
}

describe("checkBotAvailable", () => {
  beforeEach(() => {
    ProbeWebSocket.instances = [];
    ProbeWebSocket.behavior = "open";
    vi.stubGlobal("WebSocket", ProbeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves true when the socket opens", async () => {
    ProbeWebSocket.behavior = "open";
    await expect(checkBotAvailable("bee")).resolves.toBe(true);
  });

  it("resolves false when the socket errors", async () => {
    ProbeWebSocket.behavior = "error";
    await expect(checkBotAvailable("bee-mamba")).resolves.toBe(false);
  });

  it("closes the probe connection either way", async () => {
    ProbeWebSocket.behavior = "open";
    await checkBotAvailable("bee");
    expect(ProbeWebSocket.instances[0].closed).toBe(true);
  });

  it("resolves false if nothing happens before the timeout", async () => {
    vi.useFakeTimers();
    // Never fire onopen/onerror -- simulates a connection attempt that
    // just hangs (e.g. nothing listening, no immediate refusal).
    class HangingWebSocket {
      onopen: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close() {}
    }
    vi.stubGlobal("WebSocket", HangingWebSocket);

    const resultPromise = checkBotAvailable("stockfish", 100);
    await vi.advanceTimersByTimeAsync(100);
    await expect(resultPromise).resolves.toBe(false);

    vi.useRealTimers();
  });
});
