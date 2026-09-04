import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UciClient, type UciLogLine } from "./engine";

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
