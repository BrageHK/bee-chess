/**
 * UCI clients, one WebSocket per engine, talking to bridge/server.py.
 *
 * Per ADR 0001 the frontend never spawns an engine process itself: the
 * bridge owns the processes and this module only speaks UCI text over a
 * socket.
 */

/** One line of raw UCI traffic, for the log panel. */
export interface UciLogLine {
  direction: "sent" | "received";
  text: string;
  timestamp: number;
}

export class UciClient {
  private readonly url: string;
  private ws: WebSocket | null = null;
  private ready: Promise<void> | null = null;
  private readonly listeners = new Set<(line: string) => void>();
  // Separate from `listeners`: those are one-shot protocol subscribers
  // (bestMove/setOption waiting for a specific reply), these are
  // permanent log consumers that want every line, sent and received,
  // unconditionally -- see `onLog`.
  private readonly logListeners = new Set<(line: UciLogLine) => void>();

  /** Name the engine reports via `id name`, until then the given label. */
  name: string;

  constructor(url: string, label: string) {
    this.url = url;
    this.name = label;
  }

  /**
   * Subscribes to every line of raw UCI traffic this client sends or
   * receives, in order, for as long as the client exists. Returns an
   * unsubscribe function.
   */
  onLog(listener: (line: UciLogLine) => void): () => void {
    this.logListeners.add(listener);
    return () => this.logListeners.delete(listener);
  }

  private logSent(text: string) {
    const line: UciLogLine = { direction: "sent", text, timestamp: Date.now() };
    for (const listener of this.logListeners) listener(line);
  }

  private logReceived(text: string) {
    const line: UciLogLine = { direction: "received", text, timestamp: Date.now() };
    for (const listener of this.logListeners) listener(line);
  }

  /** Connects and runs the `uci` / `isready` handshake. */
  init(): Promise<void> {
    if (this.ready) return this.ready;

    const ws = new WebSocket(this.url);
    this.ws = ws;

    this.ready = new Promise<void>((resolve, reject) => {
      const fail = () =>
        reject(new Error(`no bridge on ${this.url} — is bridge/server.py running?`));
      ws.onerror = fail;
      ws.onclose = fail;
      ws.onopen = () => this.send("uci");
      // One `onmessage` for the life of the socket: `bestMove` and
      // `setOption` subscribe through `listeners` instead of replacing
      // this handler, so nothing clobbers the handshake or a concurrent
      // search.
      ws.onmessage = (e: MessageEvent<string>) => {
        for (const raw of e.data.split("\n")) {
          const line = raw.trim();
          if (!line) continue;
          this.logReceived(line);
          if (line.startsWith("id name ")) this.name = line.slice(8);
          if (line === "uciok") this.send("isready");
          if (line === "readyok") resolve();
          for (const listener of this.listeners) listener(line);
        }
      };
    });

    return this.ready;
  }

  /**
   * Sends `setoption name <name> value <value>` and waits for the
   * engine to confirm it's still ready, so a caller can rely on the
   * option being applied before the next `go`. Safe to call again
   * before a new game to change settings (e.g. Stockfish's Elo) without
   * reconnecting.
   */
  async setOption(name: string, value: string | number | boolean): Promise<void> {
    await this.init();

    return new Promise<void>((resolve) => {
      const listener = (line: string) => {
        if (line !== "readyok") return;
        this.listeners.delete(listener);
        resolve();
      };
      this.listeners.add(listener);
      this.send(`setoption name ${name} value ${value}`);
      this.send("isready");
    });
  }

  /**
   * Sends `debug on`/`debug off` and waits for the engine to confirm
   * it's still ready, so a caller can rely on it being applied before
   * the next `go`. Safe to call again before a new game to change the
   * setting without reconnecting. This only affects the engine's own
   * diagnostic output (see #42/#44 on the Bee side); it has no bearing
   * on whether raw traffic appears in the log panel, which always
   * shows everything regardless.
   */
  async setDebug(on: boolean): Promise<void> {
    await this.init();

    return new Promise<void>((resolve) => {
      const listener = (line: string) => {
        if (line !== "readyok") return;
        this.listeners.delete(listener);
        resolve();
      };
      this.listeners.add(listener);
      this.send(on ? "debug on" : "debug off");
      this.send("isready");
    });
  }

  /**
   * Asks for a move in the position reached by `moves` from the start
   * position, thinking for exactly `moveTimeMs` (a fixed move-time
   * budget, not a chess clock -- see GameConfig).
   *
   * The move list is sent rather than a bare FEN so the engine can see
   * repetitions. Rejects if the engine stays silent well past its
   * budget, so a stuck engine can't hang the game loop forever.
   */
  async bestMove(moves: readonly string[], moveTimeMs: number): Promise<string> {
    await this.init();

    const budget = moveTimeMs + 5000;

    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(`${this.name} did not answer 'go' within ${Math.round(budget)}ms`));
      }, budget);

      const listener = (line: string) => {
        if (!line.startsWith("bestmove")) return;
        const uci = line.split(/\s+/)[1];
        cleanup();
        if (!uci || uci === "(none)") reject(new Error(`${this.name} has no move`));
        else resolve(uci);
      };

      const cleanup = () => {
        clearTimeout(timer);
        this.listeners.delete(listener);
      };

      this.listeners.add(listener);
      this.send(moves.length ? `position startpos moves ${moves.join(" ")}` : "position startpos");
      this.send(`go movetime ${Math.max(1, Math.round(moveTimeMs))}`);
    });
  }

  private send(cmd: string) {
    this.logSent(cmd);
    this.ws?.send(cmd);
  }
}

export const whiteEngine = new UciClient("ws://localhost:8765", "stockfish");
export const blackEngine = new UciClient("ws://localhost:8766", "bee");
export const mambaEngine = new UciClient("ws://localhost:8767", "bee-mamba");
