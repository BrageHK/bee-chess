/**
 * UCI clients, one WebSocket per engine, talking to bridge/server.py.
 *
 * Per ADR 0001 the frontend never spawns an engine process itself: the
 * bridge owns the processes and this module only speaks UCI text over a
 * socket.
 */

/** Remaining time and increment for both sides, in milliseconds. */
export interface Clock {
  wtime: number;
  btime: number;
  winc: number;
  binc: number;
}

class UciClient {
  private readonly url: string;
  private readonly options: readonly string[];
  private ws: WebSocket | null = null;
  private ready: Promise<void> | null = null;
  private readonly listeners = new Set<(line: string) => void>();

  /** Name the engine reports via `id name`, until then the given label. */
  name: string;

  constructor(url: string, label: string, options: readonly string[] = []) {
    this.url = url;
    this.name = label;
    this.options = options;
  }

  /** Connects and runs the `uci` / `setoption` / `isready` handshake. */
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
      // One `onmessage` for the life of the socket: `bestMove` subscribes
      // through `listeners` instead of replacing this handler, so nothing
      // clobbers the handshake or a concurrent search.
      ws.onmessage = (e: MessageEvent<string>) => {
        for (const raw of e.data.split("\n")) {
          const line = raw.trim();
          if (!line) continue;
          if (line.startsWith("id name ")) this.name = line.slice(8);
          if (line === "uciok") {
            for (const option of this.options) this.send(option);
            this.send("isready");
          }
          if (line === "readyok") resolve();
          for (const listener of this.listeners) listener(line);
        }
      };
    });

    return this.ready;
  }

  /**
   * Asks for a move in the position reached by `moves` from the start
   * position, under the given clock.
   *
   * The move list is sent rather than a bare FEN so the engine can see
   * repetitions. Rejects if the engine stays silent past its own clock —
   * the Bee engine only implements the UCI handshake so far, so it never
   * answers `go`.
   */
  async bestMove(moves: readonly string[], clk: Clock, white: boolean): Promise<string> {
    await this.init();

    const budget = (white ? clk.wtime : clk.btime) + 5000;

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
      this.send(
        `go wtime ${ms(clk.wtime)} btime ${ms(clk.btime)} winc ${ms(clk.winc)} binc ${ms(clk.binc)}`,
      );
    });
  }

  private send(cmd: string) {
    this.ws?.send(cmd);
  }
}

const ms = (t: number) => Math.max(0, Math.round(t));

/**
 * Stockfish at full strength is not a fair fight, so it is capped here.
 * Raise `UCI_Elo` (1320-3190) or drop both lines to unleash it.
 */
const STOCKFISH_OPTIONS = [
  "setoption name UCI_LimitStrength value true",
  "setoption name UCI_Elo value 1600",
];

export const whiteEngine = new UciClient("ws://localhost:8765", "stockfish", STOCKFISH_OPTIONS);
export const blackEngine = new UciClient("ws://localhost:8766", "bee");
