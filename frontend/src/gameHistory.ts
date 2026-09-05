import type { Key } from "@lichess-org/chessground/types";

/** One position reached during a game: the ply that led to it (`san`,
 * absent for the starting position) and enough to render the board at
 * that point. */
export interface Ply {
  fen: string;
  lastMove?: Key[];
  san?: string;
}

export interface Nav {
  history: Ply[];
  /** Index into `history` currently shown on the board. Auto-tracks
   * the latest ply as moves come in, unless the user has stepped back
   * to look at an earlier one (see `pushPly`). */
  viewIndex: number;
}

export function startNav(startFen: string): Nav {
  return { history: [{ fen: startFen }], viewIndex: 0 };
}

/** Appends a ply, keeping `viewIndex` pinned to "latest" unless the
 * user had already stepped back to browse history -- in which case a
 * new move arriving shouldn't yank the board out from under them. */
export function pushPly(nav: Nav, ply: Ply): Nav {
  const wasFollowing = nav.viewIndex === nav.history.length - 1;
  const history = [...nav.history, ply];
  return { history, viewIndex: wasFollowing ? history.length - 1 : nav.viewIndex };
}
