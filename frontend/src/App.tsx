import { useState } from "react";
import { SpectateGame } from "./SpectateGame";
import { PlayVsMamba } from "./PlayVsMamba";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

type Mode = "play" | "spectate";

export default function App() {
  const [mode, setMode] = useState<Mode>("play");

  return (
    <main
      style={{
        display: "grid",
        // An explicit 1fr column (rather than relying on the default
        // auto-sized implicit track) makes every child's available
        // width equal to <main>'s own width, not the width of the
        // widest child -- otherwise the grid track itself grows and
        // shrinks with content, and everything centered inside it
        // (including the log row below) reflows along with it.
        gridTemplateColumns: "1fr",
        justifyItems: "center",
        alignItems: "center",
        gap: 8,
        padding: 24,
        textAlign: "center",
      }}
    >
      <nav style={{ display: "flex", gap: 8 }}>
        <button type="button" disabled={mode === "play"} onClick={() => setMode("play")}>
          Play vs Bee-Mamba
        </button>
        <button type="button" disabled={mode === "spectate"} onClick={() => setMode("spectate")}>
          Spectate: Stockfish vs Bee
        </button>
      </nav>
      {mode === "play" ? <PlayVsMamba /> : <SpectateGame />}
    </main>
  );
}
