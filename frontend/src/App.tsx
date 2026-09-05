import { useState } from "react";
import { GameSetup } from "./GameSetup";
import { Game } from "./Game";
import type { Participant } from "./participant";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

type Screen =
  | { phase: "setup" }
  | { phase: "playing"; white: Participant; black: Participant; gameSeq: number };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ phase: "setup" });

  return (
    <main
      style={{
        display: "grid",
        // An explicit 1fr column (rather than relying on the default
        // auto-sized implicit track) makes every child's available
        // width equal to <main>'s own width, not the width of the
        // widest child -- otherwise the grid track itself grows and
        // shrinks with content, and everything centered inside it
        // reflows along with it.
        gridTemplateColumns: "1fr",
        justifyItems: "center",
        alignItems: "center",
        gap: 8,
        padding: 24,
        textAlign: "center",
      }}
    >
      <h1>Bee Chess</h1>
      {screen.phase === "setup" ? (
        <GameSetup
          onStart={(white, black) =>
            setScreen({ phase: "playing", white, black, gameSeq: Date.now() })
          }
        />
      ) : (
        <Game
          key={screen.gameSeq}
          white={screen.white}
          black={screen.black}
          onBackToSetup={() => setScreen({ phase: "setup" })}
        />
      )}
    </main>
  );
}
