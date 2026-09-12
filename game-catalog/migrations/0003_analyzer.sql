-- Legacy records keep NULL provenance; the offline analyzer supplies all fields.
ALTER TABLE analysis_runs ADD COLUMN configuration TEXT;
ALTER TABLE move_analysis ADD COLUMN mover_color TEXT;
ALTER TABLE move_analysis ADD COLUMN is_bee INTEGER;
ALTER TABLE move_analysis ADD COLUMN pv TEXT;
CREATE INDEX move_analysis_run_bee_loss
    ON move_analysis (analysis_run_id, is_bee, centipawn_loss);

-- Corrected imports must not retain analysis of a different game sequence.
CREATE TRIGGER invalidate_changed_game_analysis
AFTER UPDATE OF moves, variant, white, black, raw_pgn ON games
WHEN OLD.moves IS NOT NEW.moves OR OLD.variant IS NOT NEW.variant
  OR OLD.white IS NOT NEW.white OR OLD.black IS NOT NEW.black
  OR OLD.raw_pgn IS NOT NEW.raw_pgn
BEGIN
    DELETE FROM move_analysis WHERE game_id = NEW.id;
    DELETE FROM game_analysis WHERE game_id = NEW.id;
END;
