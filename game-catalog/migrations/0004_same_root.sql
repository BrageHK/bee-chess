-- Analysis method 3 keeps the played-move score at the SAME root as the best
-- score. Older runs retain NULLs here and their original score-delta semantics.
ALTER TABLE move_analysis ADD COLUMN eval_played_cp INTEGER;
ALTER TABLE move_analysis ADD COLUMN mate_played INTEGER;
ALTER TABLE move_analysis ADD COLUMN played_pv TEXT;
