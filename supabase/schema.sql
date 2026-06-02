-- Run in your Supabase SQL Editor
-- Stores training run metadata for every checkpoint save

CREATE TABLE IF NOT EXISTS training_runs (
  id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  epoch           INTEGER     NOT NULL,
  loss            FLOAT       NOT NULL,
  config          JSONB,
  checkpoint_path TEXT,
  model_version   TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Efficient queries: best run by loss, latest runs by time
CREATE INDEX IF NOT EXISTS idx_training_runs_loss    ON training_runs (loss ASC);
CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_version ON training_runs (model_version);

-- Row-level security: only service role can write
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_insert" ON training_runs
  FOR INSERT TO service_role USING (true);

CREATE POLICY "anon_select" ON training_runs
  FOR SELECT TO anon USING (true);
