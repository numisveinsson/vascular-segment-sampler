-- SQLite schema for the vascular dataset catalog.
-- Stores metadata + on-disk paths only; image/mesh blobs stay on disk.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    root_path   TEXT NOT NULL,
    img_ext     TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('vmr', 'other')),
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    id              INTEGER PRIMARY KEY,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    case_id         TEXT NOT NULL,
    anatomy         TEXT,
    modality        TEXT,
    disease         TEXT,
    sex             TEXT,
    age             TEXT,
    species         TEXT,
    has_image       INTEGER NOT NULL DEFAULT 0,
    has_seg         INTEGER NOT NULL DEFAULT 0,
    has_centerline  INTEGER NOT NULL DEFAULT 0,
    has_surface     INTEGER NOT NULL DEFAULT 0,
    image_path      TEXT,
    seg_path        TEXT,
    centerline_path TEXT,
    surface_path    TEXT,
    source_meta     TEXT,  -- JSON blob for dataset-specific extras
    UNIQUE (dataset_id, case_id)
);

CREATE TABLE IF NOT EXISTS splits (
    id          INTEGER PRIMARY KEY,
    split_name  TEXT NOT NULL,
    dataset_id  INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    case_id     TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('train', 'test')),
    UNIQUE (split_name, dataset_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_cases_anatomy    ON cases(anatomy);
CREATE INDEX IF NOT EXISTS idx_cases_modality   ON cases(modality);
CREATE INDEX IF NOT EXISTS idx_cases_dataset_id ON cases(dataset_id);
CREATE INDEX IF NOT EXISTS idx_splits_name      ON splits(split_name);
