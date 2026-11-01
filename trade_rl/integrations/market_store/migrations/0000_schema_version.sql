-- Migration 0000: the schema_version ledger.
--
-- The migration runner records one row per applied migration file together with
-- the SHA-256 of the file bytes. A recorded checksum that no longer matches the
-- shipped file is refused (an applied migration must never be edited; add a new
-- numbered file instead).

CREATE TABLE schema_version (
    version integer PRIMARY KEY CHECK (version >= 0),
    name text NOT NULL CHECK (name <> ''),
    checksum_sha256 text NOT NULL CHECK (checksum_sha256 ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT now()
);
