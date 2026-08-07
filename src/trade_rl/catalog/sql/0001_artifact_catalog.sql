CREATE TABLE IF NOT EXISTS catalog_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS catalog_artifacts (
    artifact_digest CHAR(64) PRIMARY KEY CHECK (artifact_digest ~ '^[0-9a-f]{64}$'),
    artifact_kind TEXT NOT NULL CHECK (length(artifact_kind) > 0),
    schema_version TEXT NOT NULL CHECK (length(schema_version) > 0),
    dataset_id CHAR(64) NULL CHECK (dataset_id IS NULL OR dataset_id ~ '^[0-9a-f]{64}$'),
    cache_key_digest CHAR(64) NOT NULL CHECK (cache_key_digest ~ '^[0-9a-f]{64}$'),
    cache_key JSONB NOT NULL,
    metadata JSONB NOT NULL,
    location TEXT NOT NULL CHECK (length(location) > 0),
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN ('ready', 'failed', 'superseded')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (artifact_kind, cache_key_digest)
);

CREATE INDEX IF NOT EXISTS catalog_artifacts_kind_created_idx
    ON catalog_artifacts (artifact_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS catalog_artifacts_dataset_idx
    ON catalog_artifacts (dataset_id) WHERE dataset_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS catalog_artifacts_status_idx
    ON catalog_artifacts (status, created_at DESC);

CREATE TABLE IF NOT EXISTS catalog_artifact_dependencies (
    parent_digest CHAR(64) NOT NULL REFERENCES catalog_artifacts(artifact_digest) ON DELETE CASCADE,
    child_digest CHAR(64) NOT NULL REFERENCES catalog_artifacts(artifact_digest) ON DELETE CASCADE,
    dependency_role TEXT NOT NULL CHECK (length(dependency_role) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (parent_digest, child_digest, dependency_role),
    CHECK (parent_digest <> child_digest)
);

CREATE INDEX IF NOT EXISTS catalog_dependencies_child_idx
    ON catalog_artifact_dependencies (child_digest);
