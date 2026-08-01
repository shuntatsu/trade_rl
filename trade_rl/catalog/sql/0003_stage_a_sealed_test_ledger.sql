CREATE TABLE IF NOT EXISTS catalog_stage_a_sealed_test_batches (
    experiment_plan_digest CHAR(64) PRIMARY KEY
        CHECK (experiment_plan_digest ~ '^[0-9a-f]{64}$'),
    batch_digest CHAR(64) NOT NULL UNIQUE
        CHECK (batch_digest ~ '^[0-9a-f]{64}$'),
    schema_version TEXT NOT NULL
        CHECK (length(schema_version) > 0),
    evaluation_dataset_manifest_digest CHAR(64) NOT NULL
        CHECK (evaluation_dataset_manifest_digest ~ '^[0-9a-f]{64}$'),
    evaluation_identity CHAR(64) NOT NULL
        CHECK (evaluation_identity ~ '^[0-9a-f]{64}$'),
    selected_configuration TEXT NOT NULL
        CHECK (length(selected_configuration) > 0),
    selected_policy_digest CHAR(64) NOT NULL
        CHECK (selected_policy_digest ~ '^[0-9a-f]{64}$'),
    cell_count INTEGER NOT NULL CHECK (cell_count > 0),
    authorized_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_plan_digest, batch_digest)
);

CREATE TABLE IF NOT EXISTS catalog_stage_a_sealed_test_cells (
    experiment_plan_digest CHAR(64) NOT NULL
        CHECK (experiment_plan_digest ~ '^[0-9a-f]{64}$'),
    batch_digest CHAR(64) NOT NULL
        CHECK (batch_digest ~ '^[0-9a-f]{64}$'),
    cell_digest CHAR(64) NOT NULL UNIQUE
        CHECK (cell_digest ~ '^[0-9a-f]{64}$'),
    schema_version TEXT NOT NULL
        CHECK (length(schema_version) > 0),
    evaluation_dataset_manifest_digest CHAR(64) NOT NULL
        CHECK (evaluation_dataset_manifest_digest ~ '^[0-9a-f]{64}$'),
    triplet_id CHAR(64) NOT NULL
        CHECK (triplet_id ~ '^[0-9a-f]{64}$'),
    dataset_id CHAR(64) NOT NULL
        CHECK (dataset_id ~ '^[0-9a-f]{64}$'),
    fold_index INTEGER NOT NULL CHECK (fold_index >= 0),
    test_start INTEGER NOT NULL CHECK (test_start >= 0),
    test_stop INTEGER NOT NULL CHECK (test_stop > test_start),
    generic_access_digest CHAR(64) NOT NULL UNIQUE
        CHECK (generic_access_digest ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (experiment_plan_digest, triplet_id, fold_index),
    FOREIGN KEY (experiment_plan_digest, batch_digest)
        REFERENCES catalog_stage_a_sealed_test_batches (
            experiment_plan_digest, batch_digest
        )
        ON DELETE RESTRICT,
    FOREIGN KEY (experiment_plan_digest, dataset_id, fold_index)
        REFERENCES catalog_sealed_test_access (
            experiment_plan_digest, dataset_id, fold_index
        )
        ON DELETE RESTRICT,
    FOREIGN KEY (generic_access_digest)
        REFERENCES catalog_sealed_test_access (access_digest)
        ON DELETE RESTRICT
);
