CREATE TABLE IF NOT EXISTS catalog_sealed_test_access (
    experiment_plan_digest CHAR(64) NOT NULL
        CHECK (experiment_plan_digest ~ '^[0-9a-f]{64}$'),
    dataset_id CHAR(64) NOT NULL
        CHECK (dataset_id ~ '^[0-9a-f]{64}$'),
    fold_index INTEGER NOT NULL CHECK (fold_index >= 0),
    test_start INTEGER NOT NULL CHECK (test_start >= 0),
    test_stop INTEGER NOT NULL CHECK (test_stop > test_start),
    selected_configuration TEXT NOT NULL
        CHECK (length(selected_configuration) > 0),
    selected_policy_digest CHAR(64) NULL
        CHECK (
            selected_policy_digest IS NULL
            OR selected_policy_digest ~ '^[0-9a-f]{64}$'
        ),
    access_digest CHAR(64) NOT NULL UNIQUE
        CHECK (access_digest ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (experiment_plan_digest, dataset_id, fold_index)
);
