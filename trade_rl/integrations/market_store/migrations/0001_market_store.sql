-- Migration 0001: reusable public market-data store (Issue #810).
--
-- Contract:
--   * exact NUMERIC for every price, quantity and rate (no float for money);
--   * klines / funding_rates / sources are append-only: a changed value for an
--     existing natural key can never be written (UPDATE is rejected, a duplicate
--     insert is a primary-key violation); the application additionally compares
--     re-ingested rows and records any difference in ingestion_conflicts;
--   * the FS1 sealed window (>= 2026-11-01T00:00Z) cannot be stored: the
--     market_embargo row plus the triggers below reject such rows at the database
--     layer, independent of the Python ingester. The only release path is an
--     explicit, permanent, audited update of market_embargo.released_at together
--     with an Issue #810 comment URL.
--   The sealed_from literal below must equal
--   trade_rl.integrations.market_store.embargo.SEALED_WINDOW_START (pinned by a
--   test against the FS1 registry scoring_start).

-- ---------------------------------------------------------------------------
-- Embargo
-- ---------------------------------------------------------------------------

CREATE TABLE market_embargo (
    id smallint PRIMARY KEY CHECK (id = 1),
    sealed_from timestamptz NOT NULL,
    reason text NOT NULL,
    released_at timestamptz,
    release_authorization_url text,
    CONSTRAINT market_embargo_release_consistent
        CHECK ((released_at IS NULL) = (release_authorization_url IS NULL)),
    CONSTRAINT market_embargo_release_url_shape
        CHECK (
            release_authorization_url IS NULL
            OR release_authorization_url ~
                '^https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-[0-9]+$'
        )
);

INSERT INTO market_embargo (id, sealed_from, reason)
VALUES (
    1,
    TIMESTAMPTZ '2026-11-01 00:00:00+00',
    'FS1 forward study sealed unused window: scored once after 2027-05-02, '
    'no data at or after sealed_from may be stored or viewed before then'
);

CREATE FUNCTION market_embargo_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'market_embargo row cannot be deleted'
            USING ERRCODE = 'restrict_violation';
    END IF;
    IF NEW.id IS DISTINCT FROM OLD.id
        OR NEW.sealed_from IS DISTINCT FROM OLD.sealed_from
        OR NEW.reason IS DISTINCT FROM OLD.reason THEN
        RAISE EXCEPTION 'market_embargo boundary and reason are immutable'
            USING ERRCODE = 'restrict_violation';
    END IF;
    IF OLD.released_at IS NOT NULL THEN
        RAISE EXCEPTION 'market_embargo release is permanent and cannot be changed'
            USING ERRCODE = 'restrict_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER market_embargo_guard_row
    BEFORE UPDATE OR DELETE ON market_embargo
    FOR EACH ROW EXECUTE FUNCTION market_embargo_guard();

CREATE FUNCTION market_embargo_no_truncate() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'market_embargo cannot be truncated'
        USING ERRCODE = 'restrict_violation';
END
$$;

CREATE TRIGGER market_embargo_guard_truncate
    BEFORE TRUNCATE ON market_embargo
    FOR EACH STATEMENT EXECUTE FUNCTION market_embargo_no_truncate();

CREATE FUNCTION market_embargo_sealed_from() RETURNS timestamptz
LANGUAGE plpgsql STABLE AS $$
DECLARE
    boundary timestamptz;
    released timestamptz;
BEGIN
    SELECT sealed_from, released_at INTO boundary, released
    FROM market_embargo WHERE id = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'market_embargo row is missing'
            USING ERRCODE = 'check_violation';
    END IF;
    IF released IS NOT NULL THEN
        RETURN 'infinity'::timestamptz;
    END IF;
    RETURN boundary;
END
$$;

CREATE FUNCTION forbid_row_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only (% rejected)', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'restrict_violation';
END
$$;

-- ---------------------------------------------------------------------------
-- Instruments and exchange rules
-- ---------------------------------------------------------------------------

CREATE TABLE instruments (
    instrument_id bigserial PRIMARY KEY,
    exchange text NOT NULL CHECK (exchange = 'binance'),
    market text NOT NULL CHECK (market IN ('spot', 'usds_m')),
    symbol text NOT NULL CHECK (symbol ~ '^[A-Z0-9]{3,32}$'),
    contract_type text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT instruments_natural_key UNIQUE (exchange, market, symbol),
    CONSTRAINT instruments_market_contract_type CHECK (
        (market = 'usds_m' AND contract_type IS NOT DISTINCT FROM 'PERPETUAL')
        OR (market = 'spot' AND contract_type IS NULL)
    )
);

CREATE TABLE ingestion_runs (
    run_id bigserial PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed')),
    code_commit text NOT NULL,
    command_line text NOT NULL,
    request jsonb NOT NULL,
    embargo_released boolean NOT NULL DEFAULT false,
    release_authorization_url text,
    archives_planned integer NOT NULL DEFAULT 0 CHECK (archives_planned >= 0),
    archives_ingested integer NOT NULL DEFAULT 0 CHECK (archives_ingested >= 0),
    archives_skipped integer NOT NULL DEFAULT 0 CHECK (archives_skipped >= 0),
    archives_missing integer NOT NULL DEFAULT 0 CHECK (archives_missing >= 0),
    rows_inserted bigint NOT NULL DEFAULT 0 CHECK (rows_inserted >= 0),
    rows_identical bigint NOT NULL DEFAULT 0 CHECK (rows_identical >= 0),
    conflicts integer NOT NULL DEFAULT 0 CHECK (conflicts >= 0),
    error text,
    CONSTRAINT ingestion_runs_finished_consistent
        CHECK ((status = 'running') = (finished_at IS NULL))
);

CREATE TABLE exchange_info_snapshots (
    snapshot_id bigserial PRIMARY KEY,
    exchange text NOT NULL CHECK (exchange = 'binance'),
    market text NOT NULL CHECK (market IN ('spot', 'usds_m')),
    retrieved_at timestamptz NOT NULL,
    source_uri text NOT NULL,
    raw_payload_sha256 text NOT NULL CHECK (raw_payload_sha256 ~ '^[0-9a-f]{64}$'),
    raw_payload bytea NOT NULL,
    run_id bigint REFERENCES ingestion_runs (run_id),
    CONSTRAINT exchange_info_snapshots_content UNIQUE (exchange, market, raw_payload_sha256)
);

CREATE TRIGGER exchange_info_snapshots_append_only
    BEFORE UPDATE OR DELETE ON exchange_info_snapshots
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

-- Rules are current-snapshot values valid from the retrieval time onwards; they
-- are NOT historical point-in-time rules. valid_from is the snapshot retrieval
-- time and the rule is superseded by the next snapshot of the same instrument.
CREATE TABLE instrument_rule_snapshots (
    rule_snapshot_id bigserial PRIMARY KEY,
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    snapshot_id bigint NOT NULL REFERENCES exchange_info_snapshots (snapshot_id),
    valid_from timestamptz NOT NULL,
    trading_status text NOT NULL,
    onboard_date timestamptz,
    tick_size numeric NOT NULL CHECK (tick_size >= 0),
    lot_size numeric NOT NULL CHECK (lot_size >= 0),
    min_notional numeric NOT NULL CHECK (min_notional >= 0),
    CONSTRAINT instrument_rule_snapshots_natural_key UNIQUE (instrument_id, snapshot_id)
);

CREATE INDEX instrument_rule_snapshots_validity
    ON instrument_rule_snapshots (instrument_id, valid_from);

CREATE TRIGGER instrument_rule_snapshots_append_only
    BEFORE UPDATE OR DELETE ON instrument_rule_snapshots
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

-- ---------------------------------------------------------------------------
-- Sources (one row per archive, with its checksum evidence)
-- ---------------------------------------------------------------------------

CREATE TABLE sources (
    source_id bigserial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    kind text NOT NULL CHECK (kind IN ('klines', 'funding_rate')),
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    bar_interval text,
    period_kind text NOT NULL CHECK (period_kind IN ('monthly', 'daily')),
    period_start timestamptz NOT NULL,
    period_end timestamptz NOT NULL,
    status text NOT NULL CHECK (status IN ('ingested', 'missing')),
    http_status integer NOT NULL CHECK (http_status BETWEEN 100 AND 599),
    sha256 text CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint CHECK (size_bytes > 0),
    checksum_url text NOT NULL,
    checksum_file_sha256 text CHECK (checksum_file_sha256 ~ '^[0-9a-f]{64}$'),
    checksum_line_sha256 text CHECK (checksum_line_sha256 ~ '^[0-9a-f]{64}$'),
    fetched_at timestamptz NOT NULL,
    row_count integer CHECK (row_count >= 0),
    run_id bigint NOT NULL REFERENCES ingestion_runs (run_id),
    CONSTRAINT sources_period_order CHECK (period_start < period_end),
    CONSTRAINT sources_kind_interval CHECK (
        (kind = 'klines' AND bar_interval IS NOT NULL)
        OR (kind = 'funding_rate' AND bar_interval IS NULL)
    ),
    CONSTRAINT sources_bar_interval_known CHECK (
        bar_interval IS NULL
        OR bar_interval IN ('15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d')
    ),
    CONSTRAINT sources_status_evidence CHECK (
        (status = 'ingested' AND http_status = 200 AND sha256 IS NOT NULL
            AND size_bytes IS NOT NULL AND checksum_file_sha256 IS NOT NULL
            AND checksum_line_sha256 IS NOT NULL AND checksum_line_sha256 = sha256
            AND row_count IS NOT NULL)
        OR (status = 'missing' AND http_status = 404 AND sha256 IS NULL
            AND size_bytes IS NULL AND row_count IS NULL)
    )
);

CREATE INDEX sources_instrument_period ON sources (instrument_id, kind, bar_interval, period_start);

CREATE FUNCTION sources_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'sources is append-only (DELETE rejected)'
            USING ERRCODE = 'restrict_violation';
    END IF;
    -- A 404 record may later be replaced by the published archive; an ingested
    -- source is immutable evidence.
    IF OLD.status <> 'missing' THEN
        RAISE EXCEPTION 'sources is append-only (UPDATE of an ingested source rejected)'
            USING ERRCODE = 'restrict_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER sources_guard_row
    BEFORE UPDATE OR DELETE ON sources
    FOR EACH ROW EXECUTE FUNCTION sources_guard();

CREATE FUNCTION sources_embargo() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.period_end > market_embargo_sealed_from() THEN
        RAISE EXCEPTION 'source period % .. % reaches the sealed window',
            NEW.period_start, NEW.period_end
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER sources_embargo_guard
    BEFORE INSERT OR UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION sources_embargo();

-- ---------------------------------------------------------------------------
-- Klines
-- ---------------------------------------------------------------------------

CREATE TABLE klines (
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    bar_interval text NOT NULL
        CHECK (bar_interval IN ('15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d')),
    open_time timestamptz NOT NULL,
    open numeric NOT NULL CHECK (open > 0),
    high numeric NOT NULL CHECK (high > 0),
    low numeric NOT NULL CHECK (low > 0),
    close numeric NOT NULL CHECK (close > 0),
    volume numeric NOT NULL CHECK (volume >= 0),
    close_time timestamptz NOT NULL,
    quote_volume numeric NOT NULL CHECK (quote_volume >= 0),
    trades bigint NOT NULL CHECK (trades >= 0),
    taker_buy_base_volume numeric NOT NULL CHECK (taker_buy_base_volume >= 0),
    taker_buy_quote_volume numeric NOT NULL CHECK (taker_buy_quote_volume >= 0),
    source_id bigint NOT NULL REFERENCES sources (source_id),
    CONSTRAINT klines_natural_key PRIMARY KEY (instrument_id, bar_interval, open_time),
    CONSTRAINT klines_high_bound CHECK (high >= GREATEST(open, close)),
    CONSTRAINT klines_low_bound CHECK (low <= LEAST(open, close)),
    CONSTRAINT klines_close_after_open CHECK (close_time > open_time)
);

CREATE INDEX klines_source ON klines (source_id);

CREATE FUNCTION klines_embargo() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.open_time >= market_embargo_sealed_from()
        OR NEW.close_time >= market_embargo_sealed_from() THEN
        RAISE EXCEPTION 'kline at % reaches the sealed window', NEW.open_time
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER klines_embargo_guard
    BEFORE INSERT ON klines
    FOR EACH ROW EXECUTE FUNCTION klines_embargo();

CREATE TRIGGER klines_append_only
    BEFORE UPDATE OR DELETE ON klines
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

-- ---------------------------------------------------------------------------
-- Funding rates
-- ---------------------------------------------------------------------------

CREATE TABLE funding_rates (
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    funding_time timestamptz NOT NULL,
    rate numeric NOT NULL CHECK (rate > -1 AND rate < 1),
    mark_price numeric CHECK (mark_price > 0),
    funding_interval_hours smallint CHECK (funding_interval_hours > 0),
    source_id bigint NOT NULL REFERENCES sources (source_id),
    CONSTRAINT funding_rates_natural_key PRIMARY KEY (instrument_id, funding_time)
);

CREATE INDEX funding_rates_source ON funding_rates (source_id);

CREATE FUNCTION funding_rates_embargo() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.funding_time >= market_embargo_sealed_from() THEN
        RAISE EXCEPTION 'funding rate at % reaches the sealed window', NEW.funding_time
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER funding_rates_embargo_guard
    BEFORE INSERT ON funding_rates
    FOR EACH ROW EXECUTE FUNCTION funding_rates_embargo();

CREATE TRIGGER funding_rates_append_only
    BEFORE UPDATE OR DELETE ON funding_rates
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

-- ---------------------------------------------------------------------------
-- Conflicts (a changed value for an existing key is refused and recorded here)
-- ---------------------------------------------------------------------------

CREATE TABLE ingestion_conflicts (
    conflict_id bigserial PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES ingestion_runs (run_id),
    kind text NOT NULL CHECK (kind IN ('kline', 'funding_rate', 'archive_sha256')),
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    source_url text NOT NULL,
    bar_interval text,
    key_time timestamptz,
    column_name text NOT NULL,
    existing_value text,
    incoming_value text,
    detected_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ingestion_conflicts_run ON ingestion_conflicts (run_id);
