-- Migration 0002: USD-M daily open-interest ``metrics`` and ``bookDepth`` archives.
--
-- Adds two evidence tables beside klines / funding_rates, populated only by opt-in
-- ingestion of the public Binance Vision USD-M **daily** archives
--   futures/um/daily/metrics/SYM/SYM-metrics-YYYY-MM-DD.zip
--   futures/um/daily/bookDepth/SYM/SYM-bookDepth-YYYY-MM-DD.zip
-- Same contract as migration 0001 (never edit that file; the ledger refuses it):
--   * exact NUMERIC values; NaN and +/-Infinity are rejected (``value < 'Infinity'``
--     is false for both NaN and +Infinity in PostgreSQL), as are negatives;
--   * append-only: UPDATE / DELETE are rejected by forbid_row_mutation(), TRUNCATE by
--     forbid_truncate(); a duplicate natural key is a primary-key violation, and the
--     application compares re-ingested rows and records differences as conflicts;
--   * the FS1 sealed window (>= 2026-11-01T00:00Z) can never be stored and the
--     warm-up span needs an authorized running ingestion run: both go through the
--     unchanged market_require_storable() of migration 0001, so there is no new
--     release path;
--   * only USD-M instruments and only daily one-UTC-day sources may carry these kinds.
-- Existing tables are only widened where a CHECK listed the closed set of kinds.

-- ---------------------------------------------------------------------------
-- Widen the closed kind sets (sources, ingestion_conflicts)
-- ---------------------------------------------------------------------------

ALTER TABLE sources DROP CONSTRAINT sources_kind_check;
ALTER TABLE sources ADD CONSTRAINT sources_kind_check
    CHECK (kind IN ('klines', 'funding_rate', 'metrics', 'book_depth'));

ALTER TABLE sources DROP CONSTRAINT sources_kind_interval;
ALTER TABLE sources ADD CONSTRAINT sources_kind_interval CHECK (
    (kind = 'klines' AND bar_interval IS NOT NULL)
    OR (kind IN ('funding_rate', 'metrics', 'book_depth') AND bar_interval IS NULL)
);

-- metrics / bookDepth exist only as one-UTC-day archives (24 hours, not a calendar
-- interval, so the session time zone cannot change the arithmetic)
ALTER TABLE sources ADD CONSTRAINT sources_derivatives_daily CHECK (
    kind NOT IN ('metrics', 'book_depth')
    OR (
        period_kind = 'daily'
        AND period_end = period_start + INTERVAL '24 hours'
        AND period_start = date_trunc('day', period_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
    )
);

ALTER TABLE ingestion_conflicts DROP CONSTRAINT ingestion_conflicts_kind_check;
ALTER TABLE ingestion_conflicts ADD CONSTRAINT ingestion_conflicts_kind_check
    CHECK (kind IN (
        'kline', 'funding_rate', 'archive_sha256', 'open_interest_metric', 'book_depth'
    ));

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

-- NaN sorts above +Infinity, so ``value < 'Infinity'`` rejects both; -Infinity fails >= 0.
CREATE FUNCTION market_is_finite_nonnegative(value numeric) RETURNS boolean
LANGUAGE sql IMMUTABLE AS $$
    SELECT value >= 0 AND value < 'Infinity'::numeric
$$;

-- These two kinds are USD-M only; the instrument row carries the market.
CREATE FUNCTION market_require_usds_m(target_instrument bigint, what text) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM instruments
        WHERE instrument_id = target_instrument AND market = 'usds_m'
    ) THEN
        RAISE EXCEPTION '% exists only for USD-M instruments', what
            USING ERRCODE = 'check_violation';
    END IF;
END
$$;

CREATE FUNCTION sources_derivatives_market() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.kind IN ('metrics', 'book_depth') THEN
        PERFORM market_require_usds_m(NEW.instrument_id, NEW.kind || ' source ' || NEW.url);
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER sources_derivatives_market_guard
    BEFORE INSERT OR UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION sources_derivatives_market();

-- ---------------------------------------------------------------------------
-- Open-interest metrics (5-minute snapshots)
-- ---------------------------------------------------------------------------

CREATE TABLE open_interest_metrics (
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    create_time timestamptz NOT NULL,
    sum_open_interest numeric NOT NULL
        CHECK (market_is_finite_nonnegative(sum_open_interest)),
    sum_open_interest_value numeric NOT NULL
        CHECK (market_is_finite_nonnegative(sum_open_interest_value)),
    count_toptrader_long_short_ratio numeric NOT NULL
        CHECK (market_is_finite_nonnegative(count_toptrader_long_short_ratio)),
    sum_toptrader_long_short_ratio numeric NOT NULL
        CHECK (market_is_finite_nonnegative(sum_toptrader_long_short_ratio)),
    count_long_short_ratio numeric NOT NULL
        CHECK (market_is_finite_nonnegative(count_long_short_ratio)),
    sum_taker_long_short_vol_ratio numeric NOT NULL
        CHECK (market_is_finite_nonnegative(sum_taker_long_short_vol_ratio)),
    source_id bigint NOT NULL REFERENCES sources (source_id),
    CONSTRAINT open_interest_metrics_natural_key PRIMARY KEY (instrument_id, create_time)
);

CREATE INDEX open_interest_metrics_source ON open_interest_metrics (source_id);

CREATE FUNCTION open_interest_metrics_embargo() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM market_require_usds_m(NEW.instrument_id, 'open interest metrics');
    PERFORM market_require_storable(
        NEW.create_time, 'open interest metric at ' || NEW.create_time);
    RETURN NEW;
END
$$;

CREATE TRIGGER open_interest_metrics_embargo_guard
    BEFORE INSERT ON open_interest_metrics
    FOR EACH ROW EXECUTE FUNCTION open_interest_metrics_embargo();

CREATE TRIGGER open_interest_metrics_append_only
    BEFORE UPDATE OR DELETE ON open_interest_metrics
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

CREATE TRIGGER open_interest_metrics_no_truncate
    BEFORE TRUNCATE ON open_interest_metrics
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_truncate();

-- ---------------------------------------------------------------------------
-- Book depth (cumulative depth inside signed percentage bands)
-- ---------------------------------------------------------------------------

CREATE TABLE book_depth (
    instrument_id bigint NOT NULL REFERENCES instruments (instrument_id),
    snapshot_time timestamptz NOT NULL,
    percentage smallint NOT NULL CHECK (percentage BETWEEN -100 AND 100 AND percentage <> 0),
    depth numeric NOT NULL CHECK (market_is_finite_nonnegative(depth)),
    notional numeric NOT NULL CHECK (market_is_finite_nonnegative(notional)),
    source_id bigint NOT NULL REFERENCES sources (source_id),
    CONSTRAINT book_depth_natural_key PRIMARY KEY (instrument_id, snapshot_time, percentage)
);

CREATE INDEX book_depth_source ON book_depth (source_id);

CREATE FUNCTION book_depth_embargo() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM market_require_usds_m(NEW.instrument_id, 'book depth');
    PERFORM market_require_storable(
        NEW.snapshot_time, 'book depth at ' || NEW.snapshot_time);
    RETURN NEW;
END
$$;

CREATE TRIGGER book_depth_embargo_guard
    BEFORE INSERT ON book_depth
    FOR EACH ROW EXECUTE FUNCTION book_depth_embargo();

CREATE TRIGGER book_depth_append_only
    BEFORE UPDATE OR DELETE ON book_depth
    FOR EACH ROW EXECUTE FUNCTION forbid_row_mutation();

CREATE TRIGGER book_depth_no_truncate
    BEFORE TRUNCATE ON book_depth
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_truncate();
