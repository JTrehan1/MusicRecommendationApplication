-- =============================================================================
--  Spotify Recommender — Supabase Dev Schema
--  Run once against your Supabase project via psql or the SQL editor.
--  Schema mirrors what production (local Postgres) will use; dev just has
--  a smaller slice of MPD data loaded.
-- To run this command, you can use the psql CLI or the SQL editor in the Supabase dashboard:
-- psql "postgresql://postgres:<your-password>@db.<your-project-ref>.supabase.co:5432/postgres" \
--  -f schema/init.sql
-- =============================================================================

-- =============================================================================
--  COLLABORATIVE FILTERING TABLES  (sourced from MPD)
-- =============================================================================

-- Every unique playlist in the MPD slice
CREATE TABLE IF NOT EXISTS playlists (
    playlist_id         BIGINT          PRIMARY KEY,   -- MPD pid field
    playlist_name       TEXT            NOT NULL,
    collaborative       BOOLEAN         NOT NULL DEFAULT FALSE,
    modified_at         TIMESTAMPTZ,                   -- epoch timestamp
    num_tracks          INT             NOT NULL,a
    num_albums          INT,
    num_followers       INT,
    num_edits           INT,
    duration_ms         BIGINT,
    ingested_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Every unique track seen across all playlists (deduplicated)
CREATE TABLE IF NOT EXISTS tracks (
    track_uri           TEXT            PRIMARY KEY,   -- spotify:track:<id>
    track_id            TEXT            GENERATED ALWAYS AS (split_part(track_uri, ':', 3)) STORED,
    track_name          TEXT            NOT NULL,
    artist_uri          TEXT,
    artist_name         TEXT,
    album_uri           TEXT,
    album_name          TEXT,
    duration_ms         INT,
    -- enrichment status
    enriched_at         TIMESTAMPTZ,                   -- NULL = not yet enriched
    enrich_status       TEXT            NOT NULL DEFAULT 'pending'
                            CHECK (enrich_status IN ('pending','done','unavailable','error')),
    ingested_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracks_enrich_status ON tracks(enrich_status)
    WHERE enrich_status = 'pending';

CREATE INDEX IF NOT EXISTS idx_tracks_artist_uri ON tracks(artist_uri);

-- Many-to-many: playlist ↔ track (with position)
CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id         BIGINT          NOT NULL REFERENCES playlists(playlist_id) ON DELETE CASCADE,
    track_uri           TEXT            NOT NULL REFERENCES tracks(track_uri)      ON DELETE CASCADE,
    position            SMALLINT        NOT NULL,
    PRIMARY KEY (playlist_id, track_uri, position)
);

CREATE INDEX IF NOT EXISTS idx_pt_track_uri ON playlist_tracks(track_uri);


-- =============================================================================
--  CONTENT-BASED FILTERING TABLES  (sourced from Spotify API)
-- =============================================================================

-- Audio features from GET /audio-features (batch endpoint)
CREATE TABLE IF NOT EXISTS audio_features (
    track_uri           TEXT            PRIMARY KEY REFERENCES tracks(track_uri) ON DELETE CASCADE,
    -- Perceptual high-level features
    danceability        NUMERIC(4,3),   -- 0.0–1.0
    energy              NUMERIC(4,3),
    valence             NUMERIC(4,3),
    acousticness        NUMERIC(4,3),
    instrumentalness    NUMERIC(4,3),
    liveness            NUMERIC(4,3),
    speechiness         NUMERIC(4,3),
    -- Objective sonic descriptors
    loudness            NUMERIC(6,3),   -- dB, typically -60–0
    tempo               NUMERIC(6,3),   -- BPM
    time_signature      SMALLINT,       -- beats per bar (3–7)
    key                 SMALLINT,       -- pitch class 0–11 (-1 = no key detected)
    mode                SMALLINT,       -- 0 = minor, 1 = major
    duration_ms         INT,
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Track metadata from GET /tracks (batch endpoint)
CREATE TABLE IF NOT EXISTS track_metadata (
    track_uri           TEXT            PRIMARY KEY REFERENCES tracks(track_uri) ON DELETE CASCADE,
    isrc                TEXT,           -- international standard recording code
    explicit            BOOLEAN,
    popularity          SMALLINT,       -- 0–100, time-decayed play count proxy
    preview_url         TEXT,
    release_date        DATE,
    release_date_precision TEXT,        -- 'year' | 'month' | 'day'
    -- Nested artist/album objects flattened
    primary_artist_id   TEXT,
    primary_artist_name TEXT,
    album_id            TEXT,
    album_name          TEXT,
    album_type          TEXT,           -- 'album' | 'single' | 'compilation'
    total_tracks        SMALLINT,       -- tracks on album
    available_markets   TEXT[],         -- array of ISO 3166-1 alpha-2 codes
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Artist metadata from GET /artists (batch, up to 50 per call)
CREATE TABLE IF NOT EXISTS artist_metadata (
    artist_uri          TEXT            PRIMARY KEY,
    artist_id           TEXT            GENERATED ALWAYS AS (split_part(artist_uri, ':', 3)) STORED,
    artist_name         TEXT,
    genres              TEXT[],         -- Spotify genre tags
    popularity          SMALLINT,
    followers           INT,
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artist_genres ON artist_metadata USING GIN(genres);


-- =============================================================================
--  FEATURE STORE VIEW  (joins CBF + CF into one flat row per track)
--  This is the "model-ready" layer training code reads from.
-- =============================================================================

CREATE OR REPLACE VIEW feature_store AS
SELECT
    t.track_uri,
    t.track_id,
    t.track_name,
    t.artist_uri,
    t.artist_name,
    t.album_uri,
    t.album_name,
    t.duration_ms                       AS mpd_duration_ms,
    -- Audio features (CBF)
    af.danceability,
    af.energy,
    af.valence,
    af.acousticness,
    af.instrumentalness,
    af.liveness,
    af.speechiness,
    af.loudness,
    af.tempo,
    af.time_signature,
    af.key,
    af.mode,
    -- Track metadata (CBF)
    tm.isrc,
    tm.explicit,
    tm.popularity                       AS track_popularity,
    tm.release_date,
    tm.album_type,
    tm.total_tracks,
    -- Artist metadata (CBF)
    am.genres                           AS artist_genres,
    am.popularity                       AS artist_popularity,
    am.followers                        AS artist_followers,
    -- Collaborative signal (aggregated)
    cf.playlist_count,
    cf.avg_position,
    cf.min_position,
    cf.max_position
FROM tracks t
LEFT JOIN audio_features    af  ON af.track_uri  = t.track_uri
LEFT JOIN track_metadata    tm  ON tm.track_uri  = t.track_uri
LEFT JOIN artist_metadata   am  ON am.artist_uri = t.artist_uri
LEFT JOIN (
    SELECT
        track_uri,
        COUNT(DISTINCT playlist_id)     AS playlist_count,
        AVG(position)                   AS avg_position,
        MIN(position)                   AS min_position,
        MAX(position)                   AS max_position
    FROM playlist_tracks
    GROUP BY track_uri
) cf ON cf.track_uri = t.track_uri
WHERE t.enrich_status = 'done';

COMMENT ON VIEW feature_store IS
    'Model-ready flat feature table. Only includes fully enriched tracks.';


-- =============================================================================
--  PIPELINE AUDIT TABLE  (observability — mirrors what prod would use)
-- =============================================================================

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    stage               TEXT            NOT NULL,   -- 'mpd_ingest' | 'cbf_enrich' | 'validate'
    status              TEXT            NOT NULL,   -- 'running' | 'done' | 'failed'
    records_processed   INT,
    records_failed      INT,
    error_message       TEXT,
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ
);



