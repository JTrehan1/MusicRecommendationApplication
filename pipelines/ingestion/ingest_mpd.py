"""ingestion/ingest_mpd.py - Loads the mpd.slice.XX-YY.json slice files into the Supabase database.

WHAT THIS SCRIPT DOES:
─────────────────────────────────────────
1. Finds all MPD slice files in your MPD_ROOT folder
2. Parses each slice into three groups of rows:
     - playlists   (one row per playlist)
     - tracks      (one row per UNIQUE track — deduplicated)
     - playlist_tracks (one row per track-in-playlist — NOT deduplicated)
3. Writes each group to Postgres in FK order:
     playlists FIRST → tracks SECOND → playlist_tracks THIRD
     (order matters — playlist_tracks has foreign keys to both)
4. Logs a pipeline_runs audit row when done

TABLES WRITTEN:
    playlists, tracks, playlist_tracks, pipeline_runs

RUN:
    python -m pipelines.ingestion.01_ingest_mpd
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Two main DB helpers — bulk_upsert for writing rows,
# get_cursor for raw SQL (used in the audit log functions below)
from pipelines.utils.db import bulk_upsert, get_cursor

load_dotenv("credentials/.env") 
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

# Retrieve the values from the .env file 
MPD_ROOT = Path(os.environ["MPD_SAMPLE"])
UPSERT_BATCH = int(os.environ["UPSERT_BATCH"]) 

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PARSING
#  Goal: open one slice file and return three lists of dicts ready for DB insert
# ══════════════════════════════════════════════════════════════════════════════

def _parse_slice(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse a single mpd.slice.*.json file.

    Returns:
        playlists      — list of dicts, one per playlist in this slice
        tracks         — list of dicts, one per UNIQUE track uri seen
        playlist_tracks — list of dicts, one per track-position in a playlist

    DESIGN NOTES:
    ─────────────
    - Open the file and load the JSON. The top-level key is "playlists".
    - Loop over each playlist. Each has a "pid" (playlist id) and a "tracks" list.
    - For each track in the playlist:
        → ALWAYS append to playlist_tracks (preserves the CF co-occurrence signal)
        → ONLY append to tracks if you haven't seen that track_uri before
          (use a local set called seen_uris to track this)
    - Column names in your dicts must exactly match your Postgres column names.
    - Guard against very long strings — slice [:500] on name fields.
    - The "modified_at" field in MPD is a Unix epoch integer → convert to datetime.
    - The "collaborative" field is the string "true"/"false" → convert to bool.

    MPD track fields available to you:
        item["track_uri"], item["track_name"], item["artist_uri"],
        item["artist_name"], item["album_uri"], item["album_name"],
        item["duration_ms"], item["pos"]

    MPD playlist fields available to you:
        pl["pid"], pl["name"], pl["collaborative"], pl["modified_at"],
        pl["num_tracks"], pl["num_albums"], pl["num_followers"],
        pl["num_edits"], pl["duration_ms"]
    """

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    playlists = []
    tracks = []
    playlist_tracks = []
    seen_uris: set[str] = set() # to track which track_uris already added to tracks to remove duplicates

    for playlist in data.get("playlists", []):

        pid = playlist["pid"]
        playlists.append({
            'playlist_id': pid,
            'playlist_name': playlist.get("name", "")[:500],
            'collaborative': playlist.get("collaborative", "false").lower() == "true",
            'modified_at': datetime.fromtimestamp(playlist.get("modified_at", 0), tz=timezone.utc) if playlist.get("modified_at") else None,
            'num_tracks': playlist.get("num_tracks", 0),
            'num_albums': playlist.get("num_albums"),
            'num_followers': playlist.get("num_followers"),
            'num_edits': playlist.get("num_edits"),
            'duration_ms': playlist.get("duration_ms")
            })

        # Only need these three to match what songs are in what playlist. 
        # Extra information achieved by joining playlist_tracks table with tracks table on track_uri
        for item in playlist.get("tracks", []):
            uri = item["track_uri"]
            playlist_tracks.append({
                'playlist_id': pid,
                'track_uri': uri,
                'position': item["pos"]
            })

            if uri not in seen_uris:
                seen_uris.add(uri)
                tracks.append({
                    'track_uri': uri,
                    'track_name': item.get("track_name"),
                    'artist_uri': item.get("artist_uri"),
                    'artist_name': item.get("artist_name"),
                    'album_uri': item.get("album_uri"),
                    'album_name': item.get("album_name"),
                    'duration_ms': item.get("duration_ms"),
                    'enrich_status': "pending" # pending until postgres updates
                })

    return playlists, tracks, playlist_tracks


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — WRITING TO DB
#  Goal: flush a list of row dicts to the correct Postgres table
# ══════════════════════════════════════════════════════════════════════════════

def _flush(table: str, rows: list[dict], conflict_col: str, update_cols: list[str]) -> int:
    """
    Write a batch of rows to a Postgres table using bulk_upsert.
    Returns the number of rows affected.

    DESIGN NOTES:
    ─────────────
    - This is just a thin wrapper around bulk_upsert that also logs the result.
    - conflict_col tells Postgres what to do if the row already exists
      (e.g. "playlist_id" for playlists, "track_uri" for tracks,
       "playlist_id, track_uri" for playlist_tracks).
    - update_cols lists which columns to overwrite on conflict.
      For tracks, do NOT include enrich_status — you don't want to reset
      a track back to "pending" if it was already enriched.

    """
    n = bulk_upsert(table, rows, conflict_col, update_cols)
    log.debug("  upserted %d rows → %s", n, table)
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MAIN ORCHESTRATION
#  Goal: find all slice files, parse each one, write to DB in batches
# ══════════════════════════════════════════════════════════════════════════════

def run():
    """
    Main entry point. Orchestrates the full ingestion run.

    DESIGN NOTES:
    ─────────────
    - Use Path.glob("mpd.slice.*.json") to find slice files.
    - Raise an error early if no files are found (fail fast).
    - Open a pipeline_runs audit row before processing (call _start_run).
    - Loop over files with tqdm for a progress bar.
    - For each file: parse it, then flush in FK order:
        1. playlists
        2. tracks
        3. playlist_tracks
      Write in batches of UPSERT_BATCH rows — don't write all at once.
    - Catch parse errors per-file so one bad file doesn't kill the whole run.
    - Track totals and failed counts for the audit log.
    - Call _finish_run with "done" or "failed" at the end.
    """

    slice_files = sorted(MPD_ROOT.glob("mpd.slice.*.json"))
    if not slice_files:
        raise FileNotFoundError(f"No slice files found in {MPD_ROOT} to upload to databse.")
    log.info("Found %d MPD slice files in %s", len(slice_files), MPD_ROOT)

    run_id = _start_run("mpd_ingest")

    total_playlists = total_tracks = total_pt = 0
    failed = 0

    try:
        for path in tqdm(slice_files, desc="MPD slices", unit="file"):

            # On exception: log the error, increment failed, continue to next file
            try:
                playlists, tracks, playlist_tracks = _parse_slice(path)
            except Exception as exc:
                log.error("Failed to parse %s: %s", path.name, exc)
                failed += 1
                continue

            update_cols  = ["playlist_id", "playlist_name", "collaborative", "modified_at", "num_tracks", "num_albums"
            , "num_followers", "num_edits", "duration_ms"]
            for i in range(0, len(playlists), UPSERT_BATCH):
                _flush("playlists", playlists[i:i+UPSERT_BATCH], "playlist_id", update_cols)


            update_cols  = ["track_name","artist_uri","artist_name",
                        "album_uri","album_name","duration_ms"]
            #                (don't overwrite enrich_status if already done)
            for i in range(0, len(tracks), UPSERT_BATCH):
                _flush("tracks", tracks[i:i+UPSERT_BATCH], "track_uri", update_cols)


            for i in range(0, len(playlist_tracks), UPSERT_BATCH):
                _flush("playlist_tracks", playlist_tracks[i:i+UPSERT_BATCH], "playlist_id, track_uri, position", [])

            total_playlists += len(playlists)
            total_tracks    += len(tracks)
            total_pt        += len(playlist_tracks)

        log.info(
            "Ingestion complete — playlists: %d | tracks: %d | "
            "playlist_tracks: %d | failed files: %d",
            total_playlists, total_tracks, total_pt, failed,
        )
        _finish_run(run_id, "done", total_playlists + total_tracks, failed)

    except Exception as exc:
        _finish_run(run_id, "failed", total_playlists + total_tracks, failed, str(exc))
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — AUDIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _start_run(stage: str) -> str:
    """Insert a 'running' row into pipeline_runs and return the run_id."""
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO pipeline_runs (stage, status)
               VALUES (%s, 'running') RETURNING run_id""",
            (stage,),
        )
        return str(cur.fetchone()[0])


def _finish_run(run_id: str, status: str, processed: int, failed: int,
                error: str | None = None):
    """Update the pipeline_runs row with final status and counts."""
    with get_cursor() as cur:
        cur.execute(
            """UPDATE pipeline_runs
               SET status=%s, records_processed=%s, records_failed=%s,
                   error_message=%s, finished_at=NOW()
               WHERE run_id=%s""",
            (status, processed, failed, error, run_id),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()