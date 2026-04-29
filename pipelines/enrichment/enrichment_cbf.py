import os
import logging
from dotenv import load_dotenv
from pipelines.utils import bulk_upsert, get_cursor

# Load environment variables from .env file
load_dotenv("configs/.env")
log = logging.getLogger(__name__)
 
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))

# Read the current state from the tracks table in database and obtain only the pending items 
def get_pending_tracks() -> list[str]:
    """Query the database for tracks that are pending enrichment."""

    with get_cursor() as cur:
        cur.execute("""
            SELECT track_id FROM tracks
            WHERE enrichment_status = 'pending'
            ORDER BY track_id
        """
                    
        rows = cur.fetchall()

        # Extract track IDs
        track_ids = [row[0] for row in rows]
        log.info("Fetched %d pending tracks for enrichment", len(track_ids))
        
    return track_ids

# Setup the API client
def get_api_client():
