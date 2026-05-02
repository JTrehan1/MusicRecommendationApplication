import os
import logging
from dotenv import load_dotenv
from pipelines.utils.db import bulk_upsert, get_cursor
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables from .env file
load_dotenv("credentials/.env")
log = logging.getLogger(__name__)
 
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))

# Read the current state from the tracks table in database and obtain only the pending items 
def get_pending_tracks() -> list[str]:
    """Query the database for tracks that are pending enrichment.
    
    Returns a list of track IDs that need to be enriched.
    """

    with get_cursor() as cur:
        cur.execute("""
            SELECT track_id FROM tracks
            WHERE enrichment_status = 'pending'
            ORDER BY track_id
        """)
                    
        rows = cur.fetchall()

        # Extract track IDs
        track_ids = [row[0] for row in rows]
        log.info("Fetched %d pending tracks for enrichment", len(track_ids))
        
    return track_ids

# Setup the API client
def get_api_client():
    """Initialise and return an authenticated API client"""
    
    client_credentials_manager = SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"], client_secret=os.environ["SPOTIFY_CLIENT_SECRET"])
    client = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return client

# Make the API call 
def fetch_api_data(track_ids: list[str], client):
    """Make an API call for a batch of track IDs and return the enriched data.
    
    For track_metadata the API call is

    A comma-separated list of the Spotify IDs for the tracks. Maximum: 50 IDs.
    Example: ids=7ouMYWpwJ422jRcDASZB7P,4VqPOruhp5EdPBeR92t6lQ,2takcwOaAZWiXQijPHIx7B

    curl --request GET \
    --url 'https://api.spotify.com/v1/tracks?ids=7ouMYWpwJ422jRcDASZB7P%2C4VqPOruhp5EdPBeR92t6lQ%2C2takcwOaAZWiXQijPHIx7B' \
    --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'

    :param track_ids: List of Spotify track IDs to fetch data for.
    :param client: Authenticated Spotify API client.
    
    Returns a dictionary of {id: data} for each track
    """ 
    
    
    # Make the API call for the batch of track IDs
    response_audio_features = client.audio_features(track_ids)
    response_track_features = client.tracks(track_ids)
    
    artist_ids = list(
        {track["artists"][0]["id"] for track in response_track_features["tracks"] if track is not None and track.get("artists")})
    
    
    response_artist_features = client.artists(artist_ids)
    
    # Store responses in dicts for easy access
    
    audio_features = {item["id"]: item for item in response_audio_features if item is not None}
    track_features = {item["id"]: item for item in response_track_features["tracks"] if item is not None}
    artist_features = {item["id"]: item for item in response_artist_features["artists"] if item is not None}
    
    return audio_features, track_features, artist_features

# Transform the API response into the format required for database upsert

def build_audio_features_table(track_id: str, api_data: dict) -> dict:
    """Transform the API response for a single track into the format required for database upsert.

    Called in main loop to populate the audio_features table.

    :param track_id: Spotify track ID.
    :param api_data: API response data for the track.
    
    Returns a dictionary with keys matching the database columns.
    """

    audio_features_dict = {
            "track_uri":     f"spotify:track:{track_id}",
            "danceability":  api_data.get("danceability"),
            "energy":        api_data.get("energy"),
            "valence":       api_data.get("valence"),
            "acousticness":   api_data.get("acousticness"),
            "instrumentalness":   api_data.get("instrumentalness"),
            "liveness":      api_data.get("liveness"),
            "speechiness":     api_data.get("speechiness"),
            "loudness":       api_data.get("loudness"),
            "tempo":         api_data.get("tempo"),
            "time_signature": api_data.get("time_signature"),
            "key":            api_data.get("key"),
            "mode":           api_data.get("mode"),
            "duration_ms":    api_data.get("duration_ms"),
        }
    
    return audio_features_dict

def build_track_metadata_table(track_id: str, api_data: dict) -> dict:
    """
    Transform the API response for a single track into the format required for database upsert.
    
    Called in main loop to populate the track_metadata table.

    :param track_id: Spotify track ID.
    :param api_data: API response data for the track.

    Returns a dictionary with keys matching the database columns."""
    
    track_metadata_dict = {
        "track_uri":    f"spotify:track:{track_id}",
        "isrc": api_data.get("isrc"),
        "explicit": api_data.get("explicit"),
        "popularity": api_data.get("popularity"),
        "preview_url": api_data.get("preview_url"),
        "release_date": api_data.get("album", {}).get("release_date"),
        "release_date_precision": api_data.get("album", {}).get("release_date_precision"),
        "primary_artist_id": api_data.get("artists", [{}])[0].get("id"),
        "primary_artist_name": api_data.get("artists", [{}])[0].get("name"),
        "album_id": api_data.get("album", {}).get("id"),
        "album_name": api_data.get("album", {}).get("name"),
        "album_type": api_data.get("album", {}).get("album_type"),
        "total_tracks": api_data.get("album", {}).get("total_tracks"),
        "available_markets": api_data.get("available_markets"),  
    }
    
    return track_metadata_dict

def build_artist_metadata_table(artist_id: str, api_data: dict) -> dict:
    """
    Transform the API response for a single artist into the format required for database upsert.
    
    Called in main loop to populate the artist_metadata table.

    :param artist_id: Spotify artist ID.
    :param api_data: API response data for the artist.

    Returns a dictionary with keys matching the database columns."""
    
    artist_metadata_dict = {
        "artist_uri": f"spotify:artist:{artist_id}",
        "artist_id": artist_id,
        "artist_name": api_data.get("name"),
        "genres": api_data.get("genres"),
        "popularity": api_data.get("popularity"),
        "followers": api_data.get("followers", {}).get("total"),
    }
    
    return artist_metadata_dict

def update_enrichment_status(track_ids: list, status: str) -> None:
    """Update the enrichment status for a given track_id in the tracks table.
    
    Status options:
    - 'pending': track is pending enrichment (default state)
    - 'enriched': track has been successfully enriched and upserted into the database
    - 'error': enrichment failed for this track 

    :param track_ids: List of Spotify track IDs to update status for.
    :param status: New enrichment status to set for the track IDs.
    """

    if not track_ids:
        return
    
    with get_cursor() as cur:
        cur.execute("""
            UPDATE tracks
            SET enrichment_status = %s
            WHERE track_id = ANY(%s)
  
        """, 
        (status, track_ids))


def batch_maker(full_list: list, batch_size: int) -> list[list]:
    """Utility function to split a full list into batches of a specified size.
    
    :param full_list: The complete list to be split into batches.
    :param batch_size: The desired size of each batch.

    Returns a list of batches, where each batch is a sublist of the original list.
    """

    batch_list = []
    for i in range(0, len(full_list), batch_size):
        batch_list.append(full_list[i : i + batch_size])
    
    return batch_list

def run():
    """ Orchestrate the enrichment process: fetch pending tracks, make API calls, transform data, and upsert into database. """

    # STEP 1: Fetch pending tracks from database
    pending_track_ids = get_pending_tracks()

    # Guard against empty pending stracks 
    if not pending_track_ids:
        log.info("No pending tracks found for enrichment. Exiting.")
        return

    # Step 2: Set up the API client
    api_client = get_api_client()

    # Step 3 create the batches of track IDs to process
    batches_of_tracks = batch_maker(pending_track_ids, BATCH_SIZE)

    log.info("Processing %d batches of up to %d", len(batches_of_tracks), BATCH_SIZE)
 
    total_done      = 0
    total_unavail   = 0

    # Step 4: Loop through each batch and make API calls
    for batch in batches_of_tracks:
        done_tracks = []
        unavailable_tracks = []

        # Make the API call for the batch of track IDs
        audio_features, track_features, artist_features = fetch_api_data(batch, api_client)

        # For each batch build the tables 
        for track_id in batch:
            

            audio_metadata_dict = build_audio_features_table(track_id, audio_features.get(track_id, {}))
            track_metadata_dict = build_track_metadata_table(track_id, track_features.get(track_id, {}))
            artist_metadata_dict = build_artist_metadata_table(track_id, artist_features.get(track_id, {}))

            # Upsert the data into the database 
            bulk_upsert("audio_features", [audio_metadata_dict], conflict_col="track_uri", update_cols=list(audio_metadata_dict.keys()))
            bulk_upsert("track_metadata", [track_metadata_dict], conflict_col="track_uri", update_cols=list(track_metadata_dict.keys()))
            bulk_upsert("artist_metadata", [artist_metadata_dict], conflict_col="artist_uri", update_cols=list(artist_metadata_dict.keys()))

            # Update the enrichment status for the track in the tracks table to 'enriched'
            update_enrichment_status([track_id], status="enriched")

        total_done += len(batch)

if __name__ == "__main__":
    run()





