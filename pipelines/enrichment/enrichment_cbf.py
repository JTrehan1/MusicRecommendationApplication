import os
import logging
from dotenv import load_dotenv
from pipelines.utils import bulk_upsert, get_cursor
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
def fetch_track_api_data(track_ids: list[str], client):
    """Make an API call for a batch of track IDs and return the enriched data.
    
    For track_metadata the API call is

    A comma-separated list of the Spotify IDs for the tracks. Maximum: 50 IDs.
    Example: ids=7ouMYWpwJ422jRcDASZB7P,4VqPOruhp5EdPBeR92t6lQ,2takcwOaAZWiXQijPHIx7B

    curl --request GET \
    --url 'https://api.spotify.com/v1/tracks?ids=7ouMYWpwJ422jRcDASZB7P%2C4VqPOruhp5EdPBeR92t6lQ%2C2takcwOaAZWiXQijPHIx7B' \
    --header 'Authorization: Bearer 1POdFZRZbvb...qqillRxMr2z'
    
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