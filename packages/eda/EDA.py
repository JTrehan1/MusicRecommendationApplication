import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv("credentials/.env")

conn = psycopg2.connect(
    host=os.environ["SUPABASE_HOST"],
    port=os.environ.get("SUPABASE_PORT", 6543),
    dbname=os.environ.get("SUPABASE_DB", "postgres"),
    user=os.environ["SUPABASE_USER"],
    password=os.environ["SUPABASE_PASSWORD"],
    sslmode="require"
)

playlists_df = pd.read_sql_query("SELECT * FROM playlists", conn)
tracks_df = pd.read_sql_query("SELECT * FROM tracks", conn)
playlist_tracks_df = pd.read_sql_query("SELECT * FROM playlist_tracks", conn)

conn.close()

print("Playlists DataFrame:")
print(playlists_df.head())

print("\nTracks DataFrame:")
print(tracks_df.head())

print("\nPlaylist-Tracks DataFrame:")
print(playlist_tracks_df.head())

# Data Quality Checks
print("\nData Quality Checks:")
def check_null_counts_per_column(df):
    null_counts = df.isnull().sum()
    null_counts[null_counts > 0]
    print(null_counts)
    return null_counts

def drop_null_rows(df, column_names: list[str]):
    
    before_rows = df.shape[0]
    df.dropna(subset=column_names)
    after_rows = df.shape[0]
    print(f"Dropped {before_rows - after_rows} rows with null values in column '{column_names}'")

    return df

def drop_duplicate_tracks_in_same_playlist(df, max_duplicates=3):
    before_rows = df.shape[0]

    df = df.sort_values(by=['position'])  # Ensure consistent ordering
    # Count the number of occurrences of each track in each playlist
    counts = df.groupby(['playlist_id', 'track_uri']).cumcount()

    df = df[counts < max_duplicates]

    after_rows = df.shape[0]
    print(f"Dropped {before_rows - after_rows} duplicate tracks in the same playlist")
    
    return df

def data_type_summary(df):
    print(df.dtypes)
    return df.dtypes

def run_data_quality_checks():
    print("\nChecking Playlists DataFrame:")
    check_null_counts_per_column(playlists_df)
    drop_null_rows(playlists_df, ['playlist_id'])
    data_type_summary(playlists_df)

    print("\nChecking Tracks DataFrame:")
    check_null_counts_per_column(tracks_df)
    drop_null_rows(tracks_df, ['track_uri'])
    data_type_summary(tracks_df)

    print("\nChecking Playlist-Tracks DataFrame:")
    check_null_counts_per_column(playlist_tracks_df)
    drop_duplicate_tracks_in_same_playlist(playlist_tracks_df, max_duplicates=3)
    data_type_summary(playlist_tracks_df)

run_data_quality_checks()