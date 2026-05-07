import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

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
    """
    Checks null columns for table

    Args:
        df (dataframe): dataframe corresponding to table checking for null values

    Returns:
        null_counts (int): count of null values in each column of the dataframe
    """
    null_counts = df.isnull().sum()
    null_counts[null_counts > 0]
    print(null_counts)
    return null_counts

def drop_null_rows(df, column_names: list[str]):
    """
    Drop null rows for specific columns in specific table

    Args:
        df (df): dataframe null values wil be dropped from 
        column_names (list[str]): list of column names as string

    Returns:
        df: dataframe with dropped null values
    """
    
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

# Interaction matrix properties 
def count_total_unique_playlists(df = playlists_df):
    
    playlists_count = df['playlist_id'].nunique()
    return playlists_count

def count_total_unique_tracks(df = tracks_df):
    tracks_count = df['track_uri'].nunique()
    return tracks_count

def total_playlist_interations (df = playlist_tracks_df):
    interactions_count = len(df)
    return interactions_count

def sparsity_calculation(playlist_tracks_df, playlists_df, tracks_df):
    """Calculates the sparsity of the interaction matrix.
    Sparsity = 1 - (number of interactions / (number of unique playlists * number of unique tracks))
    
    :param playlist_tracks_df: DataFrame containing playlist-track interactions
    :param playlists_df: DataFrame containing playlist information
    :param tracks_df: DataFrame containing track information
    """
    number_of_interactions = total_playlist_interations(playlist_tracks_df)
    number_of_unique_playlists = count_total_unique_playlists(playlists_df)
    number_of_unique_tracks = count_total_unique_tracks(tracks_df)
    
    sparsity = 1 - (number_of_interactions / (number_of_unique_playlists * number_of_unique_tracks))
    
    return sparsity

def plot_distribution_of_tracks_frequency_per_playlist(playlist_tracks_df):
    """Function to plot the distribution of track frequency across playlists"""
    # Work out the number of times a track appears in playlists 
    track_frequency = playlist_tracks_df['track_uri'].value_counts()
    plt.plot(track_frequency)
    plt.show()
    
def percentage_of_tracks_in_playlists(playlist_tracks_df, threshold):
    """Function to work out the number of """
    
def run_interaction_matrix_properties():
    print("\nInteraction Matrix Properties:")
    print(f"Total unique playlists: {count_total_unique_playlists()}")
    print(f"Total unique tracks: {count_total_unique_tracks()}")
    print(f"Total playlist-track interactions: {total_playlist_interations()}")
    print(f"Sparsity of the interaction matrix: {sparsity_calculation(playlist_tracks_df, playlists_df, tracks_df)}")
    plot_distribution_of_tracks_frequency_per_playlist(playlist_tracks_df)
    
run_interaction_matrix_properties()

# Playlist level analysis 

def plot_distribution_number_of_tracks_per_playlist():
    return None

def count_playlist_with_minimum_tracks():
    return None

def plot_distribution_playlist_duration():
    return None

def plot_distribution_of_edits_per_playlist():
    return None
def plot_distribution_of_number_of_albums_per_playlist():
    return None

# Track level analysis

def find_top_20_tracks_by_playlist_occurrence():
    return None

def distribution_of_track_occurrence():
    return None

def plot_distribution_of_track_duration():
    return None

# Artist and album analysis

def find_top_artists():
    return None

def plot_artist_occurence_across_playlists()
    return None

def find_top_albums():
    return None

