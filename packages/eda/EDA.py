import psycopg2
import pandas as pd
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv

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
    df = df.dropna(subset=column_names)
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
    playlists_df = drop_null_rows(playlists_df, ['playlist_id'])
    data_type_summary(playlists_df)

    print("\nChecking Tracks DataFrame:")
    check_null_counts_per_column(tracks_df)
    tracks_df = drop_null_rows(tracks_df, ['track_uri'])
    data_type_summary(tracks_df)

    print("\nChecking Playlist-Tracks DataFrame:")
    check_null_counts_per_column(playlist_tracks_df)
    playlist_tracks_df = drop_duplicate_tracks_in_same_playlist(playlist_tracks_df, max_duplicates=3)
    data_type_summary(playlist_tracks_df)

run_data_quality_checks()

# Interaction matrix properties 
def count_total_unique_playlists(df = playlists_df):
    
    playlists_count = df['playlist_id'].nunique()
    return playlists_count

def count_total_unique_tracks(df = tracks_df):
    tracks_count = df['track_uri'].nunique()
    return tracks_count

def total_playlist_interactions(df = playlist_tracks_df):
    interactions_count = len(df)
    return interactions_count

def sparsity_calculation(playlist_tracks_df, playlists_df, tracks_df):
    """Calculates the sparsity of the interaction matrix.
    Sparsity = 1 - (number of interactions / (number of unique playlists * number of unique tracks))
    
    :param playlist_tracks_df: DataFrame containing playlist-track interactions
    :param playlists_df: DataFrame containing playlist information
    :param tracks_df: DataFrame containing track information
    """
    number_of_interactions = total_playlist_interactions(playlist_tracks_df)
    number_of_unique_playlists = count_total_unique_playlists(playlists_df)
    number_of_unique_tracks = count_total_unique_tracks(tracks_df)
    
    sparsity = 1 - (number_of_interactions / (number_of_unique_playlists * number_of_unique_tracks))
    
    return sparsity

def plot_distribution_of_tracks_frequency_per_playlist(playlist_tracks_df):
    """
    Function to plot the distribution of track frequency across playlists 
    """
    # Work out the number of times a track appears in playlists 
    track_frequency = playlist_tracks_df['track_uri'].value_counts()
    plt.hist(track_frequency, bins=50)
    plt.xlabel('Frequency in Playlists')
    plt.ylabel('Number of Tracks')
    plt.title('Distribution of Track Frequency Across Playlists')
    plt.show()

def plot_track_frequency_log_log(playlist_tracks_df):
    """
    Function to plot the distribution of track frequency across playlists on a log-log scale
    """
    # Count the number of times a track appears in playlists
    track_frequency = playlist_tracks_df['track_uri'].value_counts()
    # Number of tracks with each frequency value
    freq_of_freq = track_frequency.value_counts().sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(freq_of_freq.index, freq_of_freq.values, s=3, alpha=0.5)
    ax.set_title('Distribution of Track Frequency Across Playlists (Log-Log Scale)')
    ax.set_xlabel('Frequency in Playlists (log)')
    ax.set_ylabel('Number of Tracks (log)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

def cold_start_analysis(df=playlist_tracks_df, threshold=5):
    """Analyse the cold start problem by counting the number of tracks that appear in fewer than a specified number of playlists."""
    
    track_frequency = df['track_uri'].value_counts()
    cold_start_tracks = track_frequency[track_frequency < threshold]
    cold_start_tracks_count = cold_start_tracks.count()

    return cold_start_tracks_count

def popularity_analysis(df=playlist_tracks_df):
    """ Count the number of interactions that come from the top x% of tracks."""

    track_frequency = df['track_uri'].value_counts()
    total_interactions = track_frequency.sum()
    # Top 1% of tracks 
    popular_tracks = track_frequency[track_frequency >= track_frequency.quantile(0.99)]
    popular_interactions = popular_tracks.sum()
    popularity_percentage = (popular_interactions / total_interactions) * 100


    return popularity_percentage


def run_interaction_matrix_properties():
    print("\nInteraction Matrix Properties:")
    print(f"Total unique playlists: {count_total_unique_playlists()}")
    print(f"Total unique tracks: {count_total_unique_tracks()}")
    print(f"Total playlist-track interactions: {total_playlist_interactions()}")
    print(f"Sparsity of the interaction matrix: {sparsity_calculation(playlist_tracks_df, playlists_df, tracks_df)}")
    print(f"Number of cold start tracks (appearing in fewer than 5 playlists): {cold_start_analysis()}")
    print(f"Percentage of interactions from top 1% of tracks: {popularity_analysis()}")
    plot_distribution_of_tracks_frequency_per_playlist(playlist_tracks_df)
    
run_interaction_matrix_properties()

# Playlist level analysis 

def plot_distribution_number_of_tracks_per_playlist(df=playlist_tracks_df):
    
    count_unique_tracks_per_playlist = df.groupby('playlist_id')['track_uri'].nunique()
    # plot the distribution
    plt.hist(count_unique_tracks_per_playlist)
    plt.xlabel('Number of Unique Tracks')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Unique Tracks per Playlist')
    plt.show()

def count_playlist_with_max_tracks(df=playlists_df, min_tracks = 10):
    
    count_playlists_with_max_tracks = len(df[df['num_tracks'] < min_tracks])
    return count_playlists_with_max_tracks

def plot_distribution_playlist_duration(df=playlists_df):
    
    duration_of_each_playlist = df['duration_ms'] / 60000  # Convert duration from milliseconds to minutes  
    plt.hist(duration_of_each_playlist)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Playlist Duration')
    plt.show()
    
def plot_distribution_of_edits_per_playlist(df=playlists_df):
    
    edits_per_playlist = df['num_edits']
    plt.hist(edits_per_playlist)
    plt.xlabel('Number of Edits')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Edits per Playlist')
    plt.show()

def plot_distribution_of_number_of_albums_per_playlist(df=playlists_df):
    
    albums_per_playlist = df['num_albums']
    plt.hist(albums_per_playlist)
    plt.xlabel('Number of Albums')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Albums per Playlist')
    plt.show()


def run_playlist_level_analysis():
    print("\nPlaylist Level Analysis:")
    plot_distribution_number_of_tracks_per_playlist()
    print(f"Number of playlists with less than 10 tracks: {count_playlist_with_max_tracks()}")
    plot_distribution_playlist_duration()
    plot_distribution_of_edits_per_playlist()
    plot_distribution_of_number_of_albums_per_playlist()

run_playlist_level_analysis()

# Track level analysis
def find_top_20_tracks_by_playlist_occurrence(df=playlist_tracks_df):
    """Determine how many times each track appears in a playlist"""
    return df['track_uri'].value_counts().head(20)

def plot_distribution_of_track_duration(df = tracks_df):
    """Plot the distribution of track duration"""
    track_durations = df['duration_ms'] / 60000  # Convert duration from milliseconds to minutes
    plt.hist(track_durations, bins=50)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Track Duration')
    plt.show()

def run_track_level_analysis():
    print("\nTrack Level Analysis:")
    print("Top 20 tracks by playlist occurrence:")
    print(find_top_20_tracks_by_playlist_occurrence())
    plot_distribution_of_track_duration()   

run_track_level_analysis()

# Artist and album analysis

def find_top_artists(df=playlist_tracks_df):
    """Find the top artists by playlist occurrence"""
    top_artists = df['artist_uri'].value_counts().head(20)
    return top_artists

def plot_artist_occurence_across_playlists(df=playlist_tracks_df):
    """Find the distribution of how many times each artist appears in playlists"""
    artist_occurrence = df['artist_uri'].value_counts()
    plt.hist(artist_occurrence, bins=50)
    plt.xlabel('Frequency in Playlists')
    plt.title('Distribution of Artist Occurrence Across Playlists')
    plt.show()

def find_top_albums(df1=playlist_tracks_df, df2=tracks_df):
    """Find the top albums by playlist occurrence"""
    # Merge the playlist_tracks_df with tracks_df to get album information
    merged_df = df1.merge(df2, on='track_uri', how='left')
    top_albums = merged_df['album_name'].value_counts().head(20)
    return top_albums

def plot_album_occurence_across_playlists(df1=playlist_tracks_df, df2=tracks_df):
    """Plot the distribution of how many times each album appears in playlists"""

    # Merge the playlist_tracks_df with tracks_df to get album information
    merged_df = df1.merge(df2, on='track_uri', how='left')
    top_albums = merged_df['album_name'].value_counts()
    plt.hist(top_albums, bins=50)
    plt.xlabel('Frequency in Playlists')
    plt.title('Distribution of Album Occurrence Across Playlists')
    plt.show()

def run_artist_and_album_analysis():
    print("\nArtist and Album Analysis:")
    print("Top artists by playlist occurrence:")
    print(find_top_artists())
    plot_artist_occurence_across_playlists()
    print("Top albums by playlist occurrence:")
    print(find_top_albums())
    plot_album_occurence_across_playlists()

run_artist_and_album_analysis()
    

