# Data cleaning python script. Created as a class for the following reasons:
# 1. To encapsulate the data cleaning logic and make it reusable for different configurations.
# 2. Extendability: It allows for easy addition of new cleaning methods or configurations without modifying the existing code structure.

import pandas as pd


class DataCleaning:
    
    def __init__(self, config: dict):
        self.config = config       
        self.min_playlist_tracks = config.get("min_playlist_tracks")
        self.min_track_occurrences = config.get("min_track_occurrences")
        self.max_duplicate_tracks = config.get("max_duplicate_tracks")
        
        required_keys = ["min_playlist_tracks", "min_track_occurrences", "max_duplicate_tracks"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration parameters:{missing_keys}")
        
    
    def _remove_duplicate_tracks_within_playlists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tracks within the threshold from input dataframe, in this case playlist tracks. 

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        before = df.shape[0]
        # Sort the values by playlist_id and position to ensure duplicates are adjacent
        df = df.sort_values(['playlist_id', 'position'])
        # Groupby playlist_id and track_uri and obtain the count of occurrences
        counts = df.groupby(['playlist_id', 'track_uri']).cumcount()
        df = df[counts < self.max_duplicate_tracks]
        after = df.shape[0]
        print(f"Dropped {before - after} excess duplicate tracks")
        return df 

        
    def _remove_playlists_with_few_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove playlists with fewer tracks than the specified minimum from the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        before = df.shape[0]
        # Groupby playlist id and obtain the count of tracks in each playlist
        playlist_lengths = df.groupby('playlist_id').size()
        # Filter out playlist_ids that dont reach the threshold 
        valid_playlists = playlist_lengths[playlist_lengths >= self.min_playlist_tracks]
        df = df[df['playlist_id'].isin(valid_playlists.index)]
        after = df.shape[0]
        print(f"Dropped {before - after} playlists with few tracks")
        return df
    
    def _remove_tracks_with_few_occurrences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove tracks that occur in fewer playlists than the specified minimum from the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        before = df.shape[0]
        # Count number of track occurrences across all playlists
        track_occurrences = df['track_uri'].value_counts()
        # Filter out track_uris that dont reach the threshold
        valid_tracks = track_occurrences[track_occurrences >= self.min_track_occurrences]
        df = df[df['track_uri'].isin(valid_tracks.index)]
        after = df.shape[0]
        print(f"Dropped {before - after} tracks with few occurrences")
        return df
        
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Clean the data applying the cleaning functions for the correct dataframes"""   
        playlist_tracks_df = self._remove_duplicate_tracks_within_playlists(df) 
        playlist_tracks_df = self._remove_playlists_with_few_tracks(playlist_tracks_df)
        playlist_tracks_df = self._remove_tracks_with_few_occurrences(playlist_tracks_df)
        return playlist_tracks_df
    