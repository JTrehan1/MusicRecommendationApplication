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
        
    
    def _remove_duplicate_tracks_within_playlists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tracks from input dataframe, in this case playlist tracks

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        pass
        
    def _remove_playlists_with_few_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove playlists with fewer tracks than the specified minimum from the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        pass
    
    def _remove_tracks_with_few_occurrences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove tracks that occur in fewer playlists than the specified minimum from the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing playlist tracks.
        """
        pass
        
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Clean the data applying the cleaning functions for the correct dataframes"""   
        playlist_tracks_df = self._remove_duplicate_tracks_within_playlists(df) 
        playlist_tracks_df = self._remove_playlists_with_few_tracks(playlist_tracks_df)
        playlist_tracks_df = self._remove_tracks_with_few_occurrences(playlist_tracks_df)
        return playlist_tracks_df
    