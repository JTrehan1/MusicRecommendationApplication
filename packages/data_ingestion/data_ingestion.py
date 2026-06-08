# DataIngestion class to handle the logic for ingesting data from the cleaned playlists_tracks data ready for use by model 
import scipy.sparse as sp
import logging

from dataclasses import dataclass
from pandas import DataFrame

logger = logging.getLogger(__name__)

@dataclass
class IngestionOutput:
    """Dataclass to encapsulate the output of the data ingestion process, including the sparse matrix and the mappings."""
    sparse_matrix: sp.csr_matrix
    playlist_id_to_index: dict
    track_uri_to_index: dict
    index_to_playlist_id: dict
    index_to_track_uri: dict

class DataIngestion:
    """Class for carrying out the data ingestion process for the model training pipeline. """

    def __init__(self, data: DataFrame):
        """Instantiates the DataIngestion class, taking in the cleaned data as argument."""
        self.data = data
        
    def _map_playlist_to_ids(self):
        """ Maps the playlist ids to unique ordered indexes to populate the sparse matrix for model training.
        Returns:
            dict: A mapping of playlist_id to unique ordered index."""

        unique_playlist_ids = self.data['playlist_id'].unique()
        # Create the index mapping for playlist ids
        playlists_to_idx =  {playlist_id: idx for idx, playlist_id in enumerate(unique_playlist_ids)}
        idx_to_playlists = {idx: playlist_id for playlist_id, idx in playlists_to_idx.items()}
        return playlists_to_idx, idx_to_playlists

    def _map_track_to_ids(self):
        """Maps the track ids to unique ordered indexes to populate the sparse matrix for model training.
        Returns:
            dict: A mapping of track_uri to unique ordered index.
        """
        unique_track_uris = self.data['track_uri'].unique()
        track_to_idx = {track_uri: idx for idx, track_uri in enumerate(unique_track_uris)}
        idx_to_tracks = {idx: track_uri for track_uri, idx in track_to_idx.items()}
        return track_to_idx, idx_to_tracks

    def _map_ids_to_playlists(self):
        """Maps the unique ordered indexes back to the playlist ids for use in evaluation and inference.
        Returns:
            dict: A mapping of unique ordered index to playlist_id.
        """
        pass

    def _map_ids_to_tracks(self):
        """Maps the unique ordered indexes back to the track ids for use in evaluation and inference.
        Returns:
            dict: A mapping of unique ordered index to track_uri.
        """
        pass

    def _create_sparse_matrix(self):
        """Create the CSR sparse matrix to use as input for model training."""
        pass
    
    def ingest(self) -> IngestionOutput:
        """Main method to carry out the data ingestion process, including mapping and sparse matrix creation.
        Returns:
            IngestionOutput: A dataclass containing the sparse matrix and the mappings."""
        
        logger.info("Starting data ingestion process.")
        pass
