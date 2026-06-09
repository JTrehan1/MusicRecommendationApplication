# Unit tests for data_ingestion.py using  pytest

import pytest
import pandas as pd 

# Item under test 
from data_ingestion import DataIngestion, IngestionOutput  

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "playlist_id": [1, 1, 2, 2],
        "track_uri": ["track1", "track2", "track3", "track4"],
    })

@pytest.fixture
def ingestor(sample_data):
    return DataIngestion(sample_data)

def test_map_playlist_to_ids(ingestor):
    playlist_to_idx, idx_to_playlist = ingestor._map_playlist_to_ids()
    assert playlist_to_idx == {1: 0, 2: 1}
    assert idx_to_playlist == {0: 1, 1: 2}

def test_map_track_to_ids(ingestor):
    track_to_idx, idx_to_track = ingestor._map_track_to_ids()
    assert track_to_idx == {"track1": 0, "track2": 1, "track3": 2, "track4": 3}
    assert idx_to_track == {0: "track1", 1: "track2", 2: "track3", 3: "track4"}

@pytest.mark.parametrize(
    "playlist_id_to_index, track_uri_to_index",
    [
        pytest.param(
            {1: 0, 2: 1},
            {"track1": 0, "track2": 1, "track3": 2, "track4": 3},
            id="simple_mapping",
        ),
    ],
)

def test_create_sparse_matrix(ingestor, playlist_id_to_index, track_uri_to_index):
    sparse_matrix = ingestor._create_sparse_matrix(playlist_id_to_index, track_uri_to_index)
    assert sparse_matrix.shape == (2, 4)  # 2 playlists, 4 tracks
    assert sparse_matrix[0, 0] == 1  # Playlist 1 has track1
    assert sparse_matrix[0, 1] == 1  # Playlist 1 has track2
    assert sparse_matrix[1, 2] == 1  # Playlist 2 has track3
    assert sparse_matrix[1, 3] == 1  # Playlist 2 has track4
    assert sparse_matrix[0, 2] == 0  # Playlist 1 does not have track3
    assert sparse_matrix[0, 3] == 0  # Playlist 1 does not have track4
    assert sparse_matrix[1, 0] == 0  # Playlist 2 does not have track1
    assert sparse_matrix[1, 1] == 0  # Playlist 2 does not have track2  

def test_ingestion_output(ingestor):
    output = ingestor.ingest()
    assert isinstance(output, IngestionOutput)
    assert output.sparse_matrix.shape == (2, 4)  # 2 playlists, 4 tracks
    assert output.playlist_id_to_index == {1: 0, 2: 1}
    assert output.index_to_playlist_id == {0: 1, 1: 2}
    assert output.track_uri_to_index == {"track1": 0, "track2": 1, "track3": 2, "track4": 3}
    assert output.index_to_track_uri == {0: "track1", 1: "track2", 2: "track3", 3: "track4"}
    