# Unit tests for dataCleaning.py using monkeypatch and pytest

import pytest
import pandas as pd
import pandas.testing as pdt

# Item under test
from dataCleaning import DataCleaning

@pytest.fixture
def config(): 
    return {
    "min_playlist_tracks": 2,
    "min_track_occurrences": 2,
    "max_duplicate_tracks": 1}

@pytest.fixture
def cleaner(config):
    return DataCleaning(config)

@pytest.mark.parametrize(
    "input_df, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 1, 2, 2, 2],
                    "track_uri": ["track1", "track1", "track2", "track3", "track3", "track4"],
                    "position": [0, 1, 2, 0, 1, 2],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 2, 0, 2],
                }
            ),
            id="duplicates_present",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1],
                }
            ),
            id="no_duplicates_nothing_dropped",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 1],
                    "track_uri": ["track1", "track1", "track1"],
                    "position": [0, 1, 2],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1],
                    "track_uri": ["track1"],
                    "position": [0],
                }
            ),
            id="all_duplicates_keep_first",
        ),
    ],
)

def test_remove_duplicate_tracks_within_playlists(cleaner, input_df, expected_df):
    
    result_df = cleaner._remove_duplicate_tracks_within_playlists(input_df).reset_index(drop=True)
    pdt.assert_frame_equal(expected_df, result_df)

@pytest.mark.parametrize(
    "input_df, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2, 3],
                    "track_uri": ["track1", "track2", "track3", "track4", "track5"],
                    "position": [0, 1, 0, 1, 0],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1],
                }
            ),             
            id = "playlists_with_few_tracks_removed",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1],
                }
            ),
            id = "all_playlists_meet_threshold_nothing_dropped",
        ),
    ]
)
    
def test_remove_playlists_with_few_tracks(cleaner, input_df, expected_df):
    
    result_df = cleaner._remove_playlists_with_few_tracks(input_df).reset_index(drop=True)
    pdt.assert_frame_equal(expected_df, result_df)

@pytest.mark.parametrize(
    "input_df, expected_df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2, 3],
                    "track_uri": ["track1", "track1", "track2", "track3", "track4"],
                    "position": [0, 1, 0, 1, 0],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1],
                    "track_uri": ["track1", "track1"],
                    "position": [0, 1],
                }
            ),
            id="tracks_with_few_occurrences_removed",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track1", "track2"],
                    "position": [0, 1, 0, 1],
                }
            ),
            pd.DataFrame(
                {
                    "playlist_id": [1, 1, 2, 2],
                    "track_uri": ["track1", "track2", "track1", "track2"],
                    "position": [0, 1, 0, 1],
                }
            ),
            id="all_tracks_meet_threshold_nothing_dropped",
        ),
    ]
)
def test_remove_tracks_with_few_occurrences(cleaner, input_df, expected_df):
    
    result_df = cleaner._remove_tracks_with_few_occurrences(input_df).reset_index(drop=True)
    pdt.assert_frame_equal(expected_df, result_df)
