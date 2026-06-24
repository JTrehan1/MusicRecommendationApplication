from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from data_splitting import DataSplitting, DataSplittingOutput


@pytest.fixture
def sample_ingestion_output():
    sparse_matrix = sp.csr_matrix(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(sparse_matrix=sparse_matrix)


def test_csr_splitter_returns_expected_output_type_and_shape(sample_ingestion_output):
    splitter = DataSplitting(sample_ingestion_output)

    result = splitter.csr_splitter(test_size=0.4, random_state=42)

    assert isinstance(result, DataSplittingOutput)
    assert result.train_data.shape == sample_ingestion_output.sparse_matrix.shape
    assert result.test_data.shape == sample_ingestion_output.sparse_matrix.shape
    assert sp.isspmatrix_csr(result.train_data)
    assert sp.isspmatrix_csr(result.test_data)


def test_csr_splitter_preserves_original_data_without_train_test_overlap(sample_ingestion_output):
    splitter = DataSplitting(sample_ingestion_output)

    result = splitter.csr_splitter(test_size=0.4, random_state=42)

    reconstructed = result.train_data + result.test_data
    overlap = result.train_data.multiply(result.test_data)

    np.testing.assert_array_equal(
        reconstructed.toarray(),
        sample_ingestion_output.sparse_matrix.toarray(),
    )
    assert overlap.nnz == 0
    assert result.train_data.nnz + result.test_data.nnz == sample_ingestion_output.sparse_matrix.nnz


def test_csr_splitter_holds_out_expected_number_of_tracks_per_eligible_playlist(sample_ingestion_output):
    splitter = DataSplitting(sample_ingestion_output)

    result = splitter.csr_splitter(test_size=0.4, random_state=42)

    assert result.test_data[0].nnz == 0
    assert result.test_data[1].nnz == 2
    assert result.test_data[2].nnz == 2
    assert result.train_data[0].nnz == 4
    assert result.train_data[1].nnz == 3
    assert result.train_data[2].nnz == 4


def test_csr_splitter_keeps_short_playlists_in_train_and_logs_warning(
    sample_ingestion_output,
    caplog,
):
    splitter = DataSplitting(sample_ingestion_output)

    with caplog.at_level("WARNING"):
        result = splitter.csr_splitter(test_size=0.4, random_state=42)

    np.testing.assert_array_equal(
        result.train_data[0].toarray(),
        sample_ingestion_output.sparse_matrix[0].toarray(),
    )
    assert result.test_data[0].nnz == 0
    assert "Playlist with less than 5 tracks found" in caplog.text


def test_csr_splitter_is_reproducible_for_same_random_state(sample_ingestion_output):
    splitter = DataSplitting(sample_ingestion_output)

    first_result = splitter.csr_splitter(test_size=0.4, random_state=42)
    second_result = splitter.csr_splitter(test_size=0.4, random_state=42)

    assert (first_result.train_data != second_result.train_data).nnz == 0
    assert (first_result.test_data != second_result.test_data).nnz == 0
