import numpy as np
import pytest

from utilsforecast.grouped_array import GroupedArray


@pytest.mark.parametrize("n_features", [None, 1, 2])
@pytest.mark.parametrize(
    "idx, rows", [(-1, [1, 9]), (-2, [0, 8]), (0, [0, 2]), (1, [1, 3])]
)
def test_take_from_groups_integer(n_features, idx, rows):
    if n_features is None:
        data = np.arange(10, dtype=np.float32)
    else:
        data = np.arange(10 * n_features, dtype=np.float32).reshape(-1, n_features)
    indptr = np.array([0, 2, 10])
    grouped = GroupedArray(data, indptr)

    result, result_indptr = grouped.take_from_groups(idx)

    np.testing.assert_array_equal(result, data[rows])
    np.testing.assert_array_equal(result_indptr, [0, 1, 2])
    assert result.dtype == data.dtype
    assert result_indptr.dtype == indptr.dtype


@pytest.mark.parametrize("n_features", [None, 2])
def test_take_from_groups_last_with_empty_groups(n_features):
    data = np.arange(5 if n_features is None else 10, dtype=np.int64)
    if n_features is not None:
        data = data.reshape(-1, n_features)
    grouped = GroupedArray(data, np.array([0, 0, 2, 2, 5]))

    result, result_indptr = grouped.take_from_groups(-1)

    np.testing.assert_array_equal(result, data[[1, 4]])
    np.testing.assert_array_equal(result_indptr, [0, 0, 1, 1, 2])
