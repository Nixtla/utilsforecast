# test _append_one
import numpy as np
from conftest import assert_raises_with_message

from utilsforecast.data import generate_series
from utilsforecast.grouped_array import GroupedArray, _append_one, _append_several


def test_append_one():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new = np.array([7, 8])
    new_data, new_indptr = _append_one(data, indptr, new)
    np.testing.assert_equal(new_data, np.array([0, 1, 7, 2, 3, 4, 8]))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 3, 7]),
    )

# 2d
def test_append_one_2d():
    data = np.arange(5).reshape(-1, 1)
    indptr = np.array([0, 2, 5])
    new = np.array([7, 8])
    new_data, new_indptr = _append_one(data, indptr, new)
    np.testing.assert_equal(new_data, np.array([0, 1, 7, 2, 3, 4, 8]).reshape(-1, 1))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 3, 7]),
    )

# test append several
def test_append_several():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5])
    new_groups = np.array([False, True, False])
    new_data, new_indptr = _append_several(data, indptr, new_sizes, new_values, new_groups)
    np.testing.assert_equal(new_data, np.array([0, 1, 6, 7, 2, 3, 4, 5]))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 2, 4, 8]),
    )

# 2d
def test_append_several_2d():
    data = np.arange(5).reshape(-1, 1)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5]).reshape(-1, 1)
    new_groups = np.array([False, True, False])
    new_data, new_indptr = _append_several(data, indptr, new_sizes, new_values, new_groups)
    np.testing.assert_equal(new_data, np.array([0, 1, 6, 7, 2, 3, 4, 5]).reshape(-1, 1))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 2, 4, 8]),
    )

# 2d data, 1d new_values: exercises the `data.ndim == 2 and new_values.ndim
# == 1` reshape branch in `_append_several` (as opposed to `test_append_
# several_2d` above, where `new_values` is already 2D and that branch is
# never taken).
def test_append_several_2d_data_1d_new_values():
    data = np.arange(5).reshape(-1, 1)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5])  # 1D, must be reshaped to match `data`
    new_groups = np.array([False, True, False])
    new_data, new_indptr = _append_several(data, indptr, new_sizes, new_values, new_groups)
    np.testing.assert_equal(new_data, np.array([0, 1, 6, 7, 2, 3, 4, 5]).reshape(-1, 1))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 2, 4, 8]),
    )


# The `GroupedArray` is used internally for storing the series values and performing transformations.
def test_grouped_array():
    data = np.arange(20, dtype=np.float32).reshape(-1, 2)
    indptr = np.array([0, 2, 10])  # group 1: [0, 1], group 2: [2..9]
    ga = GroupedArray(data, indptr)
    assert len(ga) == 2
    # Iterate through the groups
    ga_iter = iter(ga)
    np.testing.assert_equal(next(ga_iter), np.arange(4).reshape(-1, 2))
    np.testing.assert_equal(next(ga_iter), np.arange(4, 20).reshape(-1, 2))
    # Take the last two observations from each group
    last2_data, last2_indptr = ga.take_from_groups(slice(-2, None))
    np.testing.assert_equal(
        last2_data,
        np.vstack(
            [
                np.arange(4).reshape(-1, 2),
                np.arange(16, 20).reshape(-1, 2),
            ]
        ),
    )
    np.testing.assert_equal(last2_indptr, np.array([0, 2, 4]))

# 1d
def test_grouped_array_1d():
    data = np.arange(20, dtype=np.float32).reshape(-1, 2)
    indptr = np.array([0, 2, 10])
    ga = GroupedArray(data, indptr)
    ga1d = GroupedArray(np.arange(10), indptr)
    last2_data1d, last2_indptr1d = ga1d.take_from_groups(slice(-2, None))
    np.testing.assert_equal(last2_data1d, np.array([0, 1, 8, 9]))
    np.testing.assert_equal(last2_indptr1d, np.array([0, 2, 4]))
    # Take the second observation from each group
    second_data, second_indptr = ga.take_from_groups(1)
    np.testing.assert_equal(second_data, np.array([[2, 3], [6, 7]]))
    np.testing.assert_equal(second_indptr, np.array([0, 1, 2]))

    # 1d
    second_data1d, second_indptr1d = ga1d.take_from_groups(1)
    np.testing.assert_equal(second_data1d, np.array([1, 3]))
    np.testing.assert_equal(second_indptr1d, np.array([0, 1, 2]))
    # Take the last four observations from every group. Note that since group 1 only has two elements, only these are returned.
    last4_data, last4_indptr = ga.take_from_groups(slice(-4, None))
    np.testing.assert_equal(
        last4_data,
        np.vstack(
            [
                np.arange(4).reshape(-1, 2),
                np.arange(12, 20).reshape(-1, 2),
            ]
        ),
    )
    np.testing.assert_equal(last4_indptr, np.array([0, 2, 6]))

    # 1d
    last4_data1d, last4_indptr1d = ga1d.take_from_groups(slice(-4, None))
    np.testing.assert_equal(last4_data1d, np.array([0, 1, 6, 7, 8, 9]))
    np.testing.assert_equal(last4_indptr1d, np.array([0, 2, 6]))
    # Select a specific subset of groups
    indptr = np.array([0, 2, 4, 7, 10])
    ga2 = GroupedArray(data, indptr)
    subset = GroupedArray(*ga2.take([0, 2]))
    np.testing.assert_allclose(subset[0].data, ga2[0].data)
    np.testing.assert_allclose(subset[1].data, ga2[2].data)

    # 1d
    ga2_1d = GroupedArray(np.arange(10), indptr)
    subset1d = GroupedArray(*ga2_1d.take([0, 2]))
    np.testing.assert_allclose(subset1d[0].data, ga2_1d[0].data)
    np.testing.assert_allclose(subset1d[1].data, ga2_1d[2].data)
    # try to append new values that don't match the number of groups
    assert_raises_with_message(lambda: ga.append(np.array([1.0, 2.0, 3.0])), "new must have 2 rows")
    # build from df
    series_pd = generate_series(10, static_as_categorical=False, engine="pandas")
    ga_pd = GroupedArray.from_sorted_df(series_pd, "unique_id", "ds", "y")
    series_pl = generate_series(10, static_as_categorical=False, engine="polars")
    ga_pl = GroupedArray.from_sorted_df(series_pl, "unique_id", "ds", "y")
    np.testing.assert_allclose(ga_pd.data, ga_pl.data)
    np.testing.assert_equal(ga_pd.indptr, ga_pl.indptr)


# ========================================
# _append_one / _append_several: vectorized scatter (perf)
# ========================================
#
# Both used a python loop over every group to splice new rows in. They're
# now vectorized scatter-writes (see `_ranges_to_indexer` reuse in
# grouped_array.py). These tests pin the vectorized versions against the
# literal original per-group loop on adversarial inputs: empty groups, all
# groups new/old, 1D and 2D data, and varying dtypes.


def _reference_append_one(data, indptr, new):
    n_groups = len(indptr) - 1
    n_rows = data.shape[0] + new.shape[0]
    if data.ndim == 2:
        new_data = np.empty_like(data, shape=(n_rows, data.shape[1]))
    else:
        new_data = np.empty_like(data, shape=n_rows)
    new_indptr = indptr.copy()
    new_indptr[1:] += np.arange(1, n_groups + 1)
    for i in range(n_groups):
        prev_slice = slice(indptr[i], indptr[i + 1])
        new_slice = slice(new_indptr[i], new_indptr[i + 1] - 1)
        new_data[new_slice] = data[prev_slice]
        new_data[new_indptr[i + 1] - 1] = new[i]
    return new_data, new_indptr


def _reference_append_several(data, indptr, new_sizes, new_values, new_groups):
    n_rows = data.shape[0] + new_values.shape[0]
    if data.ndim == 2:
        new_data = np.empty_like(data, shape=(n_rows, data.shape[1]))
    else:
        new_data = np.empty_like(data, shape=n_rows)
    new_indptr = np.empty_like(indptr, shape=new_sizes.size + 1)
    new_indptr[0] = 0
    old_indptr_idx = 0
    new_vals_idx = 0
    for i, is_new in enumerate(new_groups):
        new_size = new_sizes[i]
        if is_new:
            old_size = 0
        else:
            prev_slice = slice(indptr[old_indptr_idx], indptr[old_indptr_idx + 1])
            old_indptr_idx += 1
            old_size = prev_slice.stop - prev_slice.start
            new_size += old_size
            new_data[new_indptr[i] : new_indptr[i] + old_size] = data[prev_slice]
        new_indptr[i + 1] = new_indptr[i] + new_size
        new_data[new_indptr[i] + old_size : new_indptr[i + 1]] = new_values[
            new_vals_idx : new_vals_idx + new_sizes[i]
        ]
        new_vals_idx += new_sizes[i]
    return new_data, new_indptr


class TestAppendVectorizedMatchesReferenceLoop:
    def test_append_one_adversarial(self):
        rng = np.random.default_rng(0)
        for trial in range(100):
            n_groups = rng.integers(1, 30)
            sizes = rng.integers(0, 10, size=n_groups)  # allow empty groups
            indptr = np.append(0, sizes.cumsum()).astype(np.int64)
            total = int(sizes.sum())
            if rng.random() < 0.5:
                data = rng.normal(size=total)
                new = rng.normal(size=n_groups)
            else:
                ncols = rng.integers(1, 4)
                data = rng.normal(size=(total, ncols))
                new = rng.normal(size=(n_groups, ncols))
            actual_data, actual_indptr = _append_one(data, indptr, new)
            expected_data, expected_indptr = _reference_append_one(data, indptr, new)
            np.testing.assert_array_equal(
                actual_indptr, expected_indptr, err_msg=f"trial {trial}"
            )
            np.testing.assert_allclose(
                actual_data, expected_data, err_msg=f"trial {trial}"
            )

    def test_append_several_adversarial(self):
        rng = np.random.default_rng(1)
        for trial in range(100):
            n_groups = rng.integers(1, 30)
            new_groups = rng.random(n_groups) < 0.3
            n_old_groups = int((~new_groups).sum())
            old_sizes = rng.integers(0, 8, size=n_old_groups)
            indptr = np.append(0, old_sizes.cumsum()).astype(np.int64)
            total_old = int(old_sizes.sum())
            new_sizes = rng.integers(0, 5, size=n_groups)
            total_new = int(new_sizes.sum())
            if rng.random() < 0.5:
                data = rng.normal(size=total_old)
                new_values = rng.normal(size=total_new)
            else:
                ncols = rng.integers(1, 4)
                data = rng.normal(size=(total_old, ncols))
                new_values = rng.normal(size=(total_new, ncols))
            actual_data, actual_indptr = _append_several(
                data, indptr, new_sizes, new_values, new_groups
            )
            expected_data, expected_indptr = _reference_append_several(
                data, indptr, new_sizes, new_values, new_groups
            )
            np.testing.assert_array_equal(
                actual_indptr, expected_indptr, err_msg=f"trial {trial}"
            )
            np.testing.assert_allclose(
                actual_data, expected_data, err_msg=f"trial {trial}"
            )

    def test_append_several_all_new_groups(self):
        # no pre-existing groups at all (e.g. GroupedArray starting empty)
        data = np.array([])
        indptr = np.array([0])
        new_sizes = np.array([2, 0, 3])
        new_values = np.array([1.0, 2.0, 5.0, 6.0, 7.0])
        new_groups = np.array([True, True, True])
        actual_data, actual_indptr = _append_several(
            data, indptr, new_sizes, new_values, new_groups
        )
        expected_data, expected_indptr = _reference_append_several(
            data, indptr, new_sizes, new_values, new_groups
        )
        np.testing.assert_array_equal(actual_indptr, expected_indptr)
        np.testing.assert_allclose(actual_data, expected_data)

    def test_append_several_all_old_groups_no_new_values(self):
        data = np.arange(6.0)
        indptr = np.array([0, 2, 4, 6])
        new_sizes = np.array([0, 0, 0])
        new_values = np.array([])
        new_groups = np.array([False, False, False])
        actual_data, actual_indptr = _append_several(
            data, indptr, new_sizes, new_values, new_groups
        )
        expected_data, expected_indptr = _reference_append_several(
            data, indptr, new_sizes, new_values, new_groups
        )
        np.testing.assert_array_equal(actual_indptr, expected_indptr)
        np.testing.assert_allclose(actual_data, expected_data)
