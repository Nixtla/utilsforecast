# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/grouped_array.ipynb.

# %% auto 0
__all__ = ['GroupedArray']

# %% ../nbs/grouped_array.ipynb 1
from typing import Sequence, Tuple, Union

import numpy as np

from .compat import DataFrame
from .processing import counts_by_id, value_cols_to_numpy

# %% ../nbs/grouped_array.ipynb 2
def _append_one(
    data: np.ndarray, indptr: np.ndarray, new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Append each value of new to each group in data formed by indptr."""
    n_groups = len(indptr) - 1
    rows = data.shape[0] + new.shape[0]
    new_data = np.empty((rows, data.shape[1]), dtype=data.dtype)
    new_indptr = indptr.copy()
    new_indptr[1:] += np.arange(1, n_groups + 1)
    for i in range(n_groups):
        prev_slice = slice(indptr[i], indptr[i + 1])
        new_slice = slice(new_indptr[i], new_indptr[i + 1] - 1)
        new_data[new_slice] = data[prev_slice]
        new_data[new_indptr[i + 1] - 1] = new[i]
    return new_data, new_indptr

# %% ../nbs/grouped_array.ipynb 4
def _append_several(
    data: np.ndarray,
    indptr: np.ndarray,
    new_sizes: np.ndarray,
    new_values: np.ndarray,
    new_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = data.shape[0] + new_values.shape[0]
    new_data = np.empty((rows, data.shape[1]), dtype=data.dtype)
    new_indptr = np.empty(new_sizes.size + 1, dtype=indptr.dtype)
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

# %% ../nbs/grouped_array.ipynb 6
class GroupedArray:
    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = data
        self.indptr = indptr
        self.n_groups = len(indptr) - 1

    def __len__(self):
        return self.n_groups

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx = self.n_groups + idx
        return self.data[self.indptr[idx] : self.indptr[idx + 1]]

    @classmethod
    def from_sorted_df(
        cls, df: DataFrame, id_col: str, time_col: str, target_col: str
    ) -> "GroupedArray":
        id_counts = counts_by_id(df, id_col)
        sizes = id_counts["counts"].to_numpy()
        indptr = np.append(0, sizes.cumsum())
        data = value_cols_to_numpy(df, id_col, time_col, target_col)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)
        return cls(data, indptr)

    def _take_from_ranges(self, ranges: Sequence) -> "GroupedArray":
        items = [self.data[r] for r in ranges]
        sizes = np.array([item.shape[0] for item in items])
        data = np.vstack(items)
        indptr = np.append(0, sizes.cumsum())
        return GroupedArray(data, indptr)

    def take(self, idxs: Sequence[int]) -> "GroupedArray":
        """Subset specific groups by their indices."""
        ranges = [range(self.indptr[i], self.indptr[i + 1]) for i in idxs]
        return self._take_from_ranges(ranges)

    def take_from_groups(self, idx: Union[int, slice]) -> "GroupedArray":
        """Select a subset from each group."""
        if isinstance(idx, int):
            # this preserves the 2d structure of data when indexing with the range
            idx = slice(idx, idx + 1)
        ranges = [
            range(self.indptr[i], self.indptr[i + 1])[idx] for i in range(self.n_groups)
        ]
        return self._take_from_ranges(ranges)

    def append(self, new: np.ndarray) -> "GroupedArray":
        """Appends each element of `new` to each existing group. Returns a copy."""
        if new.shape[0] != self.n_groups:
            raise ValueError(f"new must have {self.n_groups} rows.")
        new_data, new_indptr = _append_one(self.data, self.indptr, new)
        return GroupedArray(new_data, new_indptr)

    def append_several(
        self, new_sizes: np.ndarray, new_values: np.ndarray, new_groups: np.ndarray
    ) -> "GroupedArray":
        new_data, new_indptr = _append_several(
            self.data, self.indptr, new_sizes, new_values, new_groups
        )
        return GroupedArray(new_data, new_indptr)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_rows={self.data.shape[0]:,}, n_groups={self.n_groups:,})"
