__all__ = ["GroupedArray"]


from typing import Sequence, Tuple, Union

import numpy as np

from .compat import DataFrame
from .processing import _ranges_to_indexer, counts_by_id, value_cols_to_numpy


def _empty_like_rows(data: np.ndarray, n_rows: int) -> np.ndarray:
    if data.ndim == 2:
        return np.empty_like(data, shape=(n_rows, data.shape[1]))
    return np.empty_like(data, shape=n_rows)


def _append_one(
    data: np.ndarray, indptr: np.ndarray, new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Append each value of new to each group in data formed by indptr."""
    n_groups = len(indptr) - 1
    n_rows = data.shape[0] + new.shape[0]
    new_data = _empty_like_rows(data, n_rows)
    new_indptr = indptr.copy()
    new_indptr[1:] += np.arange(1, n_groups + 1)
    # every group grows by exactly one row, appended at its end (position
    # new_indptr[i + 1] - 1); scatter `new` there and the untouched positions
    # keep the original per-group order from `data`.
    insert_pos = new_indptr[1:] - 1
    keep_mask = np.ones(n_rows, dtype=bool)
    keep_mask[insert_pos] = False
    new_data[keep_mask] = data
    new_data[insert_pos] = (
        new.reshape(-1, 1) if data.ndim == 2 and new.ndim == 1 else new
    )
    return new_data, new_indptr


def _append_several(
    data: np.ndarray,
    indptr: np.ndarray,
    new_sizes: np.ndarray,
    new_values: np.ndarray,
    new_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_groups = new_sizes.size
    n_rows = data.shape[0] + new_values.shape[0]
    new_data = _empty_like_rows(data, n_rows)

    is_old = ~new_groups
    old_sizes = np.zeros(n_groups, dtype=np.int64)
    old_sizes[is_old] = np.diff(indptr)
    total_sizes = old_sizes + new_sizes
    new_indptr = np.empty_like(indptr, shape=n_groups + 1)
    new_indptr[0] = 0
    np.cumsum(total_sizes, out=new_indptr[1:])

    # existing rows keep their relative order at the start of each group
    old_starts = new_indptr[:-1][is_old]
    old_idx = _ranges_to_indexer(old_starts, old_starts + old_sizes[is_old])
    new_data[old_idx] = data
    # new rows are appended right after each group's (possibly empty) old data
    new_starts = new_indptr[:-1] + old_sizes
    new_idx = _ranges_to_indexer(new_starts, new_starts + new_sizes)
    new_data[new_idx] = (
        new_values.reshape(-1, 1)
        if data.ndim == 2 and new_values.ndim == 1
        else new_values
    )
    return new_data, new_indptr


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

    def _take_from_ranges(self, ranges: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        items = [self.data[r] for r in ranges]
        sizes = np.array([item.shape[0] for item in items])
        if self.data.ndim == 2:
            data = np.vstack(items)
        else:
            data = np.hstack(items)
        indptr = np.append(0, sizes.cumsum())
        return data, indptr

    def take(self, idxs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Subset specific groups by their indices."""
        ranges = [range(self.indptr[i], self.indptr[i + 1]) for i in idxs]
        return self._take_from_ranges(ranges)

    def take_from_groups(self, idx: Union[int, slice]) -> Tuple[np.ndarray, np.ndarray]:
        """Select a subset from each group."""
        if isinstance(idx, int):
            # this preserves the 2d structure of data when indexing with the range
            idx = slice(idx, idx + 1)
        ranges = [
            range(self.indptr[i], self.indptr[i + 1])[idx] for i in range(self.n_groups)
        ]
        return self._take_from_ranges(ranges)

    def append(self, new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Appends each element of `new` to each existing group. Returns a copy."""
        if new.shape[0] != self.n_groups:
            raise ValueError(f"new must have {self.n_groups} rows.")
        return _append_one(self.data, self.indptr, new)

    def append_several(
        self, new_sizes: np.ndarray, new_values: np.ndarray, new_groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _append_several(
            self.data, self.indptr, new_sizes, new_values, new_groups
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(n_rows={self.data.shape[0]:,}, n_groups={self.n_groups:,})"
