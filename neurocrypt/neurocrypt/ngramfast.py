"""Fast character-quadgram scorer with incremental delta scoring.

Pure numpy — no torch. Space is treated as a 27th symbol because word
boundaries carry signal in aristocrats. Scores are mean log-prob per
quadgram, so they are length-independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_SYMBOLS = 27  # A-Z plus space
SPACE = 26
TABLE_SIZE = N_SYMBOLS ** 4
_POW = np.array([N_SYMBOLS ** 3, N_SYMBOLS ** 2, N_SYMBOLS, 1], dtype=np.int64)


def _to_symbols(text: str) -> np.ndarray:
    """Map cleaned text (A-Z and spaces) to symbol ids 0-26."""
    if not text:
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64)
    return np.where(arr == ord(" "), SPACE, arr - ord("A"))


def _window_indices(syms: np.ndarray) -> np.ndarray:
    """Quadgram table index for every window of 4 consecutive symbols."""
    n = len(syms)
    if n < 4:
        return np.empty(0, dtype=np.int64)
    return (
        syms[: n - 3] * _POW[0]
        + syms[1: n - 2] * _POW[1]
        + syms[2: n - 1] * _POW[2]
        + syms[3:] * _POW[3]
    )


class QuadgramModel:
    """Add-k smoothed log-prob table over character quadgrams."""

    def __init__(self, table: np.ndarray):
        if table.shape != (TABLE_SIZE,):
            raise ValueError(f"expected table shape ({TABLE_SIZE},), got {table.shape}")
        self.table = table.astype(np.float32)

    @classmethod
    def train(cls, lines: list[str], k: float = 0.5) -> "QuadgramModel":
        counts = np.zeros(TABLE_SIZE, dtype=np.float64)
        for line in lines:
            idx = _window_indices(_to_symbols(line))
            if len(idx):
                np.add.at(counts, idx, 1.0)
        total = counts.sum() + k * TABLE_SIZE
        table = np.log((counts + k) / total)
        return cls(table.astype(np.float32))

    def score(self, text: str) -> float:
        """Mean log-prob per quadgram; 0.0 for texts shorter than one window."""
        idx = _window_indices(_to_symbols(text))
        if len(idx) == 0:
            return 0.0
        return float(self.table[idx].mean())

    def save(self, path) -> None:
        np.save(Path(path), self.table)

    @classmethod
    def load(cls, path) -> "QuadgramModel":
        return cls(np.load(Path(path)))


class DeltaScorer:
    """Incremental quadgram scoring of decodes of one fixed ciphertext.

    A key swap of cipher letters (i, j) only changes quadgram windows that
    overlap an occurrence of i or j, so a swap is scored by rescoring just
    those windows instead of the whole text.
    """

    def __init__(self, cipher_text: str, model: QuadgramModel):
        self.table = model.table
        syms = _to_symbols(cipher_text)
        n = len(syms)
        self.n_windows = max(0, n - 3)
        if self.n_windows:
            # (W, 4) cipher symbols of each window, for vectorized remapping
            self._winsym = np.stack(
                [syms[k: k + self.n_windows] for k in range(4)], axis=1
            )
        else:
            self._winsym = np.empty((0, 4), dtype=np.int64)
        # windows affected by each cipher letter
        self._wins: list[np.ndarray] = []
        for c in range(26):
            pos = np.flatnonzero(syms == c)
            if len(pos) == 0 or self.n_windows == 0:
                self._wins.append(np.empty(0, dtype=np.int64))
                continue
            spans = [
                np.arange(max(0, p - 3), min(p, self.n_windows - 1) + 1)
                for p in pos
            ]
            self._wins.append(np.unique(np.concatenate(spans)))
        # per-pair cache of (windows, weight-diff) — at most 325 entries.
        # For a window with symbols s0..s3 and place values p0..p3, let
        # wi = sum of p_k where s_k == i (and wj likewise for j). Swapping the
        # mappings of i and j changes the window's table index by exactly
        # (key[j] - key[i]) * (wi - wj), so we precompute wdiff = wi - wj.
        self._pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._keyext = np.arange(N_SYMBOLS, dtype=np.int64)
        # current-key quadgram index and table value of every window
        self._cur_idx = np.zeros(self.n_windows, dtype=np.int64)
        self._cur_vals = np.zeros(self.n_windows, dtype=np.float32)
        self.current_score = 0.0
        self._last: tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None

    def _pair_data(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = (i, j) if i < j else (j, i)
        cached = self._pair_cache.get((a, b))
        if cached is None:
            wins = np.union1d(self._wins[a], self._wins[b])
            sub = self._winsym[wins]
            wa = ((sub == a) * _POW).sum(axis=1)
            wb = ((sub == b) * _POW).sum(axis=1)
            cached = (wins, wa - wb)
            self._pair_cache[(a, b)] = cached
        return cached

    def set_key(self, key: tuple[int, ...]) -> float:
        self._keyext[:26] = key
        self._keyext[26] = SPACE
        if self.n_windows:
            self._cur_idx = self._keyext.take(self._winsym) @ _POW
            self._cur_vals = self.table.take(self._cur_idx)
            self.current_score = float(self._cur_vals.mean())
        else:
            self.current_score = 0.0
        self._last = None
        return self.current_score

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._keyext[:26])

    def delta(self, i: int, j: int) -> float:
        """Score change from swapping the mappings of cipher letters i and j."""
        empty = np.empty(0, dtype=np.int64)
        if self.n_windows == 0 or i == j:
            self._last = (i, j, 0.0, empty, empty, empty)
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        wins, wdiff = self._pair_data(a, b)
        if len(wins) == 0:
            self._last = (i, j, 0.0, empty, empty, empty)
            return 0.0
        new_idx = self._cur_idx.take(wins) + int(self._keyext[b] - self._keyext[a]) * wdiff
        new_vals = self.table.take(new_idx)
        d = float((new_vals.sum() - self._cur_vals.take(wins).sum()) / self.n_windows)
        self._last = (i, j, d, wins, new_idx, new_vals)
        return d

    def apply(self, i: int, j: int) -> float:
        """Commit the swap of cipher letters i and j; returns new current score."""
        if self._last is None or self._last[:2] != (i, j):
            self.delta(i, j)
        _, _, d, wins, new_idx, new_vals = self._last
        if len(wins):
            self._cur_idx[wins] = new_idx
            self._cur_vals[wins] = new_vals
        self._keyext[i], self._keyext[j] = self._keyext[j], self._keyext[i]
        self.current_score += d
        self._last = None
        return self.current_score


class BatchDeltaScorer:
    """Delta scoring of R parallel key chains over one fixed ciphertext.

    Amortizes numpy call overhead across chains: one vectorized call scores
    one proposed swap in every chain, so per-swap cost drops with n_chains.
    Used by the fast Stage-1 search to run parallel SA restarts.
    """

    def __init__(self, cipher_text: str, model: QuadgramModel, n_chains: int):
        self.table = model.table
        self.n_chains = n_chains
        # share the per-cipher precomputation with a scalar scorer
        self._ds = DeltaScorer(cipher_text, model)
        self.n_windows = self._ds.n_windows
        self._winsym = self._ds._winsym
        self._keys = np.tile(np.arange(N_SYMBOLS, dtype=np.int64), (n_chains, 1))
        self._cur_idx = np.zeros((n_chains, self.n_windows), dtype=np.int64)
        self._cur_vals = np.zeros((n_chains, self.n_windows), dtype=np.float32)
        self.current_scores = np.zeros(n_chains, dtype=np.float64)
        self._last: tuple | None = None

    def key(self, r: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self._keys[r, :26])

    def set_keys(self, keys: list[tuple[int, ...]]) -> np.ndarray:
        for r, key in enumerate(keys):
            self._keys[r, :26] = key
            self._keys[r, 26] = SPACE
        if self.n_windows:
            for r in range(self.n_chains):
                self._cur_idx[r] = self._keys[r].take(self._winsym) @ _POW
            self._cur_vals = self.table.take(self._cur_idx)
            self.current_scores = self._cur_vals.mean(axis=1).astype(np.float64)
        else:
            self.current_scores = np.zeros(self.n_chains, dtype=np.float64)
        self._last = None
        return self.current_scores.copy()

    def set_key(self, r: int, key: tuple[int, ...]) -> float:
        """Reset a single chain (used when a restart begins)."""
        self._keys[r, :26] = key
        self._keys[r, 26] = SPACE
        if self.n_windows:
            self._cur_idx[r] = self._keys[r].take(self._winsym) @ _POW
            self._cur_vals[r] = self.table.take(self._cur_idx[r])
            self.current_scores[r] = float(self._cur_vals[r].mean())
        else:
            self.current_scores[r] = 0.0
        self._last = None
        return self.current_scores[r]

    def delta(self, pairs: np.ndarray) -> np.ndarray:
        """Score change per chain for swapping pairs[r] = (i, j) in chain r."""
        R = self.n_chains
        if self.n_windows == 0:
            d = np.zeros(R)
            self._last = (pairs, None, None, None, None, d)
            return d
        wins_parts, wd_parts, counts = [], [], np.empty(R, dtype=np.int64)
        for r in range(R):
            i, j = int(pairs[r, 0]), int(pairs[r, 1])
            a, b = (i, j) if i < j else (j, i)
            wins, wdiff = self._ds._pair_data(a, b)
            wins_parts.append(wins)
            wd_parts.append(wdiff)
            counts[r] = len(wins)
        flat_wins = np.concatenate(wins_parts)
        flat_wd = np.concatenate(wd_parts)
        rows = np.repeat(np.arange(R), counts)
        a_idx = np.minimum(pairs[:, 0], pairs[:, 1])
        b_idx = np.maximum(pairs[:, 0], pairs[:, 1])
        diff = (self._keys[np.arange(R), b_idx] - self._keys[np.arange(R), a_idx])
        new_idx = self._cur_idx[rows, flat_wins] + diff[rows] * flat_wd
        new_vals = self.table.take(new_idx)
        dvals = new_vals.astype(np.float64) - self._cur_vals[rows, flat_wins]
        d = np.bincount(rows, weights=dvals, minlength=R) / self.n_windows
        self._last = (pairs, rows, flat_wins, new_idx, new_vals, d)
        return d

    def accept(self, mask: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """Commit the last proposed swaps for chains where mask is True."""
        if self._last is None or self._last[0] is not pairs:
            self.delta(pairs)
        _, rows, flat_wins, new_idx, new_vals, d = self._last
        if rows is not None and mask.any():
            sel = mask[rows]
            self._cur_idx[rows[sel], flat_wins[sel]] = new_idx[sel]
            self._cur_vals[rows[sel], flat_wins[sel]] = new_vals[sel]
        for r in np.flatnonzero(mask):
            i, j = int(pairs[r, 0]), int(pairs[r, 1])
            self._keys[r, i], self._keys[r, j] = self._keys[r, j], self._keys[r, i]
        self.current_scores = self.current_scores + d * mask
        self._last = None
        return self.current_scores
