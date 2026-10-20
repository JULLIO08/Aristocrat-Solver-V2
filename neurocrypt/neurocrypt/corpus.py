"""Corpus loading and deterministic train/eval splitting."""
from __future__ import annotations
import random
from pathlib import Path

from .cipher import clean_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PLAIN = DATA_DIR / "quotes.txt"
DEFAULT_ENCODED = DATA_DIR / "quotesEncoded.txt"


def load_lines(path: Path) -> list[str]:
    """Read a text file, clean each line, drop blanks."""
    out = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        cleaned = clean_text(raw)
        if cleaned:
            out.append(cleaned)
    return out


def train_eval_split(
    lines: list[str], eval_frac: float = 0.05, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Deterministic split by shuffled index."""
    eval_idx = set(eval_split_indices(len(lines), eval_frac, seed))
    train = [lines[i] for i in range(len(lines)) if i not in eval_idx]
    evalset = [lines[i] for i in range(len(lines)) if i in eval_idx]
    return train, evalset


def eval_split_indices(
    n_lines: int, eval_frac: float = 0.05, seed: int = 42
) -> list[int]:
    """Line indices belonging to the eval split, sorted ascending."""
    idx = list(range(n_lines))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(n_lines * eval_frac))
    return sorted(idx[:n_eval])
