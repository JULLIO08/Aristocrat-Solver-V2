"""Lexicon coverage signal: fraction of tokens that are real words."""
from __future__ import annotations


class Lexicon:
    def __init__(self, words: set[str]):
        self.words = {w.lower() for w in words}

    @classmethod
    def from_corpus(cls, lines: list[str], min_count: int = 1) -> "Lexicon":
        counts: dict[str, int] = {}
        for line in lines:
            for tok in line.lower().split():
                counts[tok] = counts.get(tok, 0) + 1
        return cls({w for w, c in counts.items() if c >= min_count})

    def score(self, text: str) -> float:
        toks = text.lower().split()
        if not toks:
            return 0.0
        known = sum(1 for t in toks if t in self.words)
        return known / len(toks)

    def to_list(self) -> list[str]:
        return sorted(self.words)
