"""Fuse the four scoring signals into one objective."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Weights:
    charlm: float = 1.0
    wordlm: float = 0.5
    lexicon: float = 2.0
    postag: float = 0.3

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Weights":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class Scorer:
    def __init__(self, charlm, wordlm, lexicon, postag, weights: Weights):
        self.charlm = charlm
        self.wordlm = wordlm
        self.lexicon = lexicon
        self.postag = postag
        self.w = weights
        self._cache: dict[str, float] = {}

    def score(self, text: str) -> float:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        val = (
            self.w.charlm * self.charlm.score(text)
            + self.w.wordlm * self.wordlm.score(text)
            + self.w.lexicon * self.lexicon.score(text)
            + self.w.postag * self.postag.score(text)
        )
        self._cache[text] = val
        return val

    def rerank(self, candidates: list[str]) -> list[tuple[int, float]]:
        """Score candidate decodes with the fused signals, best first.

        Returns (original_index, fused_score) pairs sorted by score descending.
        Char-LM scoring is batched into one forward pass when available.
        """
        missing = [t for t in dict.fromkeys(candidates) if t not in self._cache]
        if missing and hasattr(self.charlm, "score_batch"):
            char_scores = self.charlm.score_batch(missing)
            for text, cs in zip(missing, char_scores):
                self._cache[text] = (
                    self.w.charlm * float(cs)
                    + self.w.wordlm * self.wordlm.score(text)
                    + self.w.lexicon * self.lexicon.score(text)
                    + self.w.postag * self.postag.score(text)
                )
        scored = [(i, self.score(text)) for i, text in enumerate(candidates)]
        scored.sort(key=lambda t: -t[1])
        return scored
