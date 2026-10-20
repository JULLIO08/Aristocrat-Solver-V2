"""Word trigram language model with add-k smoothing."""
from __future__ import annotations
import json
import math
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class WordLM:
    def __init__(self, tri: dict, bi: dict, vocab: set[str], k: float):
        self.tri = tri          # "w1\tw2\tw3" -> count
        self.bi = bi            # "w1\tw2" -> count
        self.vocab = vocab
        self.k = k
        self.v = len(vocab) + 1  # +1 for UNK

    @classmethod
    def train(cls, lines: list[str], k: float = 0.1, min_count: int = 1) -> "WordLM":
        counts: dict[str, int] = {}
        for line in lines:
            for tok in line.lower().split():
                counts[tok] = counts.get(tok, 0) + 1
        vocab = {w for w, c in counts.items() if c >= min_count}
        tri: dict[str, int] = {}
        bi: dict[str, int] = {}
        for line in lines:
            toks = [BOS, BOS] + [t if t in vocab else UNK for t in line.lower().split()] + [EOS]
            for i in range(len(toks) - 2):
                w1, w2, w3 = toks[i], toks[i + 1], toks[i + 2]
                tri["\t".join((w1, w2, w3))] = tri.get("\t".join((w1, w2, w3)), 0) + 1
                bi["\t".join((w1, w2))] = bi.get("\t".join((w1, w2)), 0) + 1
        return cls(tri, bi, vocab, k)

    def _logp(self, w1: str, w2: str, w3: str) -> float:
        num = self.tri.get("\t".join((w1, w2, w3)), 0) + self.k
        den = self.bi.get("\t".join((w1, w2)), 0) + self.k * self.v
        return math.log(num / den)

    def score(self, text: str) -> float:
        toks = [BOS, BOS] + [
            t if t in self.vocab else UNK for t in text.lower().split()
        ] + [EOS]
        if len(toks) <= 3:
            return 0.0
        total = 0.0
        n = 0
        for i in range(len(toks) - 2):
            total += self._logp(toks[i], toks[i + 1], toks[i + 2])
            n += 1
        return total / n

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "tri": self.tri, "bi": self.bi,
            "vocab": sorted(self.vocab), "k": self.k,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "WordLM":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["tri"], d["bi"], set(d["vocab"]), d["k"])
