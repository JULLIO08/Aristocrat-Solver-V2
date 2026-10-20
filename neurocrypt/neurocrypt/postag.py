"""POS tag-trigram plausibility signal using nltk's perceptron tagger."""
from __future__ import annotations
import json
import math
from pathlib import Path

import nltk

BOS, EOS = "<s>", "</s>"


def _ensure_nltk() -> None:
    for pkg, path in [
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _tags(text: str) -> list[str]:
    toks = text.lower().split()
    if not toks:
        return []
    try:
        _ensure_nltk()
        return [t for _, t in nltk.pos_tag(toks)]
    except LookupError:
        # offline and resources missing: degrade to a neutral signal
        # rather than crashing the solve
        return []


class POSModel:
    def __init__(self, tri: dict, bi: dict, tagset: set[str], k: float):
        self.tri = tri
        self.bi = bi
        self.tagset = tagset
        self.k = k
        self.v = len(tagset) + 2  # + boundary markers

    @classmethod
    def train(cls, lines: list[str], k: float = 0.1) -> "POSModel":
        tri: dict[str, int] = {}
        bi: dict[str, int] = {}
        tagset: set[str] = set()
        for line in lines:
            tags = [BOS, BOS] + _tags(line) + [EOS]
            for t in tags:
                tagset.add(t)
            for i in range(len(tags) - 2):
                tri["\t".join(tags[i:i + 3])] = tri.get("\t".join(tags[i:i + 3]), 0) + 1
                bi["\t".join(tags[i:i + 2])] = bi.get("\t".join(tags[i:i + 2]), 0) + 1
        return cls(tri, bi, tagset, k)

    def score(self, text: str) -> float:
        tags = [BOS, BOS] + _tags(text) + [EOS]
        if len(tags) <= 3:
            return 0.0
        total, n = 0.0, 0
        for i in range(len(tags) - 2):
            num = self.tri.get("\t".join(tags[i:i + 3]), 0) + self.k
            den = self.bi.get("\t".join(tags[i:i + 2]), 0) + self.k * self.v
            total += math.log(num / den)
            n += 1
        return total / n

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "tri": self.tri, "bi": self.bi,
            "tagset": sorted(self.tagset), "k": self.k,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "POSModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["tri"], d["bi"], set(d["tagset"]), d["k"])
