"""Word-pattern dictionary attack: deterministic seed keys for short ciphers.

Classic aristocrat technique: a ciphertext word's repetition pattern
(PEOPLE -> ABCADB) drastically limits which dictionary words it can be.
Backtracking over the most constrained words yields partial keys that are
completed by frequency-rank fill and handed to Stage 1 as seed keys.
"""
from __future__ import annotations

import pickle
import time
from collections import Counter
from pathlib import Path

ENGLISH_FREQ = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def signature(word: str) -> str:
    """Repetition pattern of a word: first letter A, next new letter B, ..."""
    mapping: dict[str, str] = {}
    out = []
    for ch in word:
        if ch not in mapping:
            mapping[ch] = chr(ord("A") + len(mapping))
        out.append(mapping[ch])
    return "".join(out)


class PatternIndex:
    """Dictionary words grouped by signature, most frequent first."""

    def __init__(self, by_sig: dict[str, list[str]]):
        self.by_sig = by_sig

    @classmethod
    def from_corpus(cls, lines: list[str], min_count: int = 1) -> "PatternIndex":
        counts: Counter[str] = Counter()
        for line in lines:
            counts.update(line.upper().split())
        by_sig: dict[str, list[str]] = {}
        for word, c in counts.items():
            if c >= min_count:
                by_sig.setdefault(signature(word), []).append(word)
        for sig in by_sig:
            by_sig[sig].sort(key=lambda w: (-counts[w], w))
        return cls(by_sig)

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps(self.by_sig))

    @classmethod
    def load(cls, path) -> "PatternIndex":
        return cls(pickle.loads(Path(path).read_bytes()))

    def seed_keys(
        self,
        cipher_text: str,
        max_seeds: int = 200,
        max_words: int = 5,
        branch: int = 60,
        time_budget: float = 1.0,
    ) -> list[tuple[int, ...]]:
        """Backtrack candidate words onto the most constrained cipher words.

        Each consistent assignment yields a partial cipher->plain map, which
        is completed by frequency-rank fill into a full seed key. Unknown
        signatures are skipped; caps keep the search under time_budget.
        """
        words = []
        seen = set()
        for w in cipher_text.split():
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                cands = self.by_sig.get(signature(w))
                if cands:
                    words.append((w, cands[:branch]))
        # rarest signature first = strongest constraint
        words.sort(key=lambda t: len(t[1]))
        words = words[:max_words]
        if not words:
            return []

        deadline = time.perf_counter() + time_budget
        seeds: list[tuple[int, ...]] = []
        seen_maps: set[tuple] = set()
        c2p: dict[int, int] = {}   # cipher letter -> plain letter
        used: set[int] = set()     # plain letters already assigned

        def try_word(cw: str, pw: str) -> list[tuple[int, int]] | None:
            """Return new assignments if pw is consistent with c2p, else None."""
            new: dict[int, int] = {}
            for cc, pc in zip(cw, pw):
                c, p = ord(cc) - ord("A"), ord(pc) - ord("A")
                cur = c2p.get(c, new.get(c))
                if cur is not None:
                    if cur != p:
                        return None
                    continue
                if p in used or p in new.values():
                    return None
                new[c] = p
            return list(new.items())

        def emit() -> None:
            frozen = tuple(sorted(c2p.items()))
            if frozen in seen_maps:
                return
            seen_maps.add(frozen)
            seeds.append(self._complete(cipher_text, dict(c2p)))

        def dfs(depth: int) -> None:
            if len(seeds) >= max_seeds or time.perf_counter() > deadline:
                return
            if depth == len(words):
                emit()
                return
            cw, cands = words[depth]
            matched = False
            for pw in cands:
                if len(seeds) >= max_seeds or time.perf_counter() > deadline:
                    return
                new = try_word(cw, pw)
                if new is None:
                    continue
                matched = True
                for c, p in new:
                    c2p[c] = p
                    used.add(p)
                dfs(depth + 1)
                for c, p in new:
                    del c2p[c]
                    used.discard(p)
            if not matched:
                # no candidate fits this word under current constraints;
                # skip it rather than killing the whole branch
                dfs(depth + 1)

        dfs(0)
        return seeds

    @staticmethod
    def _complete(cipher_text: str, c2p: dict[int, int]) -> tuple[int, ...]:
        """Fill unassigned cipher letters with unused plain letters by rank."""
        counts = Counter(ord(c) - ord("A") for c in cipher_text if c != " ")
        unassigned = sorted((c for c in range(26) if c not in c2p),
                            key=lambda c: (-counts.get(c, 0), c))
        unused = [ord(ch) - ord("A") for ch in ENGLISH_FREQ
                  if (ord(ch) - ord("A")) not in c2p.values()]
        key = [0] * 26
        for c, p in c2p.items():
            key[c] = p
        for c, p in zip(unassigned, unused):
            key[c] = p
        return tuple(key)
