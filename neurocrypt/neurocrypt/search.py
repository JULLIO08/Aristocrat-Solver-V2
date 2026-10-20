"""Simulated annealing over substitution keys."""
from __future__ import annotations
import math
import random
from collections import Counter
from typing import Callable

import numpy as np

from .cipher import decode, random_key, swap
from .ngramfast import QuadgramModel, BatchDeltaScorer

# English letters by corpus frequency, most common first
ENGLISH_FREQ = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def frequency_seed_key(cipher_text: str) -> tuple[int, ...]:
    """Map cipher letter frequency ranks onto English letter frequency ranks.

    The most frequent cipher letter decodes to E, the next to T, and so on.
    Cipher letters that never occur are filled with the remaining English
    letters in frequency order, so the result is always a permutation.
    """
    counts = Counter(ord(c) - ord("A") for c in cipher_text if c != " ")
    order = sorted(range(26), key=lambda c: (-counts.get(c, 0), c))
    key = [0] * 26
    for rank, c in enumerate(order):
        key[c] = ord(ENGLISH_FREQ[rank]) - ord("A")
    return tuple(key)


def seed_variants(cipher_text: str) -> list[tuple[int, ...]]:
    """Frequency seed plus A/I-locked variants for single-letter cipher words."""
    base = frequency_seed_key(cipher_text)
    seeds = [base]
    singles = [w for w in cipher_text.split() if len(w) == 1]
    if singles:
        c = ord(Counter(singles).most_common(1)[0][0]) - ord("A")
        for target in "AI":
            t = ord(target) - ord("A")
            k = list(base)
            holder = k.index(t)  # cipher letter currently decoding to target
            k[holder], k[c] = k[c], t
            seeds.append(tuple(k))
    return seeds


def anneal(
    cipher_text: str,
    score_fn: Callable[[str], float],
    restarts: int = 10,
    iters: int = 4000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int = 0,
) -> tuple[tuple[int, ...], float]:
    """Return (best_key, best_score) maximizing score_fn(decode(cipher, key))."""
    rng = random.Random(seed)
    cooling = (t_end / t_start) ** (1.0 / max(1, iters))
    best_key = random_key(rng)
    best_score = score_fn(decode(cipher_text, best_key))

    for _ in range(restarts):
        cur = random_key(rng)
        cur_score = score_fn(decode(cipher_text, cur))
        temp = t_start
        for _ in range(iters):
            i, j = rng.randrange(26), rng.randrange(26)
            if i == j:
                temp *= cooling
                continue
            cand = swap(cur, i, j)
            cand_score = score_fn(decode(cipher_text, cand))
            delta = cand_score - cur_score
            if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
                cur, cur_score = cand, cand_score
                if cur_score > best_score:
                    best_key, best_score = cur, cur_score
            temp *= cooling
    return best_key, best_score


def search_topk(
    cipher_text: str,
    qmodel: QuadgramModel,
    restarts: int = 16,
    iters: int = 4000,
    k: int = 16,
    seed: int = 0,
    seed_keys: list[tuple[int, ...]] | None = None,
    t_start: float = 1.0,
    t_end: float = 0.01,
    freq_seed: bool = True,
    bias_prob: float = 0.7,
    reheat_factor: float = 3.0,
) -> list[tuple[tuple[int, ...], float]]:
    """Fast Stage-1 search: parallel SA chains over a quadgram model.

    Runs `restarts` chains simultaneously with vectorized delta scoring and
    returns up to k best keys with distinct decodes, best first.

    seed_keys start the first chains (e.g. pattern-attack candidates). When
    freq_seed is on, the next chains start from the frequency-rank key and
    its A/I single-letter-word variants; the rest are random (diversity).
    With probability bias_prob a proposed swap involves the cipher letter with
    the worst local quadgram contribution for that chain. Chains that stall
    are reheated (temperature *= reheat_factor, capped at t_start).
    """
    rng = random.Random(seed)
    n_chains = max(1, restarts)
    keys: list[tuple[int, ...]] = []
    if seed_keys:
        keys.extend(tuple(s) for s in seed_keys[:n_chains])
    if freq_seed and len(keys) < n_chains:
        for s in seed_variants(cipher_text):
            if len(keys) < n_chains:
                keys.append(s)
    while len(keys) < n_chains:
        keys.append(random_key(rng))

    bds = BatchDeltaScorer(cipher_text, qmodel, n_chains)
    scores = bds.set_keys(keys)

    # two keys decode identically iff they agree on cipher letters present
    present = sorted({ord(c) - ord("A") for c in cipher_text if c != " "})

    def canon(key: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(key[c] for c in present)

    archive: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}

    def record(key: tuple[int, ...], score: float) -> None:
        c = canon(key)
        prev = archive.get(c)
        if prev is None or score > prev[0]:
            archive[c] = (score, key)

    for r in range(n_chains):
        record(bds.key(r), float(scores[r]))

    np_rng = np.random.default_rng(seed)
    cooling = (t_end / t_start) ** (1.0 / max(1, iters))
    temps = np.full(n_chains, t_start)
    chain_best = scores.copy()
    chain_best_key: list[tuple[int, ...]] = [bds.key(r) for r in range(n_chains)]
    since_improve = np.zeros(n_chains, dtype=np.int64)
    stall_window = max(200, iters // 8)

    # rows of letter_mask average the current window values over the windows
    # containing each cipher letter — a cheap per-letter badness signal
    n_windows = bds.n_windows
    letter_mask = None
    if bias_prob > 0 and n_windows:
        letter_mask = np.zeros((26, n_windows), dtype=np.float32)
        for c in range(26):
            wins = bds._ds._wins[c]
            if len(wins):
                letter_mask[c, wins] = 1.0 / len(wins)
    badness = None
    bias_interval = 100

    bias_start = iters // 2  # uniform mixing first, biased refinement later
    for step in range(iters):
        pairs = np_rng.integers(0, 26, size=(n_chains, 2))
        if letter_mask is not None and step >= bias_start:
            if (step - bias_start) % bias_interval == 0:
                # (26, R): mean window value per letter; lower = more suspect.
                # Letters absent from the cipher stay 0 (= relatively good).
                badness = letter_mask @ bds._cur_vals.T
            worst8 = np.argpartition(badness, 8, axis=0)[:8]  # (8, R)
            chosen = worst8[np_rng.integers(0, 8, n_chains), np.arange(n_chains)]
            use_bias = np_rng.random(n_chains) < bias_prob
            pairs[use_bias, 0] = chosen[use_bias]
        clash = pairs[:, 0] == pairs[:, 1]
        pairs[clash, 1] = (pairs[clash, 1] + 1) % 26
        d = bds.delta(pairs)
        u = np_rng.random(n_chains)
        accept = (d >= 0) | (u < np.exp(np.minimum(d, 0.0) / np.maximum(temps, 1e-12)))
        now = bds.accept(accept, pairs)
        since_improve += 1
        improved = np.flatnonzero(now > chain_best + 1e-12)
        for r in improved:
            chain_best[r] = now[r]
            since_improve[r] = 0
            chain_best_key[int(r)] = bds.key(int(r))
            record(chain_best_key[int(r)], float(now[r]))
        if reheat_factor > 1.0:
            stalled = np.flatnonzero(since_improve >= stall_window)
            for r in stalled:
                # basin hop: resume from the chain's best key, moderately hot,
                # so converged work is never thrown away
                bds.set_key(int(r), chain_best_key[int(r)])
                # gentle reheat: enough to cross small barriers near the
                # optimum, far below the chaotic regime (delta scale ~1e-2)
                temps[r] = max(temps[r], t_end * reheat_factor)
                since_improve[r] = 0
        temps *= cooling

    ranked = sorted(archive.values(), key=lambda sk: -sk[0])[: max(1, k)]
    return [(key, score) for score, key in ranked]
