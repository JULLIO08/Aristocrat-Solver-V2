"""A1 acceptance microbenchmark: DeltaScorer swap throughput + quadgram-only SA.

Usage:  python scripts/bench_delta.py
Reports swaps/sec (target >= 1e5) and whether a quadgram-only SA run solves
the e2e-style ~240-char cipher.
"""
import random
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

import numpy as np  # noqa: E402

from neurocrypt.cipher import clean_text, decode, encode, random_key, swap  # noqa: E402
from neurocrypt.corpus import load_lines, train_eval_split, DEFAULT_PLAIN  # noqa: E402
from neurocrypt.ngramfast import QuadgramModel, DeltaScorer, BatchDeltaScorer  # noqa: E402


def main() -> int:
    print("Training quadgram model on corpus train split...")
    t0 = time.perf_counter()
    train_lines, _ = train_eval_split(load_lines(DEFAULT_PLAIN), eval_frac=0.05, seed=42)
    model = QuadgramModel.train(train_lines)
    print(f"  trained on {len(train_lines)} lines in {time.perf_counter() - t0:.1f}s")

    # e2e-style cipher: several corpus sentences joined, ~240 chars
    fix = PKG_ROOT / "tests" / "fixtures" / "tiny_corpus.txt"
    plain = clean_text(" ".join(load_lines(fix)[:6]))
    key = random_key(random.Random(1))
    cipher = encode(plain, key)
    print(f"  cipher length: {len(cipher)} chars")

    # --- throughput ---
    ds = DeltaScorer(cipher, model)
    ds.set_key(random_key(random.Random(0)))
    rng = random.Random(0)
    pairs = [(rng.randrange(26), rng.randrange(26)) for _ in range(200_000)]
    pairs = [(i, j) for i, j in pairs if i != j]
    t0 = time.perf_counter()
    for i, j in pairs:
        ds.delta(i, j)
    dt = time.perf_counter() - t0
    rate = len(pairs) / dt
    print(f"  scalar delta throughput: {rate:,.0f} swaps/s "
          f"({len(pairs)} swaps in {dt:.2f}s)")

    # --- batched throughput (R parallel chains, the Stage-1 configuration) ---
    R = 16
    bds = BatchDeltaScorer(cipher, model, R)
    bds.set_keys([random_key(random.Random(s)) for s in range(R)])
    rng_np = np.random.default_rng(0)
    n_steps = 20_000
    all_pairs = rng_np.integers(0, 26, size=(n_steps, R, 2))
    clash = all_pairs[:, :, 0] == all_pairs[:, :, 1]
    all_pairs[:, :, 1] = np.where(clash, (all_pairs[:, :, 1] + 1) % 26,
                                  all_pairs[:, :, 1])
    t0 = time.perf_counter()
    for s in range(n_steps):
        p = all_pairs[s]
        d = bds.delta(p)
        bds.accept(d > 0, p)
    dt = time.perf_counter() - t0
    brate = n_steps * R / dt
    print(f"  batched delta throughput (R={R}): {brate:,.0f} swap-evals/s "
          f"({n_steps * R} evals in {dt:.2f}s)")
    ok_rate = brate >= 1e5
    print(f"  >= 1e5 swap evaluations/s: {'PASS' if ok_rate else 'FAIL'}")

    # --- quadgram-only SA solves the e2e cipher ---
    import math
    t0 = time.perf_counter()
    best_key, best_score = None, -1e18
    for restart in range(8):
        r = random.Random(100 + restart)
        cur = random_key(r)
        ds.set_key(cur)
        cur_score = ds.current_score
        temp, t_end, iters = 1.0, 0.01, 8000
        cooling = (t_end / temp) ** (1.0 / iters)
        for _ in range(iters):
            i, j = r.randrange(26), r.randrange(26)
            if i == j:
                temp *= cooling
                continue
            d = ds.delta(i, j)
            if d >= 0 or r.random() < math.exp(d / max(temp, 1e-9)):
                ds.apply(i, j)
                cur_score += d
                if cur_score > best_score:
                    best_score, best_key = cur_score, ds.key
            temp *= cooling
    dt = time.perf_counter() - t0
    plain_hat = decode(cipher, best_key)
    errors = sum(1 for a, b in zip(plain_hat, plain) if a != b)
    print(f"  quadgram-only SA: {errors} errors in {dt:.2f}s "
          f"({'PASS' if errors <= 2 else 'FAIL'})")
    print(f"  decoded: {plain_hat[:80]}...")
    return 0 if (ok_rate and errors <= 2) else 1


if __name__ == "__main__":
    sys.exit(main())
