"""D3: refit confidence calibration on real end-to-end solve outcomes.

Runs the production solve() (two-tier + pattern seeds + escalation) on
held-out ciphers, then fits the logistic Calibration on (coverage,
fused score) -> exact. Reports precision/recall of "claimed solved" at
tau=0.9 on a validation half before writing models/calibration.json.

Usage: python scripts/recalibrate.py [--n 200] [--seed 1]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from neurocrypt.cipher import clean_text  # noqa: E402
from neurocrypt.corpus import (DEFAULT_ENCODED, DEFAULT_PLAIN,  # noqa: E402
                               eval_split_indices, load_lines)
from neurocrypt.solver import Calibration, Solver  # noqa: E402

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from tune_weights import fit_logistic  # noqa: E402

MODELS = PKG_ROOT / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()

    solver = Solver.from_models(MODELS)
    plains = load_lines(DEFAULT_PLAIN)
    encs = load_lines(DEFAULT_ENCODED)
    pool = eval_split_indices(min(len(plains), len(encs)))
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.n]

    feats, labels = [], []
    t0 = time.perf_counter()
    for k, i in enumerate(chosen):
        plain, cipher = clean_text(plains[i]), encs[i]
        res = solver.solve(cipher, restarts=args.restarts, iters=args.iters,
                           seed=i)
        coverage = solver.lexicon.score(res.plaintext)
        feats.append([coverage, res.score])
        labels.append(1.0 if res.plaintext == plain else 0.0)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(chosen)} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    feats = np.array(feats)
    labels = np.array(labels)
    print(f"end-to-end exact-solve on held-out: {int(labels.sum())}/{len(labels)}")

    half = len(labels) // 2
    w = fit_logistic(feats[:half], labels[:half], seed=args.seed)
    cal = Calibration(coef_coverage=float(w[0]), coef_score=float(w[1]),
                      intercept=float(w[2]))

    probs = np.array([cal.probability(c, s) for c, s in feats[half:]])
    y = labels[half:]
    claimed = probs >= 0.9
    precision = float(y[claimed].mean()) if claimed.any() else float("nan")
    recall = float(claimed[y == 1].mean()) if (y == 1).any() else float("nan")
    print(f"validation: claimed {int(claimed.sum())}/{len(y)} at tau=0.9, "
          f"precision {precision:.3f}, recall {recall:.3f}")

    cal.save(MODELS / "calibration.json")
    print(f"Wrote {cal}")
    summary = {
        "n": len(labels), "exact": int(labels.sum()),
        "precision_at_0.9": precision, "recall_at_0.9": recall,
        "claimed_at_0.9": int(claimed.sum()), "n_validate": len(y),
        "calibration": {"coef_coverage": cal.coef_coverage,
                        "coef_score": cal.coef_score,
                        "intercept": cal.intercept},
    }
    (PKG_ROOT / "results" / "calibration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
