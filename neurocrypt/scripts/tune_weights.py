"""B1: tune fusion weights + fit confidence calibration on true held-out data.

Protocol:
- Held-out pool = the 5%% eval split (seed 42) that the models never trained on.
- For each cipher, Stage 1 (search_topk) emits K candidate decodes; each gets
  the four signal scores and an exact/not label.
- Random search (seeded) over weights maximizes exact-solve-after-rerank on the
  TUNE half; the winner must also beat the defaults on the VALIDATE half before
  models/weights.json is written.
- A logistic (Platt) calibration is then fit on (coverage, fused score) ->
  solved, written to models/calibration.json.

Usage: python scripts/tune_weights.py [--n 100] [--trials 2000] [--seed 0]
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

from neurocrypt.charlm import CharLM  # noqa: E402
from neurocrypt.cipher import decode  # noqa: E402
from neurocrypt.corpus import load_lines, DEFAULT_PLAIN, DEFAULT_ENCODED  # noqa: E402
from neurocrypt.lexicon import Lexicon  # noqa: E402
from neurocrypt.ngramfast import QuadgramModel  # noqa: E402
from neurocrypt.postag import POSModel  # noqa: E402
from neurocrypt.scorer import Weights  # noqa: E402
from neurocrypt.search import search_topk  # noqa: E402
from neurocrypt.solver import Calibration  # noqa: E402
from neurocrypt.wordlm import WordLM  # noqa: E402

MODELS = PKG_ROOT / "models"


def eval_split_indices(n_lines: int, eval_frac: float = 0.05, seed: int = 42) -> list[int]:
    """Replicates corpus.train_eval_split index selection."""
    idx = list(range(n_lines))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(n_lines * eval_frac))
    return sorted(idx[:n_eval])


def collect_candidates(args, plains, encs, indices, charlm, wordlm, lexicon,
                       postag, qmodel):
    """Per cipher: candidate decodes, 4-signal matrix, exact labels."""
    data = []
    t0 = time.perf_counter()
    for k, i in enumerate(indices):
        plain, cipher = plains[i], encs[i]
        cands = search_topk(cipher, qmodel, restarts=args.restarts,
                            iters=args.iters, k=args.k, seed=i)
        decodes = [decode(cipher, key) for key, _ in cands]
        sig = np.zeros((len(decodes), 4))
        sig[:, 0] = charlm.score_batch(decodes)
        sig[:, 1] = [wordlm.score(t) for t in decodes]
        sig[:, 2] = [lexicon.score(t) for t in decodes]
        sig[:, 3] = [postag.score(t) for t in decodes]
        labels = np.array([t == plain for t in decodes])
        data.append({"sig": sig, "labels": labels, "coverage": sig[:, 2].copy()})
        if (k + 1) % 20 == 0:
            print(f"  collected {k + 1}/{len(indices)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return data


def solve_count(weights: np.ndarray, data) -> int:
    """How many ciphers pick an exact decode after reranking with weights."""
    n = 0
    for d in data:
        fused = d["sig"] @ weights
        if d["labels"][int(np.argmax(fused))]:
            n += 1
    return n


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int, iters: int = 200,
                 ) -> np.ndarray:
    """Seeded 2-feature logistic regression via Newton's method, pure numpy."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=3)
    # standardize both features for conditioning, fold back after
    mu = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0), 1e-9)
    xs = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs @ w)))
        grad = xs.T @ (p - y) / len(y)
        s = np.maximum(p * (1 - p), 1e-6)
        hess = (xs * s[:, None]).T @ xs / len(y) + 1e-6 * np.eye(3)
        w -= np.linalg.solve(hess, grad)
    w_raw = np.array([w[0] / sd[0], w[1] / sd[1],
                      w[2] - w[0] * mu[0] / sd[0] - w[1] * mu[1] / sd[1]])
    return w_raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="total held-out ciphers")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    print("Loading models...")
    charlm = CharLM.load(MODELS / "charlm.pt")
    wordlm = WordLM.load(MODELS / "wordlm.json")
    lexicon = Lexicon(set(json.loads((MODELS / "lexicon.json").read_text(encoding="utf-8"))))
    postag = POSModel.load(MODELS / "postag.json")
    qmodel = QuadgramModel.load(MODELS / "quadgrams.npy")

    plains = load_lines(DEFAULT_PLAIN)
    encs = load_lines(DEFAULT_ENCODED)
    pool = eval_split_indices(min(len(plains), len(encs)))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen = pool[: args.n]
    half = len(chosen) // 2
    tune_idx, val_idx = chosen[:half], chosen[half:]
    print(f"Held-out pool {len(pool)}; tune {len(tune_idx)}, validate {len(val_idx)}")

    print("Collecting Stage-1 candidates (tune half)...")
    tune = collect_candidates(args, plains, encs, tune_idx, charlm, wordlm,
                              lexicon, postag, qmodel)
    print("Collecting Stage-1 candidates (validate half)...")
    val = collect_candidates(args, plains, encs, val_idx, charlm, wordlm,
                             lexicon, postag, qmodel)

    oracle_tune = sum(1 for d in tune if d["labels"].any())
    oracle_val = sum(1 for d in val if d["labels"].any())
    print(f"Oracle (exact decode present in top-K): tune {oracle_tune}/{len(tune)}, "
          f"val {oracle_val}/{len(val)}")

    default_w = Weights()
    default_vec = np.array([default_w.charlm, default_w.wordlm,
                            default_w.lexicon, default_w.postag])
    base_tune = solve_count(default_vec, tune)
    base_val = solve_count(default_vec, val)
    print(f"Default weights: tune {base_tune}/{len(tune)}, val {base_val}/{len(val)}")

    np_rng = np.random.default_rng(args.seed)
    best_vec, best_n = default_vec, base_tune
    for t in range(args.trials):
        cand = np.exp(np_rng.uniform(np.log(0.01), np.log(10.0), size=4))
        n = solve_count(cand, tune)
        if n > best_n:
            best_n, best_vec = n, cand
            print(f"  trial {t}: tune {n}/{len(tune)} w={np.round(cand, 3)}")
    val_n = solve_count(best_vec, val)
    print(f"Best: tune {best_n}/{len(tune)}, validate {val_n}/{len(val)} "
          f"(default val {base_val}/{len(val)})")

    if best_n > base_tune and val_n >= base_val:
        w = Weights(charlm=float(best_vec[0]), wordlm=float(best_vec[1]),
                    lexicon=float(best_vec[2]), postag=float(best_vec[3]))
        w.save(MODELS / "weights.json")
        print(f"Wrote tuned weights: {w}")
        final_vec = best_vec
    else:
        print("Tuned weights did not validate; keeping defaults.")
        final_vec = default_vec

    # ---- calibration on all collected ciphers with the final weights ----
    feats, labels = [], []
    for d in tune + val:
        fused = d["sig"] @ final_vec
        pick = int(np.argmax(fused))
        feats.append([d["coverage"][pick], fused[pick]])
        labels.append(1.0 if d["labels"][pick] else 0.0)
    feats = np.array(feats)
    labels = np.array(labels)
    w = fit_logistic(feats, labels, seed=args.seed)
    cal = Calibration(coef_coverage=float(w[0]), coef_score=float(w[1]),
                      intercept=float(w[2]))
    cal.save(MODELS / "calibration.json")
    print(f"Wrote calibration: {cal}")

    # precision of "claimed solved" at tau=0.9 on the validate half
    n_val = len(val)
    val_feats = feats[len(tune):]
    val_labels = labels[len(tune):]
    probs = np.array([cal.probability(c, s) for c, s in val_feats])
    claimed = probs >= 0.9
    if claimed.any():
        prec = float(val_labels[claimed].mean())
    else:
        prec = float("nan")
    print(f"Validate: {int(claimed.sum())}/{n_val} claimed solved at tau=0.9, "
          f"precision {prec:.3f}")
    summary = {
        "tune_default": base_tune, "tune_best": best_n,
        "val_default": base_val, "val_best": val_n,
        "n_tune": len(tune), "n_val": n_val,
        "weights": list(map(float, final_vec)),
        "calibration": {"coef_coverage": cal.coef_coverage,
                        "coef_score": cal.coef_score,
                        "intercept": cal.intercept},
        "precision_at_0.9": prec,
        "claimed_at_0.9": int(claimed.sum()),
    }
    out = PKG_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary -> {out / 'tuning.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
