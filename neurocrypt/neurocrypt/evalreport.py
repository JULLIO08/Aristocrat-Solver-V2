"""Eval reporting: length buckets, seeded bootstrap CIs, JSON summary."""
from __future__ import annotations

import numpy as np


def bucket_of(length: int) -> str:
    if length < 60:
        return "<60"
    if length <= 120:
        return "60-120"
    return ">120"


def bootstrap_ci(values, seed: int, n_boot: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Seeded percentile bootstrap CI for the mean of 0/1 (or float) values."""
    vals = np.asarray(list(values), dtype=np.float64)
    if len(vals) == 0:
        return (0.0, 0.0)
    if np.all(vals == vals[0]):
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(records: list[dict], seed: int = 0) -> dict:
    """Aggregate per-cipher eval records into the JSON summary block."""
    exact = [1.0 if r["errors"] == 0 else 0.0 for r in records]
    walls = sorted(r["wall_time"] for r in records)
    n = len(records)
    buckets: dict[str, dict] = {}
    for name in ("<60", "60-120", ">120"):
        sub = [r for r in records if r["bucket"] == name]
        sub_exact = [1.0 if r["errors"] == 0 else 0.0 for r in sub]
        buckets[name] = {
            "n": len(sub),
            "exact_solve_rate": (sum(sub_exact) / len(sub)) if sub else None,
            "mean_error_rate": (sum(r["error_rate"] for r in sub) / len(sub))
                               if sub else None,
            "ci95": list(bootstrap_ci(sub_exact, seed=seed)),
        }
    return {
        "n": n,
        "exact_solve_rate": sum(exact) / n if n else 0.0,
        "exact_solve_ci95": list(bootstrap_ci(exact, seed=seed)),
        "mean_error_rate": (sum(r["error_rate"] for r in records) / n) if n else 0.0,
        "median_wall_time": walls[n // 2] if n else 0.0,
        "p95_wall_time": walls[min(n - 1, int(0.95 * n))] if n else 0.0,
        "escalated": sum(1 for r in records if r.get("escalations", 0) > 0),
        "buckets": buckets,
    }
