"""Tests for eval reporting: buckets, bootstrap CIs, JSON summary."""
import json

from neurocrypt.evalreport import bucket_of, bootstrap_ci, summarize


def _rec(errors, length, conf=0.99, wall=0.5, esc=0):
    return {"length": length, "bucket": bucket_of(length), "errors": errors,
            "error_rate": errors / length, "confidence": conf,
            "wall_time": wall, "escalations": esc}


def test_bucket_boundaries():
    assert bucket_of(10) == "<60"
    assert bucket_of(59) == "<60"
    assert bucket_of(60) == "60-120"
    assert bucket_of(120) == "60-120"
    assert bucket_of(121) == ">120"


def test_bootstrap_ci_deterministic_and_ordered():
    vals = [1, 0, 1, 1, 0, 1, 1, 1]
    lo1, hi1 = bootstrap_ci(vals, seed=7)
    lo2, hi2 = bootstrap_ci(vals, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert 0.0 <= lo1 <= sum(vals) / len(vals) <= hi1 <= 1.0


def test_bootstrap_ci_degenerate():
    assert bootstrap_ci([1, 1, 1], seed=0) == (1.0, 1.0)
    assert bootstrap_ci([], seed=0) == (0.0, 0.0)


def test_summarize_schema_and_rates():
    records = ([_rec(0, 50)] * 3 + [_rec(5, 50)] +
               [_rec(0, 100)] * 4 + [_rec(0, 200, wall=2.0)] * 2)
    s = summarize(records, seed=1)
    assert s["n"] == 10
    assert s["exact_solve_rate"] == 0.9
    assert s["mean_error_rate"] > 0
    assert set(s["buckets"]) == {"<60", "60-120", ">120"}
    b = s["buckets"]["<60"]
    assert b["n"] == 4 and b["exact_solve_rate"] == 0.75
    assert len(b["ci95"]) == 2
    assert s["median_wall_time"] > 0
    json.dumps(s)  # must be JSON-serializable
