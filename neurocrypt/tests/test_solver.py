import random
from pathlib import Path

from neurocrypt.cipher import clean_text, decode, encode, random_key
from neurocrypt.corpus import load_lines
from neurocrypt.ngramfast import QuadgramModel
from neurocrypt.scorer import Scorer, Weights
from neurocrypt.lexicon import Lexicon
from neurocrypt.solver import Solver, SolveResult

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"

class ZeroSignal:
    def score(self, text): return 0.0

def test_solver_runs_and_reports_confidence():
    plain = clean_text("the quick brown fox")
    key = random_key(random.Random(5))
    cipher = encode(plain, key)
    lex = Lexicon({"the", "quick", "brown", "fox"})
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex)
    result = solver.solve(cipher, restarts=6, iters=1500, seed=1)
    assert isinstance(result, SolveResult)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.plaintext) == len(cipher)
    assert sorted(result.key) == list(range(26))


def test_two_tier_solve_with_quadgram_stage1():
    """With a quadgram model, solve runs fast Stage-1 search + rerank."""
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(" ".join(lines[:6]))
    key = random_key(random.Random(1))
    cipher = encode(plain, key)
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    result = solver.solve(cipher, restarts=12, iters=8000, seed=1)
    errors = sum(1 for a, b in zip(result.plaintext, plain) if a != b)
    assert errors <= 2, f"{errors} errors: {result.plaintext!r}"
    assert result.confidence >= 0.9


def test_two_tier_solve_deterministic():
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(" ".join(lines[:4]))
    cipher = encode(plain, random_key(random.Random(2)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    r1 = solver.solve(cipher, restarts=6, iters=2000, seed=3)
    r2 = solver.solve(cipher, restarts=6, iters=2000, seed=3)
    assert r1.plaintext == r2.plaintext and r1.key == r2.key


def test_calibration_maps_coverage_and_score_to_probability():
    from neurocrypt.solver import Calibration

    cal = Calibration(coef_coverage=10.0, coef_score=0.0, intercept=-9.0)
    hi = cal.probability(coverage=1.0, score=0.0)
    lo = cal.probability(coverage=0.5, score=0.0)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_calibration_roundtrip_and_solver_uses_it(tmp_path):
    import json
    from neurocrypt.solver import Calibration

    cal = Calibration(coef_coverage=8.0, coef_score=0.5, intercept=-6.0)
    path = tmp_path / "calibration.json"
    cal.save(path)
    loaded = Calibration.load(path)
    assert loaded == cal

    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(" ".join(lines[:4]))
    cipher = encode(plain, random_key(random.Random(2)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    raw = Solver(scorer, lex, qmodel=qmodel)
    calib = Solver(scorer, lex, qmodel=qmodel, calibration=cal)
    r_raw = raw.solve(cipher, restarts=6, iters=2000, seed=3)
    r_cal = calib.solve(cipher, restarts=6, iters=2000, seed=3)
    assert r_cal.plaintext == r_raw.plaintext
    expected = cal.probability(coverage=lex.score(r_raw.plaintext),
                               score=r_raw.score)
    assert abs(r_cal.confidence - expected) < 1e-9


def test_escalation_improves_hard_short_cipher():
    """Low base budget fails a short cipher; escalation recovers it."""
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(lines[1])  # ~56 chars: short bucket, in-vocab
    cipher = encode(plain, random_key(random.Random(4)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    base = solver.solve(cipher, restarts=2, iters=300, seed=11,
                        escalation_rounds=0)
    esc = solver.solve(cipher, restarts=2, iters=300, seed=11,
                       escalation_rounds=3)
    base_err = sum(1 for a, b in zip(base.plaintext, plain) if a != b)
    esc_err = sum(1 for a, b in zip(esc.plaintext, plain) if a != b)
    assert base.confidence < 0.9  # premise: base budget really fails
    assert esc_err < base_err
    assert esc.escalations > 0


def test_no_escalation_when_confident():
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(" ".join(lines[:6]))
    cipher = encode(plain, random_key(random.Random(1)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    result = solver.solve(cipher, restarts=12, iters=8000, seed=1,
                          escalation_rounds=2)
    assert result.confidence >= 0.9
    assert result.escalations == 0


def test_escalation_deterministic():
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(lines[1])
    cipher = encode(plain, random_key(random.Random(4)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    r1 = solver.solve(cipher, restarts=2, iters=300, seed=11, escalation_rounds=3)
    r2 = solver.solve(cipher, restarts=2, iters=300, seed=11, escalation_rounds=3)
    assert r1.plaintext == r2.plaintext and r1.key == r2.key


def test_candidates_returns_ranked_decodes():
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    plain = clean_text(" ".join(lines[:4]))
    cipher = encode(plain, random_key(random.Random(2)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    cands = solver.candidates(cipher, restarts=6, iters=2000, seed=3, k=5)
    assert 1 <= len(cands) <= 5
    texts = [t for t, _ in cands]
    scores = [s for _, s in cands]
    assert scores == sorted(scores, reverse=True)
    assert len(set(texts)) == len(texts)
    assert all(len(t) == len(plain) for t in texts)


def test_legacy_solve_path_still_works():
    """legacy=True must bypass Stage 1 even when a quadgram model exists."""
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines)
    lex = Lexicon({"the", "quick", "brown", "fox"})
    plain = clean_text("the quick brown fox")
    cipher = encode(plain, random_key(random.Random(5)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel)
    result = solver.solve(cipher, restarts=6, iters=1500, seed=1, legacy=True)
    assert isinstance(result, SolveResult)
    assert 0.0 <= result.confidence <= 1.0
