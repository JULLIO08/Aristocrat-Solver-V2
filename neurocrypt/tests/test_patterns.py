"""Tests for the word-pattern dictionary attack (Stage-0 seeding)."""
import random
from pathlib import Path

from neurocrypt.cipher import clean_text, decode, encode, random_key
from neurocrypt.corpus import load_lines
from neurocrypt.patterns import signature, PatternIndex

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


def test_signature_examples():
    assert signature("PEOPLE") == "ABCADB"
    assert signature("THE") == "ABC"
    assert signature("LL") == "AA"
    assert signature("A") == "A"


def test_index_orders_candidates_by_frequency():
    idx = PatternIndex.from_corpus(["THE CAT THE DOG THE FOX", "WAS CAT"])
    cands = idx.by_sig[signature("XYZ")]
    assert cands[0] == "THE"  # most frequent ABC-pattern word first


def test_seed_keys_are_permutations_and_capped():
    lines = load_lines(FIX)
    idx = PatternIndex.from_corpus(lines)
    plain = clean_text(lines[2])
    cipher = encode(plain, random_key(random.Random(7)))
    seeds = idx.seed_keys(cipher, max_seeds=50)
    assert 0 < len(seeds) <= 50
    for s in seeds:
        assert sorted(s) == list(range(26))


def test_seed_keys_skip_unknown_words_without_crash():
    idx = PatternIndex.from_corpus(["THE CAT SAT"])
    # second word has a signature absent from the index
    seeds = idx.seed_keys("QQQQQQQQ XYZX QQQ")
    assert isinstance(seeds, list)


def test_pattern_seeding_solves_hard_short_cipher():
    """The 29-char cipher SA alone cannot solve falls to the pattern attack."""
    from neurocrypt.lexicon import Lexicon
    from neurocrypt.ngramfast import QuadgramModel
    from neurocrypt.scorer import Scorer, Weights
    from neurocrypt.solver import Solver

    class ZeroSignal:
        def score(self, text):
            return 0.0

    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    idx = PatternIndex.from_corpus(lines)
    plain = clean_text(lines[3])  # ALL THAT GLITTERS IS NOT GOLD (29 chars)
    cipher = encode(plain, random_key(random.Random(4)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))

    no_seed = Solver(scorer, lex, qmodel=qmodel)
    base = no_seed.solve(cipher, restarts=3, iters=400, seed=11,
                         escalation_rounds=0)
    base_err = sum(1 for a, b in zip(base.plaintext, plain) if a != b)

    seeded = Solver(scorer, lex, qmodel=qmodel, patterns=idx)
    res = seeded.solve(cipher, restarts=3, iters=400, seed=11,
                       escalation_rounds=0)
    err = sum(1 for a, b in zip(res.plaintext, plain) if a != b)
    assert base_err > 2  # premise: unsolved without seeding
    assert err <= 2, f"{err} errors: {res.plaintext!r}"


def test_pattern_seeding_deterministic():
    from neurocrypt.lexicon import Lexicon
    from neurocrypt.ngramfast import QuadgramModel
    from neurocrypt.scorer import Scorer, Weights
    from neurocrypt.solver import Solver

    class ZeroSignal:
        def score(self, text):
            return 0.0

    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    idx = PatternIndex.from_corpus(lines)
    cipher = encode(clean_text(lines[3]), random_key(random.Random(4)))
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel, patterns=idx)
    r1 = solver.solve(cipher, restarts=4, iters=1000, seed=3, escalation_rounds=0)
    r2 = solver.solve(cipher, restarts=4, iters=1000, seed=3, escalation_rounds=0)
    assert r1.plaintext == r2.plaintext


def test_index_roundtrip(tmp_path):
    lines = load_lines(FIX)
    idx = PatternIndex.from_corpus(lines)
    p = tmp_path / "patterns.pkl"
    idx.save(p)
    loaded = PatternIndex.load(p)
    assert loaded.by_sig == idx.by_sig
