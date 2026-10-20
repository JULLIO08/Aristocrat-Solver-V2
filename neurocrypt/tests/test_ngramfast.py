"""Tests for the fast quadgram scorer and incremental delta scoring."""
import random

import numpy as np
import pytest

from neurocrypt.cipher import clean_text, decode, random_key, swap
from neurocrypt.ngramfast import QuadgramModel, DeltaScorer, BatchDeltaScorer

ENGLISH_LINES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES",
    "TO BE OR NOT TO BE THAT IS THE QUESTION",
    "ALL HAPPY FAMILIES ARE ALIKE EACH UNHAPPY FAMILY IS UNHAPPY IN ITS OWN WAY",
    "THE ONLY THING WE HAVE TO FEAR IS FEAR ITSELF",
    "IN THE BEGINNING WAS THE WORD AND THE WORD WAS WITH GOD",
    "ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU",
    "THE TRUTH WILL SET YOU FREE BUT FIRST IT WILL MAKE YOU MISERABLE",
]


@pytest.fixture(scope="module")
def model() -> QuadgramModel:
    return QuadgramModel.train(ENGLISH_LINES)


def test_ranks_english_above_gibberish(model):
    eng = "THE BEST OF TIMES WAS THE WORD"
    gib = "XQZJ KVWQ ZZXJ QQKV JXZQ WWQZ"
    assert model.score(eng) > model.score(gib)


def test_score_is_length_independent(model):
    text = "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES " \
           "ALL HAPPY FAMILIES ARE ALIKE EACH UNHAPPY FAMILY IS UNHAPPY"
    doubled = text + " " + text
    assert model.score(text) == pytest.approx(model.score(doubled), abs=0.05)


def test_short_text_scores_finite(model):
    for t in ["", "A", "AB", "ABC"]:
        s = model.score(t)
        assert np.isfinite(s)


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "quadgrams.npy"
    model.save(path)
    loaded = QuadgramModel.load(path)
    text = "THE ONLY THING WE HAVE TO FEAR"
    assert loaded.score(text) == pytest.approx(model.score(text), abs=1e-7)


def test_table_shape_and_dtype(model):
    assert model.table.shape == (27 ** 4,)
    assert model.table.dtype == np.float32


def test_delta_matches_full_rescore(model):
    rng = random.Random(7)
    cipher = clean_text(
        "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ DQG WKHQ "
        "UXQV DZDB LQWR WKH GDUN IRUHVW QHYHU WR EH VHHQ DJDLQ"
    )
    key = random_key(rng)
    ds = DeltaScorer(cipher, model)
    cur = ds.set_key(key)
    assert cur == pytest.approx(model.score(decode(cipher, key)), abs=1e-6)

    for _ in range(1000):
        i, j = rng.randrange(26), rng.randrange(26)
        if i == j:
            continue
        d = ds.delta(i, j)
        new_key = swap(key, i, j)
        full_new = model.score(decode(cipher, new_key))
        full_old = model.score(decode(cipher, key))
        assert d == pytest.approx(full_new - full_old, abs=1e-5)
        if rng.random() < 0.5:  # randomly accept some swaps
            ds.apply(i, j)
            key = new_key
            cur += d
    assert ds.current_score == pytest.approx(model.score(decode(cipher, key)), abs=1e-4)


def test_delta_scorer_no_spaces(model):
    """Cipher with no spaces must still work."""
    rng = random.Random(3)
    cipher = "WKHTXLFNEURZQIRAMXPSV"
    key = random_key(rng)
    ds = DeltaScorer(cipher, model)
    ds.set_key(key)
    d = ds.delta(0, 1)
    full = model.score(decode(cipher, swap(key, 0, 1))) - model.score(decode(cipher, key))
    assert d == pytest.approx(full, abs=1e-5)


def test_delta_scorer_short_cipher(model):
    """Cipher shorter than one quadgram window: deltas are zero, no crash."""
    ds = DeltaScorer("ABC", model)
    ds.set_key(tuple(range(26)))
    assert ds.delta(0, 1) == pytest.approx(0.0)


def test_batch_delta_matches_scalar(model):
    """Batched multi-chain deltas must equal the scalar DeltaScorer's."""
    rng = random.Random(11)
    cipher = clean_text(
        "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ DQG WKHQ "
        "UXQV DZDB LQWR WKH GDUN IRUHVW QHYHU WR EH VHHQ DJDLQ"
    )
    n_chains = 5
    keys = [random_key(rng) for _ in range(n_chains)]
    bds = BatchDeltaScorer(cipher, model, n_chains)
    scores = bds.set_keys(keys)
    for r in range(n_chains):
        assert scores[r] == pytest.approx(model.score(decode(cipher, keys[r])), abs=1e-5)

    for _ in range(200):
        pairs = np.array(
            [sorted(rng.sample(range(26), 2)) for _ in range(n_chains)], dtype=np.int64
        )
        d = bds.delta(pairs)
        for r in range(n_chains):
            i, j = int(pairs[r, 0]), int(pairs[r, 1])
            full_new = model.score(decode(cipher, swap(keys[r], i, j)))
            full_old = model.score(decode(cipher, keys[r]))
            assert d[r] == pytest.approx(full_new - full_old, abs=1e-5)
        accept = np.array([rng.random() < 0.4 for _ in range(n_chains)])
        bds.accept(accept, pairs)
        for r in range(n_chains):
            if accept[r]:
                keys[r] = swap(keys[r], int(pairs[r, 0]), int(pairs[r, 1]))
    for r in range(n_chains):
        assert bds.current_scores[r] == pytest.approx(
            model.score(decode(cipher, keys[r])), abs=1e-4
        )
        assert bds.key(r) == keys[r]
