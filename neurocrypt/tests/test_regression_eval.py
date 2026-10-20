"""Regression guard: fixed mini-eval that must never drop below its floor.

Marked slow - runs with the full suite only:
    python -m pytest tests/test_regression_eval.py -v
Fast suite excludes it via -m "not slow".

Models are built deterministically from the committed corpus (no
gitignored artifacts), exercising Stage 1, pattern seeding, escalation,
and lexicon-led reranking end to end.
"""
import random

import pytest

from neurocrypt.cipher import clean_text
from neurocrypt.corpus import (DEFAULT_ENCODED, DEFAULT_PLAIN,
                               eval_split_indices, load_lines,
                               train_eval_split)
from neurocrypt.lexicon import Lexicon
from neurocrypt.ngramfast import QuadgramModel
from neurocrypt.patterns import PatternIndex
from neurocrypt.scorer import Scorer, Weights
from neurocrypt.solver import Solver

N_CIPHERS = 10
EXACT_FLOOR = 6  # measured 7/10 at introduction; one cipher of slack


class _Zero:
    def score(self, text):
        return 0.0


@pytest.mark.slow
def test_mini_eval_exact_solves_meet_floor():
    plains = load_lines(DEFAULT_PLAIN)
    encs = load_lines(DEFAULT_ENCODED)
    train_lines, _ = train_eval_split(plains, eval_frac=0.05, seed=42)
    qmodel = QuadgramModel.train(train_lines)
    lex = Lexicon.from_corpus(train_lines, min_count=2)
    patterns = PatternIndex.from_corpus(train_lines, min_count=2)
    scorer = Scorer(_Zero(), _Zero(), lex, _Zero(), Weights(0, 0, 1.0, 0))
    solver = Solver(scorer, lex, qmodel=qmodel, patterns=patterns)

    pool = eval_split_indices(min(len(plains), len(encs)))
    random.Random(9).shuffle(pool)
    chosen = pool[:N_CIPHERS]

    exact = 0
    for i in chosen:
        plain, cipher = clean_text(plains[i]), encs[i]
        res = solver.solve(cipher, restarts=10, iters=4000, seed=i,
                           escalation_rounds=2)
        errors = sum(1 for a, b in zip(res.plaintext, plain) if a != b)
        if errors == 0:
            exact += 1
    assert exact >= EXACT_FLOOR, f"only {exact}/{N_CIPHERS} exact solves"
