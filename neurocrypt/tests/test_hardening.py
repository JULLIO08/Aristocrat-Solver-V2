"""C4: input edge cases and friendly CLI errors."""
import random
import subprocess
import sys
from pathlib import Path

import pytest

from neurocrypt.cipher import clean_text, encode, random_key
from neurocrypt.corpus import load_lines
from neurocrypt.lexicon import Lexicon
from neurocrypt.ngramfast import QuadgramModel
from neurocrypt.patterns import PatternIndex
from neurocrypt.scorer import Scorer, Weights
from neurocrypt.solver import Solver, SolveResult

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


class ZeroSignal:
    def score(self, text):
        return 0.0


@pytest.fixture(scope="module")
def solver():
    lines = load_lines(FIX)
    qmodel = QuadgramModel.train(lines * 40)
    lex = Lexicon.from_corpus(lines)
    patterns = PatternIndex.from_corpus(lines)
    scorer = Scorer(ZeroSignal(), ZeroSignal(), lex, ZeroSignal(),
                    Weights(0, 0, 1.0, 0))
    return Solver(scorer, lex, qmodel=qmodel, patterns=patterns)


def test_empty_input(solver):
    res = solver.solve("", restarts=2, iters=100, seed=0)
    assert isinstance(res, SolveResult)
    assert res.plaintext == ""


def test_punctuation_digits_unicode_stripped(solver):
    res = solver.solve("Don't! 123 café — QXR?", restarts=2,
                       iters=200, seed=0)
    assert isinstance(res, SolveResult)
    assert all(c == " " or "A" <= c <= "Z" for c in res.plaintext)


def test_all_identical_letters(solver):
    res = solver.solve("QQQQQ QQQ QQQQ", restarts=2, iters=200, seed=0)
    assert isinstance(res, SolveResult)
    assert len(res.plaintext) == len("QQQQQ QQQ QQQQ")


def test_already_plaintext(solver):
    text = clean_text("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
    res = solver.solve(text, restarts=6, iters=2000, seed=0)
    assert isinstance(res, SolveResult)


def test_no_spaces_cipher_still_solves(solver):
    plain = clean_text("the quick brown fox").replace(" ", "")
    cipher = encode(plain, random_key(random.Random(2)))
    res = solver.solve(cipher, restarts=2, iters=500, seed=0)
    assert isinstance(res, SolveResult)
    assert " " not in res.plaintext


def test_very_long_input(solver):
    lines = load_lines(FIX)
    plain = clean_text(" ".join(lines * 60))  # > 5000 chars
    assert len(plain) > 5000
    cipher = encode(plain, random_key(random.Random(3)))
    res = solver.solve(cipher, restarts=2, iters=500, seed=0)
    assert isinstance(res, SolveResult)
    assert len(res.plaintext) == len(plain)


def test_cli_missing_models_friendly_error(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "neurocrypt.cli", "solve", "ABC DEF",
         "--models", str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    combined = out.stdout + out.stderr
    assert "Traceback" not in combined
    assert "train" in combined  # actionable hint


def test_cli_empty_ciphertext_friendly_error():
    out = subprocess.run(
        [sys.executable, "-m", "neurocrypt.cli", "solve", ""],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "Traceback" not in (out.stdout + out.stderr)
