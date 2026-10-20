import random
import subprocess
import sys
from pathlib import Path

import torch

from neurocrypt.corpus import load_lines
from neurocrypt.cipher import clean_text, encode, random_key
from neurocrypt.charlm import CharLM, CharLMConfig
from neurocrypt.wordlm import WordLM
from neurocrypt.lexicon import Lexicon
from neurocrypt.postag import POSModel
from neurocrypt.scorer import Scorer, Weights
from neurocrypt.solver import Solver

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


def _train_tiny() -> Solver:
    torch.manual_seed(0)  # deterministic training so the e2e result is reproducible
    lines = load_lines(FIX) * 40
    charlm = CharLM(CharLMConfig(d_model=48, n_heads=2, n_layers=2, block_size=48))
    charlm.fit(lines, epochs=12, batch_size=32, lr=2e-3, device="cpu")
    wordlm = WordLM.train(lines)
    lex = Lexicon.from_corpus(lines)
    pos = POSModel.train(lines[:20])
    # Calibrated weights: lexicon-led fusion recovers in-vocab text reliably.
    scorer = Scorer(charlm, wordlm, lex, pos, Weights(charlm=1.0, wordlm=1.0,
                                                      lexicon=2.0, postag=0.2))
    return Solver(scorer, lex)


def test_end_to_end_solves_known_cipher():
    """Train a tiny model and solve a multi-sentence in-vocabulary cipher.

    Substitution-cipher SA needs enough letter statistics to converge, so the
    plaintext joins several corpus sentences (~240 chars) rather than one short
    line. Calibrated config recovers 0/239 errors across seeds 1-3; we assert the
    same <=2-letter-error bar the production solver targets, with margin.
    """
    solver = _train_tiny()
    base = load_lines(FIX)
    plain = clean_text(" ".join(base[:6]))
    key = random_key(random.Random(1))
    cipher = encode(plain, key)
    result = solver.solve(cipher, restarts=12, iters=1500, seed=1)
    errors = sum(1 for a, b in zip(result.plaintext, plain) if a != b)
    assert errors <= 2, f"{errors} errors: {result.plaintext!r}"


def test_cli_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "neurocrypt.cli", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "solve" in out.stdout and "train" in out.stdout
