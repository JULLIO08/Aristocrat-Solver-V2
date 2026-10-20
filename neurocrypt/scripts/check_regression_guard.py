"""C3 acceptance check: degrading a weight must break the regression floor."""
import random
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from neurocrypt.cipher import clean_text  # noqa: E402
from neurocrypt.corpus import (DEFAULT_ENCODED, DEFAULT_PLAIN,  # noqa: E402
                               eval_split_indices, load_lines, train_eval_split)
from neurocrypt.lexicon import Lexicon  # noqa: E402
from neurocrypt.ngramfast import QuadgramModel  # noqa: E402
from neurocrypt.patterns import PatternIndex  # noqa: E402
from neurocrypt.scorer import Scorer, Weights  # noqa: E402
from neurocrypt.solver import Solver  # noqa: E402


class _Zero:
    def score(self, text):
        return 0.0


def run(weights: Weights) -> int:
    plains = load_lines(DEFAULT_PLAIN)
    encs = load_lines(DEFAULT_ENCODED)
    train_lines, _ = train_eval_split(plains, eval_frac=0.05, seed=42)
    qmodel = QuadgramModel.train(train_lines)
    lex = Lexicon.from_corpus(train_lines, min_count=2)
    patterns = PatternIndex.from_corpus(train_lines, min_count=2)
    solver = Solver(Scorer(_Zero(), _Zero(), lex, _Zero(), weights), lex,
                    qmodel=qmodel, patterns=patterns)
    pool = eval_split_indices(min(len(plains), len(encs)))
    random.Random(9).shuffle(pool)
    exact = 0
    for i in pool[:10]:
        plain, cipher = clean_text(plains[i]), encs[i]
        res = solver.solve(cipher, restarts=10, iters=4000, seed=i,
                           escalation_rounds=2)
        if res.plaintext == plain:
            exact += 1
    return exact


if __name__ == "__main__":
    good = run(Weights(0, 0, 1.0, 0))
    bad = run(Weights(0, 0, -1.0, 0))  # inverted lexicon ranking
    print(f"healthy weights: {good}/10   degraded weights: {bad}/10")
    print("guard detects degradation:" , "PASS" if (good >= 6 and bad < 6) else "FAIL")
