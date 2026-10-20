from pathlib import Path
from neurocrypt.corpus import load_lines, train_eval_split

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"

def test_load_lines_cleans_and_drops_empties():
    lines = load_lines(FIX)
    assert len(lines) == 10
    assert all(line == line.upper() for line in lines)
    assert lines[0] == "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

def test_train_eval_split_is_disjoint_and_deterministic():
    lines = load_lines(FIX)
    tr1, ev1 = train_eval_split(lines, eval_frac=0.2, seed=42)
    tr2, ev2 = train_eval_split(lines, eval_frac=0.2, seed=42)
    assert tr1 == tr2 and ev1 == ev2
    assert len(ev1) == 2 and len(tr1) == 8
    assert set(tr1).isdisjoint(set(ev1))
