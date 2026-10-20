import pytest
from neurocrypt.postag import POSModel

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "the early bird catches the worm",
    "actions speak louder than words",
]

@pytest.fixture(scope="module")
def model():
    return POSModel.train(CORPUS)

def test_score_is_average_logprob(model):
    s = model.score("the quick brown fox jumps")
    assert s <= 0.0

def test_ranks_grammatical_above_garbage(model):
    good = model.score("the quick brown fox jumps over the lazy dog")
    bad = model.score("fox dog the lazy quick brown over jumps the")
    assert good >= bad

def test_roundtrip(tmp_path, model):
    p = tmp_path / "pos.json"
    model.save(p)
    m2 = POSModel.load(p)
    assert abs(model.score("the quick brown fox") - m2.score("the quick brown fox")) < 1e-9
