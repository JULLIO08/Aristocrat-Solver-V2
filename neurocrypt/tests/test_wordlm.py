from neurocrypt.wordlm import WordLM

CORPUS = [
    "the quick brown fox",
    "the lazy brown dog",
    "the quick brown dog",
    "a quick fox runs",
]

def test_train_then_score_is_negative_logprob():
    lm = WordLM.train(CORPUS)
    s = lm.score("the quick brown fox")
    assert s < 0.0

def test_ranks_seen_sequence_above_scrambled():
    lm = WordLM.train(CORPUS)
    good = lm.score("the quick brown fox")
    bad = lm.score("fox brown quick the")
    assert good > bad

def test_empty_text_scores_finite():
    lm = WordLM.train(CORPUS)
    assert lm.score("") <= 0.0

def test_roundtrip_serialization(tmp_path):
    lm = WordLM.train(CORPUS)
    p = tmp_path / "wordlm.json"
    lm.save(p)
    lm2 = WordLM.load(p)
    assert abs(lm.score("the quick brown fox") - lm2.score("the quick brown fox")) < 1e-9
