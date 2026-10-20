from neurocrypt.scorer import Scorer, Weights

class FakeSignal:
    def __init__(self, table): self.table = table
    def score(self, text): return self.table.get(text, -10.0)

def test_weighted_sum():
    s = Scorer(
        charlm=FakeSignal({"A": -1.0}),
        wordlm=FakeSignal({"A": -2.0}),
        lexicon=FakeSignal({"A": 1.0}),
        postag=FakeSignal({"A": -3.0}),
        weights=Weights(charlm=1.0, wordlm=1.0, lexicon=1.0, postag=1.0),
    )
    assert s.score("A") == (-1.0) + (-2.0) + (1.0) + (-3.0)

def test_weights_change_result():
    sig = dict(
        charlm=FakeSignal({"A": -1.0}), wordlm=FakeSignal({"A": -2.0}),
        lexicon=FakeSignal({"A": 1.0}), postag=FakeSignal({"A": -3.0}),
    )
    s1 = Scorer(**sig, weights=Weights(charlm=2.0, wordlm=0, lexicon=0, postag=0))
    assert s1.score("A") == -2.0

def test_cache_avoids_recompute():
    calls = {"n": 0}
    class Counting:
        def score(self, text):
            calls["n"] += 1
            return -1.0
    s = Scorer(charlm=Counting(), wordlm=Counting(), lexicon=Counting(),
               postag=Counting(), weights=Weights(1, 1, 1, 1))
    s.score("HELLO"); s.score("HELLO")
    assert calls["n"] == 4

def test_rerank_with_real_charlm_matches_loop_scores():
    import torch
    from neurocrypt.charlm import CharLM, CharLMConfig
    from neurocrypt.lexicon import Lexicon

    torch.manual_seed(0)
    lm = CharLM(CharLMConfig(d_model=32, n_heads=2, n_layers=2, block_size=32))
    lex = Lexicon({"the", "fox"})
    s = Scorer(charlm=lm, wordlm=FakeSignal({}), lexicon=lex,
               postag=FakeSignal({}), weights=Weights(1.0, 0.1, 1.0, 0.1))
    cands = ["THE FOX", "XQZ JKW", "THE THE THE"]
    ranked = s.rerank(cands)
    # independent scorer with an empty cache, forced down the loop path
    s2 = Scorer(charlm=lm, wordlm=FakeSignal({}), lexicon=lex,
                postag=FakeSignal({}), weights=Weights(1.0, 0.1, 1.0, 0.1))
    loop = sorted(((i, s2.score(t)) for i, t in enumerate(cands)),
                  key=lambda x: -x[1])
    assert [i for i, _ in ranked] == [i for i, _ in loop]
    for (_, a), (_, b) in zip(ranked, loop):
        assert abs(a - b) < 1e-4


def test_rerank_orders_candidates_by_fused_score():
    s = Scorer(
        charlm=FakeSignal({"GOOD": -1.0, "OKAY": -2.0, "BAD": -9.0}),
        wordlm=FakeSignal({"GOOD": -1.0, "OKAY": -2.0, "BAD": -9.0}),
        lexicon=FakeSignal({"GOOD": 1.0, "OKAY": 0.5, "BAD": 0.0}),
        postag=FakeSignal({"GOOD": -1.0, "OKAY": -2.0, "BAD": -9.0}),
        weights=Weights(1.0, 1.0, 1.0, 1.0),
    )
    ranked = s.rerank(["BAD", "GOOD", "OKAY"])
    assert [idx for idx, _ in ranked] == [1, 2, 0]
    scores = [sc for _, sc in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][1] == s.score("GOOD")
