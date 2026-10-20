import torch
from neurocrypt.charlm import CharLM, CharLMConfig

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps while the quick fox runs",
    "quick brown foxes jump over lazy dogs often",
] * 20

TINY = CharLMConfig(d_model=32, n_heads=2, n_layers=2, block_size=32)

def test_vocab_covers_letters_and_space():
    lm = CharLM(TINY)
    enc = lm.encode("ABC XYZ")
    assert lm.decode(enc) == "ABC XYZ"

def test_training_reduces_loss():
    torch.manual_seed(0)
    lm = CharLM(TINY)
    losses = lm.fit(CORPUS, epochs=3, batch_size=8, lr=1e-3, device="cpu")
    assert losses[-1] < losses[0]

def test_score_ranks_english_above_scramble():
    torch.manual_seed(0)
    lm = CharLM(TINY)
    lm.fit(CORPUS, epochs=5, batch_size=8, lr=1e-3, device="cpu")
    good = lm.score("the quick brown fox")
    bad = lm.score("xqz jkw bvf mqp tll")
    assert good > bad

def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    lm = CharLM(TINY)
    lm.fit(CORPUS, epochs=1, batch_size=8, lr=1e-3, device="cpu")
    p = tmp_path / "charlm.pt"
    lm.save(p)
    lm2 = CharLM.load(p, device="cpu")
    assert abs(lm.score("the quick fox") - lm2.score("the quick fox")) < 1e-4

def test_score_batch_matches_loop():
    torch.manual_seed(0)
    lm = CharLM(TINY)  # random weights are fine: comparing two scoring paths
    texts = [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "SHORT",
        "A MUCH LONGER TEXT THAT SPANS MULTIPLE BLOCK SIZE WINDOWS "
        "BECAUSE IT KEEPS GOING AND GOING AND GOING WELL PAST THIRTY TWO CHARS",
        "AB",
    ]
    batch = lm.score_batch(texts)
    loop = [lm.score(t) for t in texts]
    assert len(batch) == len(texts)
    for b, l in zip(batch, loop):
        assert abs(float(b) - l) < 1e-5

def test_score_batch_handles_empty_and_tiny():
    torch.manual_seed(0)
    lm = CharLM(TINY)
    batch = lm.score_batch(["", "A", "HELLO WORLD"])
    assert float(batch[0]) == 0.0
    assert float(batch[1]) == 0.0
    assert abs(float(batch[2]) - lm.score("HELLO WORLD")) < 1e-5
