import random

from neurocrypt.cipher import decode, encode, random_key, clean_text
from neurocrypt.ngramfast import QuadgramModel
from neurocrypt.search import anneal, search_topk, frequency_seed_key, seed_variants

ENGLISH_LINES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES",
    "TO BE OR NOT TO BE THAT IS THE QUESTION",
    "ALL HAPPY FAMILIES ARE ALIKE EACH UNHAPPY FAMILY IS UNHAPPY IN ITS OWN WAY",
    "THE ONLY THING WE HAVE TO FEAR IS FEAR ITSELF",
    "IN THE BEGINNING WAS THE WORD AND THE WORD WAS WITH GOD",
    "ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU",
    "A JOURNEY OF A THOUSAND MILES BEGINS WITH A SINGLE STEP",
]

def test_recovers_simple_cipher_with_perfect_scorer():
    plain = clean_text("the quick brown fox jumps over the lazy dog")
    key = random_key(random.Random(3))
    cipher = encode(plain, key)

    def oracle(text: str) -> float:
        return sum(1 for a, b in zip(text, plain) if a == b)

    best_key, best_score = anneal(
        cipher, oracle, restarts=8, iters=2000, seed=7,
    )
    assert decode(cipher, best_key) == plain

def test_returns_best_seen_not_last():
    plain = clean_text("hello world")
    key = random_key(random.Random(1))
    cipher = encode(plain, key)

    def oracle(text):
        return sum(1 for a, b in zip(text, plain) if a == b)

    best_key, best_score = anneal(cipher, oracle, restarts=4, iters=500, seed=2)
    assert best_score == oracle(decode(cipher, best_key))


def _make_cipher(seed=3):
    plain = clean_text(" ".join(ENGLISH_LINES[:5]))
    key = random_key(random.Random(seed))
    return plain, encode(plain, key)


def test_search_topk_recovers_cipher_with_quadgram_model():
    from pathlib import Path
    from neurocrypt.corpus import load_lines

    fix = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
    lines = load_lines(fix)
    qmodel = QuadgramModel.train(lines * 40)
    plain = clean_text(" ".join(lines[:6]))
    key = random_key(random.Random(1))
    cipher = encode(plain, key)
    results = search_topk(cipher, qmodel, restarts=12, iters=8000, k=8, seed=5)
    best_key, best_score = results[0]
    errors = sum(1 for a, b in zip(decode(cipher, best_key), plain) if a != b)
    assert errors <= 2, f"{errors} errors: {decode(cipher, best_key)!r}"


def test_search_topk_returns_distinct_sorted_candidates():
    qmodel = QuadgramModel.train(ENGLISH_LINES)
    _, cipher = _make_cipher()
    results = search_topk(cipher, qmodel, restarts=8, iters=2000, k=6, seed=1)
    assert 1 <= len(results) <= 6
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    decodes = {decode(cipher, k) for k, _ in results}
    assert len(decodes) == len(results)  # distinct decode strings


def test_search_topk_deterministic():
    qmodel = QuadgramModel.train(ENGLISH_LINES)
    _, cipher = _make_cipher()
    a = search_topk(cipher, qmodel, restarts=4, iters=1500, k=4, seed=9)
    b = search_topk(cipher, qmodel, restarts=4, iters=1500, k=4, seed=9)
    assert a == b


def test_frequency_seed_key_is_permutation():
    plain, cipher = _make_cipher()
    key = frequency_seed_key(cipher)
    assert sorted(key) == list(range(26))


def test_frequency_seed_maps_most_frequent_to_e():
    # cipher where Q is by far the most frequent letter -> Q should decode to E
    cipher = "QQQQ QQQQ QXQZ QQKQ"
    key = frequency_seed_key(cipher)
    assert decode("Q", key) == "E"


def test_seed_variants_cover_a_and_i_for_single_letter_words():
    # 'X' appears as a single-letter cipher word -> variants map X to A and I
    cipher = "X QXRS TXLM X BNOP"
    variants = seed_variants(cipher)
    assert all(sorted(v) == list(range(26)) for v in variants)
    decoded_singles = {decode("X", v) for v in variants}
    assert {"A", "I"} <= decoded_singles


def test_search_topk_uses_seed_keys():
    """A provided correct seed key must survive into the results."""
    qmodel = QuadgramModel.train(ENGLISH_LINES)
    plain, cipher = _make_cipher()
    true_key = None
    # recover the true key by brute construction: decode with identity then map
    for seed in range(100):
        k = random_key(random.Random(3))
        if decode(cipher, k) == plain:
            true_key = k
            break
    assert true_key is not None
    results = search_topk(cipher, qmodel, restarts=2, iters=50, k=4, seed=0,
                          seed_keys=[true_key])
    best_key, _ = results[0]
    assert decode(cipher, best_key) == plain
