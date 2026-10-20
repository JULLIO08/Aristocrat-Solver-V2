import random
from neurocrypt.cipher import clean_text, decode, identity_key, random_key, swap, encode

def test_clean_text_normalizes_to_az_and_space():
    assert clean_text("Hello, World!  It's  fine.") == "HELLO WORLD ITS FINE"

def test_clean_text_strips_non_ascii():
    assert clean_text("café “quote”") == "CAF QUOTE"

def test_identity_key_decodes_to_same_letters():
    assert decode("ABC XYZ", identity_key()) == "ABC XYZ"

def test_encode_then_decode_roundtrips():
    key = random_key(random.Random(0))
    plain = "THE QUICK BROWN FOX"
    cipher = encode(plain, key)
    assert decode(cipher, key) == plain

def test_swap_changes_exactly_two_positions():
    key = identity_key()
    swapped = swap(key, 0, 1)
    assert swapped[0] == 1 and swapped[1] == 0
    assert swapped[2:] == key[2:]

def test_random_key_is_a_permutation():
    key = random_key(random.Random(1))
    assert sorted(key) == list(range(26))
