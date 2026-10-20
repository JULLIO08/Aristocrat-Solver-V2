from neurocrypt.lexicon import Lexicon

WORDS = {"the", "quick", "brown", "fox", "is", "a", "word"}

def test_coverage_all_known_is_one():
    lex = Lexicon(WORDS)
    assert lex.score("THE QUICK FOX") == 1.0

def test_coverage_partial():
    lex = Lexicon(WORDS)
    assert lex.score("THE ZZZZ FOX QQQQ") == 0.5

def test_empty_text_scores_zero():
    lex = Lexicon(WORDS)
    assert lex.score("") == 0.0

def test_ranks_real_over_garbage():
    lex = Lexicon(WORDS)
    assert lex.score("THE QUICK BROWN FOX") > lex.score("XQZ Wbr FoO bAr")

def test_from_corpus_builds_vocab():
    lex = Lexicon.from_corpus(["THE FOX", "A WORD"])
    assert lex.score("THE WORD") == 1.0
