"""Quick A/B of search_topk features on the fixture cipher."""
import random
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from neurocrypt.cipher import clean_text, decode, encode, random_key  # noqa: E402
from neurocrypt.corpus import load_lines  # noqa: E402
from neurocrypt.ngramfast import QuadgramModel  # noqa: E402
from neurocrypt.search import search_topk  # noqa: E402


def run(tag, **kw):
    lines = load_lines(PKG_ROOT / "tests" / "fixtures" / "tiny_corpus.txt")
    qmodel = QuadgramModel.train(lines * 40)
    plain = clean_text(" ".join(lines[:6]))
    cipher = encode(plain, random_key(random.Random(1)))
    results = search_topk(cipher, qmodel, restarts=12, iters=8000, k=8, seed=5, **kw)
    best_key, _ = results[0]
    errors = sum(1 for a, b in zip(decode(cipher, best_key), plain) if a != b)
    print(f"{tag:40s} errors={errors}")


if __name__ == "__main__":
    run("baseline (no new features)", freq_seed=False, bias_prob=0.0, reheat_factor=1.0)
    run("freq_seed only", freq_seed=True, bias_prob=0.0, reheat_factor=1.0)
    run("bias only", freq_seed=False, bias_prob=0.7, reheat_factor=1.0)
    run("reheat only", freq_seed=False, bias_prob=0.0, reheat_factor=3.0)
    run("all on", freq_seed=True, bias_prob=0.7, reheat_factor=3.0)
