"""Command-line interface: train / solve / eval."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

from .corpus import load_lines, train_eval_split, DEFAULT_PLAIN, DEFAULT_ENCODED
from .cipher import clean_text, decode
from .charlm import CharLM, CharLMConfig, resolve_device
from .wordlm import WordLM
from .lexicon import Lexicon
from .postag import POSModel
from .ngramfast import QuadgramModel
from .patterns import PatternIndex
from .scorer import Scorer, Weights
from .solver import Solver

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = PKG_ROOT / "models"


def cmd_train(args) -> int:
    models = Path(args.models)
    models.mkdir(parents=True, exist_ok=True)
    sources = args.corpus or [str(DEFAULT_PLAIN)]
    lines: list[str] = []
    for src in sources:
        lines.extend(load_lines(Path(src)))
    train_lines, _ = train_eval_split(lines, eval_frac=0.05, seed=42)
    print(f"Training on {len(train_lines)} lines from {len(sources)} source(s).")

    device = resolve_device(args.device)
    cfg = CharLMConfig(d_model=args.dim, n_heads=args.heads,
                       n_layers=args.layers, block_size=args.block)
    charlm = CharLM(cfg, device=device)
    losses = charlm.fit(train_lines, epochs=args.epochs,
                        batch_size=args.batch, lr=args.lr, device=device)
    print(f"Char-LM final loss: {losses[-1]:.4f}")
    charlm.save(models / "charlm.pt")

    QuadgramModel.train(train_lines).save(models / "quadgrams.npy")
    PatternIndex.from_corpus(train_lines, min_count=2).save(models / "patterns.pkl")
    WordLM.train(train_lines).save(models / "wordlm.json")
    (models / "lexicon.json").write_text(
        json.dumps(Lexicon.from_corpus(train_lines, min_count=2).to_list()),
        encoding="utf-8",
    )
    pos_sample = train_lines[: args.pos_lines]
    POSModel.train(pos_sample).save(models / "postag.json")
    Weights().save(models / "weights.json")
    print(f"Saved models to {models}")
    return 0


def cmd_solve(args) -> int:
    if args.file:
        try:
            cipher = Path(args.file).read_text(encoding="utf-8")
        except OSError as e:
            print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
            return 2
    else:
        cipher = args.ciphertext
    if not clean_text(cipher):
        print("error: ciphertext is empty after normalization "
              "(only A-Z letters and spaces are kept)", file=sys.stderr)
        return 2
    solver = Solver.from_models(Path(args.models), device=args.device)
    result = solver.solve(clean_text(cipher), restarts=args.restarts,
                          iters=args.iters, seed=args.seed,
                          legacy=args.legacy_search,
                          target_confidence=args.target_confidence,
                          escalation_rounds=args.escalation_rounds,
                          max_restarts=args.max_restarts,
                          pattern_seed=args.pattern_seed)
    payload = {
        "plaintext": result.plaintext,
        "key": "".join(chr(ord("A") + p) for p in result.key),
        "score": result.score,
        "confidence": result.confidence,
        "escalations": result.escalations,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(result.plaintext)
        print(f"\nconfidence: {result.confidence:.2%}  score: {result.score:.3f}")
        if args.topk > 1:
            print(f"\ntop {args.topk} candidates:")
            for text, s in solver.candidates(cipher, restarts=args.restarts,
                                             iters=args.iters, seed=args.seed,
                                             k=args.topk):
                print(f"  {s: .3f}  {text[:70]}")
    if args.save_result:
        Path(args.save_result).write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")
    return 0


def batch_lines(text: str) -> list[str]:
    """Split a file's text into ciphertext lines worth solving.

    Keeps the original (uncleaned) line content but drops any line that is empty
    after normalization (blank lines, whitespace-only, punctuation/digits-only).
    """
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and clean_text(line):
            out.append(line)
    return out


def cmd_batch(args) -> int:
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    ciphers = batch_lines(text)
    if not ciphers:
        print(f"error: no ciphertext lines found in {args.file} "
              "(only A-Z letters and spaces are kept)", file=sys.stderr)
        return 2

    solver = Solver.from_models(Path(args.models), device=args.device)
    results = []
    solved_lines = []
    n = len(ciphers)
    for k, cipher in enumerate(ciphers):
        res = solver.solve(clean_text(cipher), restarts=args.restarts,
                           iters=args.iters, seed=args.seed,
                           legacy=args.legacy_search,
                           target_confidence=args.target_confidence,
                           escalation_rounds=args.escalation_rounds,
                           max_restarts=args.max_restarts,
                           pattern_seed=args.pattern_seed)
        results.append({
            "line": k + 1,
            "ciphertext": cipher,
            "plaintext": res.plaintext,
            "key": "".join(chr(ord("A") + p) for p in res.key),
            "score": res.score,
            "confidence": res.confidence,
            "escalations": res.escalations,
        })
        solved_lines.append(res.plaintext)
        if not args.quiet:
            print(f"[{k+1}/{n}] conf={res.confidence:.2%}  {res.plaintext}")

    solved = sum(1 for r in results if r["confidence"] >= args.target_confidence)
    if not args.quiet:
        print(f"\n{n} ciphers solved; {solved} at or above "
              f"{args.target_confidence:.0%} confidence.")
    if args.out:
        Path(args.out).write_text("\n".join(solved_lines) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"plaintext -> {args.out}")
    if args.json:
        payload = {"count": n, "results": results,
                   "config": {"restarts": args.restarts, "iters": args.iters,
                              "legacy_search": args.legacy_search}}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"JSON -> {args.json}")
    return 0


def cmd_eval(args) -> int:
    import time

    from .corpus import eval_split_indices
    from .evalreport import bucket_of, summarize

    plains = load_lines(DEFAULT_PLAIN)
    encs = load_lines(DEFAULT_ENCODED)
    n = min(len(plains), len(encs))
    if args.holdout:
        idx = eval_split_indices(n)
    else:
        idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n]
    solver = Solver.from_models(Path(args.models), device=args.device)
    records = []
    for k, i in enumerate(idx):
        plain, cipher = plains[i], encs[i]
        t0 = time.perf_counter()
        res = solver.solve(cipher, restarts=args.restarts, iters=args.iters,
                           seed=i, legacy=args.legacy_search,
                           target_confidence=args.target_confidence,
                           escalation_rounds=args.escalation_rounds,
                           max_restarts=args.max_restarts,
                           pattern_seed=args.pattern_seed)
        wall = time.perf_counter() - t0
        errs = sum(1 for a, b in zip(res.plaintext, plain) if a != b)
        records.append({
            "index": i, "length": len(plain), "bucket": bucket_of(len(plain)),
            "errors": errs, "error_rate": errs / max(1, len(plain)),
            "confidence": res.confidence, "wall_time": wall,
            "escalations": res.escalations,
        })
        if not args.quiet:
            print(f"[{k+1}/{len(idx)}] errors={errs:3d} conf={res.confidence:.2%}")
    summary = summarize(records, seed=args.seed)
    rate = summary["mean_error_rate"]
    print(f"\nExact solves: {sum(1 for r in records if r['errors'] == 0)}"
          f"/{len(records)} ({summary['exact_solve_rate']:.0%})")
    print(f"Mean letter-error rate: {rate:.4f} over {len(idx)} ciphers")
    if args.json:
        payload = {"records": records, "summary": summary,
                   "config": {"n": args.n, "seed": args.seed,
                              "restarts": args.restarts, "iters": args.iters,
                              "holdout": args.holdout,
                              "legacy_search": args.legacy_search}}
        Path(args.json).write_text(json.dumps(payload, indent=2),
                                   encoding="utf-8")
        print(f"JSON -> {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurocrypt", description="ML/NLP cipher solver")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train models on the corpus")
    t.add_argument("--corpus", action="append", help="extra/override corpus file(s)")
    t.add_argument("--models", default=str(DEFAULT_MODELS))
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--block", type=int, default=64)
    t.add_argument("--pos-lines", type=int, default=4000)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("solve", help="solve a ciphertext")
    s.add_argument("ciphertext", nargs="?", default="")
    s.add_argument("--file")
    s.add_argument("--models", default=str(DEFAULT_MODELS))
    s.add_argument("--restarts", type=int, default=10)
    s.add_argument("--iters", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cpu")
    s.add_argument("--legacy-search", action="store_true",
                   help="bypass the fast two-tier pipeline (old behavior)")
    s.add_argument("--json", action="store_true",
                   help="print the result as JSON")
    s.add_argument("--save-result", help="also write the result JSON here")
    s.add_argument("--topk", type=int, default=1,
                   help="print N runner-up decodes with scores")
    _add_escalation_args(s)
    s.set_defaults(func=cmd_solve)

    b = sub.add_parser("batch",
                       help="solve every ciphertext line in a text file")
    b.add_argument("file", help="text file with one ciphertext per line")
    b.add_argument("--models", default=str(DEFAULT_MODELS))
    b.add_argument("--restarts", type=int, default=10)
    b.add_argument("--iters", type=int, default=4000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="cpu")
    b.add_argument("--legacy-search", action="store_true",
                   help="bypass the fast two-tier pipeline (old behavior)")
    b.add_argument("--out", help="write one decoded plaintext per line to this file")
    b.add_argument("--json", help="write per-cipher records + config to this file")
    b.add_argument("--quiet", action="store_true",
                   help="suppress per-cipher progress lines")
    _add_escalation_args(b)
    b.set_defaults(func=cmd_batch)

    e = sub.add_parser("eval", help="evaluate on held-out encoded quotes")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--seed", type=int, default=123)
    e.add_argument("--models", default=str(DEFAULT_MODELS))
    e.add_argument("--restarts", type=int, default=10)
    e.add_argument("--iters", type=int, default=4000)
    e.add_argument("--device", default="cpu")
    e.add_argument("--legacy-search", action="store_true",
                   help="bypass the fast two-tier pipeline (old behavior)")
    e.add_argument("--json", help="write per-cipher records + summary to this file")
    e.add_argument("--holdout", action="store_true",
                   help="sample only from the never-trained 5%% eval split")
    e.add_argument("--quiet", action="store_true",
                   help="suppress per-cipher progress lines")
    _add_escalation_args(e)
    e.set_defaults(func=cmd_eval)
    return p


def _add_escalation_args(p) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pattern-seed", dest="pattern_seed", action="store_true",
                   default=None, help="force pattern-attack seeding on")
    g.add_argument("--no-pattern-seed", dest="pattern_seed",
                   action="store_false",
                   help="disable pattern-attack seeding (default: auto)")
    p.add_argument("--target-confidence", type=float, default=0.90,
                   help="escalate while confidence is below this")
    p.add_argument("--escalation-rounds", type=int, default=2,
                   help="max escalation rounds (0 disables)")
    p.add_argument("--max-restarts", type=int, default=64,
                   help="restart cap during escalation")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
