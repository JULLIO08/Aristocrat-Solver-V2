"""Orchestrate scoring + search into a solve."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

from .cipher import clean_text, decode
from .charlm import CharLM
from .wordlm import WordLM
from .lexicon import Lexicon
from .ngramfast import DeltaScorer, QuadgramModel
from .patterns import PatternIndex
from .postag import POSModel
from .scorer import Scorer, Weights
from .search import anneal, search_topk


def _unfrozen_letters(cipher: str, plain: str, lexicon: Lexicon) -> set[int]:
    """Cipher letters NOT locked by dictionary evidence.

    A letter is frozen when every decoded word containing it is a dictionary
    word. Letters in any non-dictionary word, plus letters absent from the
    cipher entirely, stay free to move.
    """
    in_bad: set[int] = set()
    in_good: set[int] = set()
    for cw, pw in zip(cipher.split(), plain.split()):
        target = in_good if pw.lower() in lexicon.words else in_bad
        for ch in cw:
            target.add(ord(ch) - ord("A"))
    absent = set(range(26)) - in_bad - in_good
    return in_bad | absent


@dataclass
class SolveResult:
    plaintext: str
    key: tuple[int, ...]
    score: float
    confidence: float
    escalations: int = 0


@dataclass
class Calibration:
    """Platt-style logistic map from (lexicon coverage, fused score) to P(solved)."""
    coef_coverage: float
    coef_score: float
    intercept: float

    def probability(self, coverage: float, score: float) -> float:
        z = (self.coef_coverage * coverage + self.coef_score * score
             + self.intercept)
        return 1.0 / (1.0 + math.exp(-z))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class Solver:
    def __init__(self, scorer: Scorer, lexicon: Lexicon,
                 qmodel: QuadgramModel | None = None,
                 calibration: Calibration | None = None,
                 patterns: PatternIndex | None = None):
        self.scorer = scorer
        self.lexicon = lexicon
        self.qmodel = qmodel
        self.calibration = calibration
        self.patterns = patterns

    @classmethod
    def from_models(cls, models_dir: Path, device: str = "cpu") -> "Solver":
        models_dir = Path(models_dir)
        for f in ["charlm.pt", "wordlm.json", "lexicon.json", "postag.json", "weights.json"]:
            if not (models_dir / f).exists():
                raise FileNotFoundError(
                    f"Missing {f} in {models_dir}. Run `neurocrypt train` first."
                )
        import json
        from .charlm import resolve_device
        device = resolve_device(device)
        charlm = CharLM.load(models_dir / "charlm.pt", device=device)
        wordlm = WordLM.load(models_dir / "wordlm.json")
        words = set(json.loads((models_dir / "lexicon.json").read_text(encoding="utf-8")))
        lexicon = Lexicon(words)
        postag = POSModel.load(models_dir / "postag.json")
        weights = Weights.load(models_dir / "weights.json")
        scorer = Scorer(charlm, wordlm, lexicon, postag, weights)
        qpath = models_dir / "quadgrams.npy"
        qmodel = QuadgramModel.load(qpath) if qpath.exists() else None
        cpath = models_dir / "calibration.json"
        calibration = Calibration.load(cpath) if cpath.exists() else None
        ppath = models_dir / "patterns.pkl"
        patterns = PatternIndex.load(ppath) if ppath.exists() else None
        return cls(scorer, lexicon, qmodel=qmodel, calibration=calibration,
                   patterns=patterns)

    def solve(self, cipher_text: str, restarts: int = 10, iters: int = 4000,
              seed: int = 0, legacy: bool = False, topk: int = 16,
              seed_keys: list[tuple[int, ...]] | None = None,
              target_confidence: float = 0.90,
              escalation_rounds: int = 2,
              max_restarts: int = 64,
              pattern_seed: bool | None = None) -> SolveResult:
        cipher = clean_text(cipher_text)
        if legacy or self.qmodel is None:
            key, score = anneal(cipher, self.scorer.score, restarts=restarts,
                                iters=iters, seed=seed)
            return self._finish(cipher, key, score, escalations=0)

        use_patterns = self.patterns is not None and (
            pattern_seed if pattern_seed is not None
            else (len(cipher) < 120 and len(cipher.split()) >= 3))
        p_seeds = self._pattern_seeds(cipher, restarts) if use_patterns else []
        combined = (list(seed_keys or []) + p_seeds) or None
        key, score = self._solve_two_tier(cipher, restarts, iters, seed,
                                          topk, combined)
        result = self._finish(cipher, key, score, escalations=0)
        rounds = 0
        b_restarts, b_iters = restarts, iters
        while result.confidence < target_confidence and rounds < escalation_rounds:
            rounds += 1
            if rounds == 1 or len(_unfrozen_letters(cipher, result.plaintext,
                                                    self.lexicon)) < 2:
                # round 1: doubled budget, seeded from the best key so far
                # plus fresh pattern seeds from a wider beam
                b_restarts = min(b_restarts * 2, max_restarts)
                b_iters *= 2
                wide = (self._pattern_seeds(cipher, b_restarts, widen=2)
                        if use_patterns else [])
                key, score = self._solve_two_tier(
                    cipher, b_restarts, b_iters, seed + rounds, topk,
                    [result.key] + wide)
            else:
                # later rounds: freeze letters that only occur in dictionary
                # words and re-anneal the small remainder
                key, score = self._reanneal_unfrozen(cipher, result,
                                                     seed + rounds)
            cand = self._finish(cipher, key, score, escalations=rounds)
            if cand.score > result.score or cand.confidence > result.confidence:
                result = SolveResult(cand.plaintext, cand.key, cand.score,
                                     cand.confidence, escalations=rounds)
            else:
                result = SolveResult(result.plaintext, result.key, result.score,
                                     result.confidence, escalations=rounds)
        return result

    def candidates(self, cipher_text: str, restarts: int = 10,
                   iters: int = 4000, seed: int = 0, k: int = 5
                   ) -> list[tuple[str, float]]:
        """Top-k candidate decodes with fused scores, best first."""
        if self.qmodel is None:
            raise ValueError("candidates() requires a quadgram model")
        cipher = clean_text(cipher_text)
        cands = search_topk(cipher, self.qmodel, restarts=restarts,
                            iters=iters, k=k, seed=seed)
        decodes = [decode(cipher, key) for key, _ in cands]
        ranked = self.scorer.rerank(decodes)
        return [(decodes[i], s) for i, s in ranked]

    def _pattern_seeds(self, cipher: str, restarts: int,
                       widen: int = 1) -> list[tuple[int, ...]]:
        """Pattern-attack seed keys ranked by quadgram score, best first.

        Capped at half the chains so frequency-seeded and random restarts
        keep their diversity role.
        """
        seeds = self.patterns.seed_keys(
            cipher, max_seeds=200 * widen, branch=60 * widen,
            time_budget=1.0 * widen)
        if not seeds:
            return []
        ds = DeltaScorer(cipher, self.qmodel)
        scored = sorted(((ds.set_key(k), k) for k in seeds), reverse=True)
        cap = max(1, restarts // 2)
        return [k for _, k in scored[:cap]]

    def _finish(self, cipher: str, key: tuple[int, ...], score: float,
                escalations: int) -> SolveResult:
        plain = decode(cipher, key)
        coverage = self.lexicon.score(plain)
        if self.calibration is not None:
            confidence = self.calibration.probability(coverage=coverage, score=score)
        else:
            confidence = coverage
        return SolveResult(plaintext=plain, key=key, score=score,
                           confidence=confidence, escalations=escalations)

    def _reanneal_unfrozen(self, cipher: str, result: SolveResult,
                           seed: int) -> tuple[tuple[int, ...], float]:
        """Re-anneal only the cipher letters not locked by dictionary words.

        Swapping within the unfrozen set turns the 26-symbol problem into a
        handful of symbols, which a short focused anneal solves reliably.
        """
        import math as _math
        import random as _random

        from .ngramfast import DeltaScorer

        unfrozen = sorted(_unfrozen_letters(cipher, result.plaintext, self.lexicon))
        ds = DeltaScorer(cipher, self.qmodel)
        best_key = list(result.key)
        ds.set_key(tuple(best_key))
        best_score = ds.current_score
        rng = _random.Random(seed)
        cur_score = best_score
        temp, t_end, iters = 0.05, 0.005, 3000
        cooling = (t_end / temp) ** (1.0 / iters)
        cur_key = list(best_key)
        for _ in range(iters):
            i, j = rng.choice(unfrozen), rng.choice(unfrozen)
            if i == j:
                temp *= cooling
                continue
            d = ds.delta(i, j)
            if d >= 0 or rng.random() < _math.exp(d / max(temp, 1e-12)):
                ds.apply(i, j)
                cur_score += d
                cur_key[i], cur_key[j] = cur_key[j], cur_key[i]
                if cur_score > best_score:
                    best_score, best_key = cur_score, list(cur_key)
            temp *= cooling
        # rerank old vs new with the fused scorer
        cands = [tuple(result.key), tuple(best_key)]
        decodes = [decode(cipher, k) for k in cands]
        ranked = self.scorer.rerank(decodes)
        idx, fused = ranked[0]
        return cands[idx], fused

    def _solve_two_tier(self, cipher: str, restarts: int, iters: int,
                        seed: int, topk: int,
                        seed_keys: list[tuple[int, ...]] | None
                        ) -> tuple[tuple[int, ...], float]:
        """Stage 1: fast quadgram SA emits top-K keys. Stage 2: neural rerank."""
        candidates = search_topk(cipher, self.qmodel, restarts=restarts,
                                 iters=iters, k=topk, seed=seed,
                                 seed_keys=seed_keys)
        decodes = [decode(cipher, key) for key, _ in candidates]
        ranked = self.scorer.rerank(decodes)
        best_idx, best_fused = ranked[0]
        return candidates[best_idx][0], best_fused
