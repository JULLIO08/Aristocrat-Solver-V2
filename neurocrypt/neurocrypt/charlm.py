"""Char-level decoder-only Transformer language model."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device(device: str) -> str:
    """Map 'auto' to the best available torch device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Vocab: 0 = PAD, 1 = space, 2..27 = A..Z
PAD = 0
SPACE = 1
VOCAB_SIZE = 28


def _ctoi(ch: str) -> int:
    if ch == " ":
        return SPACE
    return ord(ch) - ord("A") + 2


def _itoc(i: int) -> str:
    if i == SPACE:
        return " "
    if i == PAD:
        return ""
    return chr(i - 2 + ord("A"))


@dataclass
class CharLMConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    block_size: int = 64
    dropout: float = 0.1


class _Block(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class _Net(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


class CharLM:
    def __init__(self, cfg: CharLMConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.net = _Net(cfg).to(device)

    def encode(self, text: str) -> list[int]:
        # Normalize case: vocab is uppercase A-Z. All pipeline text passes through
        # clean_text() (uppercase), but encode is defensive so lowercase input works too.
        return [_ctoi(c) for c in text.upper()]

    def decode(self, ids: list[int]) -> str:
        return "".join(_itoc(i) for i in ids)

    def _windows(self, text: str) -> list[list[int]]:
        bs = self.cfg.block_size
        ids = self.encode(text)
        out = []
        for i in range(0, max(1, len(ids) - 1), bs):
            chunk = ids[i:i + bs + 1]
            if len(chunk) < 2:
                continue
            out.append(chunk)
        return out

    def fit(self, lines: list[str], epochs: int = 1, batch_size: int = 64,
            lr: float = 3e-4, device: str | None = None) -> list[float]:
        dev = device or self.device
        self.net.to(dev)
        self.device = dev
        bs = self.cfg.block_size
        # Build fixed-length training windows from concatenated corpus.
        stream = " ".join(lines)
        ids = self.encode(stream)
        samples = []
        for i in range(0, len(ids) - bs - 1, bs):
            samples.append(ids[i:i + bs + 1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        losses = []
        self.net.train()
        for _ in range(epochs):
            torch.manual_seed(torch.randint(0, 1 << 30, (1,)).item())
            perm = torch.randperm(len(samples))
            epoch_loss, nb = 0.0, 0
            for s in range(0, len(samples), batch_size):
                batch = [samples[j] for j in perm[s:s + batch_size].tolist()]
                arr = torch.tensor(batch, dtype=torch.long, device=dev)
                x, y = arr[:, :-1], arr[:, 1:]
                logits = self.net(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                nb += 1
            losses.append(epoch_loss / max(1, nb))
        return losses

    @torch.no_grad()
    def score(self, text: str) -> float:
        """Average per-character log-likelihood (higher = more English)."""
        self.net.eval()
        ids = self.encode(text)
        if len(ids) < 2:
            return 0.0
        bs = self.cfg.block_size
        total, n = 0.0, 0
        for i in range(0, len(ids) - 1, bs):
            chunk = ids[i:i + bs + 1]
            if len(chunk) < 2:
                continue
            arr = torch.tensor([chunk], dtype=torch.long, device=self.device)
            x, y = arr[:, :-1], arr[:, 1:]
            logits = self.net(x)
            logp = F.log_softmax(logits, dim=-1)
            tok_lp = logp[0, torch.arange(y.shape[1]), y[0]]
            total += tok_lp.sum().item()
            n += y.shape[1]
        return total / max(1, n)

    @torch.no_grad()
    def score_batch(self, texts: list[str], max_rows: int = 256) -> np.ndarray:
        """Average per-char log-likelihood of each text via padded batched forwards.

        Equivalent to [score(t) for t in texts] but with one forward per
        max_rows chunk rows instead of one per text window. Padding is safe:
        the attention mask is causal, so real positions never see pad tokens,
        and pad targets are excluded from the average.
        """
        self.net.eval()
        bs = self.cfg.block_size
        chunks: list[list[int]] = []
        owner: list[int] = []
        for ti, text in enumerate(texts):
            ids = self.encode(text)
            for i in range(0, max(0, len(ids) - 1), bs):
                chunk = ids[i:i + bs + 1]
                if len(chunk) >= 2:
                    chunks.append(chunk)
                    owner.append(ti)
        totals = np.zeros(len(texts), dtype=np.float64)
        counts = np.zeros(len(texts), dtype=np.float64)
        for start in range(0, len(chunks), max_rows):
            part = chunks[start:start + max_rows]
            maxlen = max(len(c) for c in part)
            arr = torch.full((len(part), maxlen), PAD, dtype=torch.long,
                             device=self.device)
            for r, c in enumerate(part):
                arr[r, : len(c)] = torch.tensor(c, dtype=torch.long,
                                                device=self.device)
            x, y = arr[:, :-1], arr[:, 1:]
            logp = F.log_softmax(self.net(x), dim=-1)
            tok_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            mask = y != PAD
            sums = (tok_lp * mask).sum(dim=1).cpu().numpy()
            ns = mask.sum(dim=1).cpu().numpy()
            for r, row_sum in enumerate(sums):
                ti = owner[start + r]
                totals[ti] += row_sum
                counts[ti] += ns[r]
        return totals / np.maximum(counts, 1.0)

    def save(self, path) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "CharLM":
        ckpt = torch.load(path, map_location=device)
        lm = cls(CharLMConfig(**ckpt["cfg"]), device=device)
        lm.net.load_state_dict(ckpt["state"])
        lm.net.to(device)
        return lm
