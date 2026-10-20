"""Substitution-key utilities and text normalization."""
from __future__ import annotations
import random
import re

A = ord("A")
_NON_AZ = re.compile(r"[^A-Z ]+")
_SPACES = re.compile(r" +")


def clean_text(text: str) -> str:
    """Normalize to uppercase A-Z separated by single spaces."""
    up = text.upper()
    up = up.replace("-", " ")
    up = _NON_AZ.sub("", up)
    up = _SPACES.sub(" ", up)
    return up.strip()


def identity_key() -> tuple[int, ...]:
    return tuple(range(26))


def random_key(rng: random.Random) -> tuple[int, ...]:
    vals = list(range(26))
    rng.shuffle(vals)
    return tuple(vals)


def swap(key: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    k = list(key)
    k[i], k[j] = k[j], k[i]
    return tuple(k)


def decode(cipher_text: str, key: tuple[int, ...]) -> str:
    """Map each cipher letter through key; spaces pass through."""
    out = []
    for ch in cipher_text:
        if ch == " ":
            out.append(" ")
        else:
            out.append(chr(A + key[ord(ch) - A]))
    return "".join(out)


def encode(plain_text: str, key: tuple[int, ...]) -> str:
    """Inverse of decode: apply key^-1 so decode(encode(x,key),key)==x."""
    inv = [0] * 26
    for c, p in enumerate(key):
        inv[p] = c
    out = []
    for ch in plain_text:
        if ch == " ":
            out.append(" ")
        else:
            out.append(chr(A + inv[ord(ch) - A]))
    return "".join(out)
