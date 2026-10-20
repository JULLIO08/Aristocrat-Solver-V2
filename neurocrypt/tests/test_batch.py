"""`neurocrypt batch FILE`: solve every ciphertext line in a file."""
import json
import random
from pathlib import Path

import pytest

from neurocrypt.cli import batch_lines, main
from neurocrypt.cipher import clean_text, encode, random_key

MODELS = Path(__file__).resolve().parent.parent / "models"

needs_models = pytest.mark.skipif(
    not (MODELS / "charlm.pt").exists(),
    reason="trained models not present (gitignored artifact)",
)


def test_batch_lines_keeps_nonempty_cleanable_lines():
    raw = "WKH TXLFN\n\n   \nEURZQ IRA\n123 !!!\nDET\n"
    lines = batch_lines(raw)
    # blank lines and the punctuation/digits-only line drop out
    assert lines == ["WKH TXLFN", "EURZQ IRA", "DET"]


def test_batch_lines_empty_file_returns_empty():
    assert batch_lines("") == []
    assert batch_lines("\n\n   \n") == []


def test_cli_batch_missing_file_friendly_error(capsys):
    rc = main(["batch", "definitely_not_a_file.txt"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "cannot read" in err


def test_cli_batch_empty_file_friendly_error(tmp_path, capsys):
    f = tmp_path / "empty.txt"
    f.write_text("\n   \n!!!\n", encoding="utf-8")
    rc = main(["batch", str(f)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no ciphertext" in err.lower()


@needs_models
def test_cli_batch_solves_every_line(tmp_path, capsys):
    # Build a file of 3 ciphertexts (each its own substitution).
    plains = [
        clean_text("the quick brown fox jumps over the lazy dog every morning"),
        clean_text("a journey of a thousand miles begins with a single step forward"),
        clean_text("knowledge is power and power tends to corrupt those who hold it"),
    ]
    ciphers = [encode(p, random_key(random.Random(i))) for i, p in enumerate(plains)]
    infile = tmp_path / "ciphers.txt"
    infile.write_text("\n".join(ciphers) + "\n", encoding="utf-8")

    out_json = tmp_path / "out.json"
    out_txt = tmp_path / "solved.txt"
    rc = main(["batch", str(infile),
               "--json", str(out_json), "--out", str(out_txt),
               "--restarts", "4", "--iters", "800",
               "--escalation-rounds", "0"])
    assert rc == 0

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["count"] == 3
    assert len(data["results"]) == 3
    for rec, cipher in zip(data["results"], ciphers):
        assert rec["ciphertext"] == cipher
        assert len(rec["plaintext"]) == len(cipher)
        assert len(rec["key"]) == 26
        for field in ("line", "score", "confidence", "escalations"):
            assert field in rec

    # --out holds one decoded line per input cipher, in order
    solved = out_txt.read_text(encoding="utf-8").splitlines()
    assert len(solved) == 3
    assert solved[0] == data["results"][0]["plaintext"]

    printed = capsys.readouterr().out
    assert "[1/3]" in printed and "[3/3]" in printed
