"""CLI output flags: solve --json / --save-result, eval --quiet."""
import json
from pathlib import Path

import pytest

from neurocrypt.cli import main

MODELS = Path(__file__).resolve().parent.parent / "models"

needs_models = pytest.mark.skipif(
    not (MODELS / "charlm.pt").exists(),
    reason="trained models not present (gitignored artifact)",
)


@needs_models
def test_solve_json_and_save_result(tmp_path, capsys):
    out_file = tmp_path / "result.json"
    rc = main(["solve", "WKH TXLFN EURZQ IRA", "--json",
               "--save-result", str(out_file),
               "--restarts", "2", "--iters", "300",
               "--escalation-rounds", "0"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert printed == saved
    for field in ("plaintext", "key", "score", "confidence", "escalations"):
        assert field in printed
    assert len(printed["key"]) == 26


@needs_models
def test_eval_quiet_suppresses_progress(capsys):
    rc = main(["eval", "--n", "2", "--restarts", "2", "--iters", "300",
               "--escalation-rounds", "0", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1/2]" not in out
    assert "Exact solves" in out
