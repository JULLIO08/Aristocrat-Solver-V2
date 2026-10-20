"""Estimate legacy (per-step neural) eval wall time by timing a short run."""
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

from neurocrypt.corpus import load_lines, DEFAULT_ENCODED  # noqa: E402
from neurocrypt.solver import Solver  # noqa: E402


def main() -> None:
    solver = Solver.from_models(PKG_ROOT / "models")
    cipher = load_lines(DEFAULT_ENCODED)[7]
    t0 = time.perf_counter()
    solver.solve(cipher, restarts=1, iters=200, seed=0, legacy=True)
    dt = time.perf_counter() - t0
    per_step = dt / 200
    full = per_step * 4000 * 10 * 50
    print(f"legacy per-step: {per_step * 1000:.1f} ms ({dt:.1f}s for 200 iters)")
    print(f"extrapolated legacy eval (50 ciphers x 10 restarts x 4000 iters): "
          f"{full / 3600:.1f} h")


if __name__ == "__main__":
    main()
