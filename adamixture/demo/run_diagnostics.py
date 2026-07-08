import sys
from pathlib import Path

import numpy as np


RTOL = 1e-6
ATOL = 1e-8


def _load_matrix(path: Path):
    try:
        return np.genfromtxt(path)
    except FileNotFoundError:
        print(f"Could not find file: {path}")
        return None


def run_checks() -> bool:
    outputs_dir = Path("outputs")
    expected_dir = outputs_dir / "cpu" / "brqn"

    actual_q = _load_matrix(outputs_dir / "demo_run.7.Q")
    actual_p = _load_matrix(outputs_dir / "demo_run.7.P")
    expected_q = _load_matrix(expected_dir / "demo_run.brqn.Q.expected")
    expected_p = _load_matrix(expected_dir / "demo_run.brqn.P.expected")

    if any(x is None for x in (actual_q, actual_p, expected_q, expected_p)):
        print("Please run the demo from adamixture/demo and make sure expected outputs are present.")
        return False

    q_ok = np.allclose(actual_q, expected_q, rtol=RTOL, atol=ATOL)
    p_ok = np.allclose(actual_p, expected_p, rtol=RTOL, atol=ATOL)

    if not q_ok:
        print("Q output differs from expected output.")
    if not p_ok:
        print("P output differs from expected output.")

    return q_ok and p_ok


if __name__ == "__main__":
    passed = run_checks()
    print(f"Output and expected output are {'' if passed else 'NOT '}similar.")
    sys.exit(0 if passed else 1)
