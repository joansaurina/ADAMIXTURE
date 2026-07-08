from pathlib import Path
import sys

import pytest


TEST_FILES = (
    "test_reader.py",
    "test_brqn_cpu.py",
    "test_adamem_cpu.py",
    "test_cuda.py",
    "test_mps.py",
)


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    root = tests_dir.parent
    sys.path.insert(0, str(root))
    test_paths = [str(tests_dir / name) for name in TEST_FILES]
    return pytest.main([*test_paths, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
