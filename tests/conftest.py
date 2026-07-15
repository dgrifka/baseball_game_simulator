"""
Ensures the repo root is importable as `Simulator.*` regardless of how
pytest's import mode resolves test-file paths (there's no top-level
conftest.py/pyproject.toml pinning rootdir insertion in this repo).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
