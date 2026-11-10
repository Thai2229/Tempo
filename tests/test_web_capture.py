"""Runs the JS unit tests of the demo page's capture logic (needs Node 20+)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_web_capture_js():
    result = subprocess.run(
        ["node", "--test", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "web").glob("*.test.mjs"))], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
