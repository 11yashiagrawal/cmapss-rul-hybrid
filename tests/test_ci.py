"""Exercise the actual PR branch guard without calling the GitHub API."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.parametrize(
    ("base", "head", "head_repo", "allowed"),
    [
        ("dev", "feat/affan80/example", "vks-g/cmapss-rul-hybrid", True),
        ("dev", "feature", "contributor/cmapss-rul-hybrid", True),
        ("main", "dev", "vks-g/cmapss-rul-hybrid", True),
        ("main", "feature", "vks-g/cmapss-rul-hybrid", False),
        ("main", "dev", "contributor/cmapss-rul-hybrid", False),
        ("main", "dev; exit 0", "vks-g/cmapss-rul-hybrid", False),
    ],
)
def test_pr_branch_guard(base, head, head_repo, allowed):
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/pr-branch-guard.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text())
    script = workflow["jobs"]["branch-guard"]["steps"][0]["run"]

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "GITHUB_BASE_REF": base,
            "GITHUB_HEAD_REF": head,
            "GITHUB_REPOSITORY": "vks-g/cmapss-rul-hybrid",
            "HEAD_REPOSITORY": head_repo,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (0 if allowed else 1), result.stdout + result.stderr
    if not allowed:
        assert "::error::" in result.stdout
        assert "dev" in result.stdout
