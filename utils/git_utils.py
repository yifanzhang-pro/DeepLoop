from __future__ import annotations

from pathlib import Path
import subprocess


def get_repo_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit SHA for `repo_root` if available."""
    resolved_repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(resolved_repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    if not commit:
        return None
    return commit


def get_checkpoint_commit_mismatch_warning(
    *,
    current_commit: str | None,
    checkpoint_commit: object,
    checkpoint_dir: str | Path,
) -> str | None:
    """Return a warning message when the checkpoint commit differs from current codebase commit."""
    if current_commit is None or not isinstance(checkpoint_commit, str):
        return None
    checkpoint_commit_stripped = checkpoint_commit.strip()
    if checkpoint_commit_stripped == "":
        return None
    if checkpoint_commit_stripped == current_commit:
        return None
    return (
        "WARNING: Checkpoint commit mismatch detected: "
        f"checkpoint_commit={checkpoint_commit_stripped} "
        f"current_commit={current_commit} "
        f"checkpoint_dir={Path(checkpoint_dir)}"
    )
