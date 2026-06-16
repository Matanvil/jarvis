# jarvis-core/git_context.py
import subprocess


def get_git_context(cwd: str) -> dict | None:
    """Return {branch, commits, remote} for the git repo at cwd, or None on any failure."""
    try:
        branch = _git(cwd, ["branch", "--show-current"]).strip()
        if not branch:
            return None
        log_out = _git(cwd, ["log", "--oneline", "-3"]).strip()
        commits = [line for line in log_out.splitlines() if line.strip()]
        remote = None
        try:
            remote = _git(cwd, ["remote", "get-url", "origin"]).strip() or None
        except Exception:
            pass
        return {"branch": branch, "commits": commits, "remote": remote}
    except Exception:
        return None


def _git(cwd: str, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", cwd] + args,
        capture_output=True, text=True, timeout=3,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "git")
    return result.stdout
