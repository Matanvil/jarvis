import subprocess
import sys
import tempfile
from pathlib import Path

# Maps language aliases to (runtime_args, file_suffix)
# runtime_args may include a placeholder "{file}" for the temp file path
_RUNTIMES: dict[str, tuple[list[str], str]] = {
    "python":     ([sys.executable, "{file}"], ".py"),
    "py":         ([sys.executable, "{file}"], ".py"),
    "javascript": (["node", "{file}"], ".js"),
    "js":         (["node", "{file}"], ".js"),
    "node":       (["node", "{file}"], ".js"),
    "typescript": (["npx", "--yes", "tsx", "{file}"], ".ts"),
    "ts":         (["npx", "--yes", "tsx", "{file}"], ".ts"),
    "bash":       (["bash", "{file}"], ".sh"),
    "sh":         (["bash", "{file}"], ".sh"),
    "ruby":       (["ruby", "{file}"], ".rb"),
    "rb":         (["ruby", "{file}"], ".rb"),
    "swift":      (["swift", "{file}"], ".swift"),
}

SUPPORTED_LANGUAGES = sorted(_RUNTIMES.keys())


class CodeTool:
    def run_snippet(self, code: str, language: str, cwd: str | None = None, timeout: int = 30) -> dict:
        """Run a code snippet in the given language. cwd sets the working directory
        so the snippet can access project files and local imports."""
        lang = language.lower().strip()
        if lang not in _RUNTIMES:
            return {
                "exit_code": -1, "stdout": "", "stderr": "",
                "error": f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}",
            }

        runtime_args, suffix = _RUNTIMES[lang]

        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = [a.replace("{file}", tmp_path) for a in runtime_args]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": None,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "", "error": f"Timed out after {timeout}s"}
        except FileNotFoundError as e:
            return {
                "exit_code": -1, "stdout": "", "stderr": "",
                "error": f"Runtime not found for '{language}': {e}. Is it installed?",
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Convenience shorthands
    def run_python(self, code: str, cwd: str | None = None, timeout: int = 30) -> dict:
        return self.run_snippet(code, "python", cwd=cwd, timeout=timeout)

    def run_js(self, code: str, cwd: str | None = None, timeout: int = 30) -> dict:
        return self.run_snippet(code, "javascript", cwd=cwd, timeout=timeout)

    def run_ts(self, code: str, cwd: str | None = None, timeout: int = 30) -> dict:
        return self.run_snippet(code, "typescript", cwd=cwd, timeout=timeout)

    def run_shell_script(self, script: str, cwd: str | None = None, timeout: int = 30) -> dict:
        return self.run_snippet(script, "bash", cwd=cwd, timeout=timeout)
