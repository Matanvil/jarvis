import fnmatch
import subprocess
from pathlib import Path


class ShellTool:
    def run(self, command: str, cwd: str | None = None, timeout: int = 30) -> dict:
        try:
            result = subprocess.run(
                command,
                shell=True,
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
            return {"exit_code": -1, "stdout": "", "stderr": "", "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": "", "error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, path: str) -> dict:
        try:
            content = Path(path).read_text()
            return {"content": content, "error": None}
        except Exception as e:
            return {"content": None, "error": str(e)}

    def find_files(self, pattern: str, directory: str = "~", file_type: str = "any") -> dict:
        """Case-insensitive file/directory search by name pattern under directory."""
        try:
            root = Path(directory).expanduser().resolve()
            pattern_lower = pattern.lower()
            matches = []
            for p in root.rglob("*"):
                if file_type == "file" and not p.is_file():
                    continue
                if file_type == "dir" and not p.is_dir():
                    continue
                if fnmatch.fnmatch(p.name.lower(), pattern_lower):
                    matches.append(str(p))
                if len(matches) >= 100:
                    break
            return {"matches": matches, "count": len(matches), "error": None}
        except Exception as e:
            return {"matches": [], "count": 0, "error": str(e)}

    def search_content(self, pattern: str, directory: str = "~",
                       file_glob: str = "*", case_sensitive: bool = False) -> dict:
        """Search for a text pattern inside files. Returns matching file paths and lines."""
        try:
            root = Path(directory).expanduser().resolve()
            flags = [] if case_sensitive else ["-i"]
            # Add --exclude-dir for common large/irrelevant trees to avoid timeouts
            excludes = ["--exclude-dir=.git", "--exclude-dir=.venv", "--exclude-dir=node_modules",
                        "--exclude-dir=__pycache__", "--exclude-dir=.DS_Store"]
            result = subprocess.run(
                ["grep", "-r", "--include", file_glob, "-l", *flags, *excludes, pattern, str(root)],
                capture_output=True, text=True, timeout=10
            )
            files = [f for f in result.stdout.strip().splitlines() if f]
            # For each matched file get the matching lines (up to 5 per file)
            snippets = []
            for fpath in files[:20]:
                r2 = subprocess.run(
                    ["grep", "-n", "--include", file_glob, *flags, "-m", "5", pattern, fpath],
                    capture_output=True, text=True, timeout=5
                )
                if r2.stdout.strip():
                    snippets.append({"file": fpath, "lines": r2.stdout.strip().splitlines()})
            return {"files": files, "snippets": snippets, "count": len(files), "error": None}
        except Exception as e:
            return {"files": [], "snippets": [], "count": 0, "error": str(e)}

    def file_edit(self, path: str, old_string: str, new_string: str,
                  replace_all: bool = False) -> dict:
        """Replace old_string with new_string in a file.
        Fails if old_string is not found or is ambiguous (appears >1 time) unless replace_all=True."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return {"success": False, "error": f"File not found: {path}"}
            content = p.read_text()
            count = content.count(old_string)
            if count == 0:
                return {"success": False, "error": f"String not found in {p.name}"}
            if count > 1 and not replace_all:
                return {"success": False, "error": f"Ambiguous edit: '{old_string}' found {count} times. Use replace_all=true or provide more context."}
            p.write_text(content.replace(old_string, new_string) if replace_all
                         else content.replace(old_string, new_string, 1))
            replaced = count if replace_all else 1
            return {"success": True, "replaced": replaced, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_dir(self, path: str) -> dict:
        try:
            entries = list(Path(path).expanduser().iterdir())
            return {
                "entries": [{"name": e.name, "is_dir": e.is_dir()} for e in entries],
                "error": None,
            }
        except Exception as e:
            return {"entries": [], "error": str(e)}
