import fnmatch
import shutil
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
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, path: str) -> dict:
        try:
            content = Path(path).expanduser().read_text()
            return {"content": content, "error": None}
        except Exception as e:
            return {"content": None, "error": str(e)}

    def find_files(self, pattern: str, directory: str = "~", file_type: str = "any") -> dict:
        """Case-insensitive file/directory search by name pattern under directory."""
        try:
            root = Path(directory).expanduser().resolve()
            mdfind_bin = shutil.which("mdfind")
            if mdfind_bin:
                return self._find_files_mdfind(mdfind_bin, pattern, root, file_type)
            return self._find_files_rglob(pattern, root, file_type)
        except Exception as e:
            return {"matches": [], "count": 0, "error": str(e)}

    def _find_files_mdfind(self, mdfind_bin: str, pattern: str, root: Path, file_type: str) -> dict:
        query = f'kMDItemFSName == "{pattern}"c'
        result = subprocess.run(
            [mdfind_bin, query, "-onlyin", str(root)],
            capture_output=True, text=True, timeout=15,
        )
        raw = [p for p in result.stdout.strip().splitlines() if p]
        matches = []
        for p_str in raw:
            p = Path(p_str)
            if file_type == "file" and not p.is_file():
                continue
            if file_type == "dir" and not p.is_dir():
                continue
            matches.append(p_str)
            if len(matches) >= 100:
                break
        return {"matches": matches, "count": len(matches), "error": None}

    def _find_files_rglob(self, pattern: str, root: Path, file_type: str) -> dict:
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

    def search_content(self, pattern: str, directory: str = "~",
                       file_glob: str = "*", case_sensitive: bool = False) -> dict:
        """Search for a text pattern inside files. Returns matching file paths and lines."""
        try:
            root = Path(directory).expanduser().resolve()
            rg_bin = shutil.which("rg")
            if rg_bin:
                return self._search_content_rg(rg_bin, pattern, root, file_glob, case_sensitive)
            return self._search_content_grep(pattern, root, file_glob, case_sensitive)
        except Exception as e:
            return {"files": [], "snippets": [], "count": 0, "error": str(e)}

    def _search_content_rg(self, rg_bin: str, pattern: str, root: Path,
                            file_glob: str, case_sensitive: bool) -> dict:
        flags = [] if case_sensitive else ["-i"]
        result = subprocess.run(
            [rg_bin, "-l", "--glob", file_glob, *flags, pattern, str(root)],
            capture_output=True, text=True, timeout=30,
        )
        files = [f for f in result.stdout.strip().splitlines() if f]
        snippets = []
        for fpath in files[:20]:
            r2 = subprocess.run(
                [rg_bin, "-n", "-m", "5", "--glob", file_glob, *flags, pattern, fpath],
                capture_output=True, text=True, timeout=10,
            )
            if r2.stdout.strip():
                snippets.append({"file": fpath, "lines": r2.stdout.strip().splitlines()})
        return {"files": files, "snippets": snippets, "count": len(files), "error": None}

    def _search_content_grep(self, pattern: str, root: Path,
                              file_glob: str, case_sensitive: bool) -> dict:
        flags = [] if case_sensitive else ["-i"]
        excludes = ["--exclude-dir=.git", "--exclude-dir=.venv", "--exclude-dir=node_modules",
                    "--exclude-dir=__pycache__", "--exclude-dir=.DS_Store"]
        result = subprocess.run(
            ["grep", "-r", "--include", file_glob, "-l", *flags, *excludes, pattern, str(root)],
            capture_output=True, text=True, timeout=30,
        )
        files = [f for f in result.stdout.strip().splitlines() if f]
        snippets = []
        for fpath in files[:20]:
            r2 = subprocess.run(
                ["grep", "-n", "--include", file_glob, *flags, "-m", "5", pattern, fpath],
                capture_output=True, text=True, timeout=10,
            )
            if r2.stdout.strip():
                snippets.append({"file": fpath, "lines": r2.stdout.strip().splitlines()})
        return {"files": files, "snippets": snippets, "count": len(files), "error": None}

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
