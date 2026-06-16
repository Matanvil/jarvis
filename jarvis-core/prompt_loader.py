# jarvis-core/prompt_loader.py
import hashlib
import logging
from pathlib import Path

_log = logging.getLogger("jarvis.errors")


class PromptLoader:
    """Loads prompt files and user refs at startup. All results cached in memory."""

    def __init__(
        self,
        prompts_dir: Path | None = None,
        refs_dir: Path | None = None,
        projects_dir: Path | None = None,
    ):
        self._prompts_dir = prompts_dir or (Path(__file__).parent / "prompts")
        self._refs_dir = refs_dir or (Path.home() / ".jarvis" / "refs")
        self._projects_dir = projects_dir or (Path.home() / ".jarvis" / "projects")
        self._base = self._read(self._prompts_dir / "base.md")
        self._local = self._read(self._prompts_dir / "local.md")
        self._profile = self._read_profile()

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _read_profile(self) -> str:
        path = self._refs_dir / "profile.md"
        if not path.exists():
            _log.info(
                "No profile found at %s — copy prompts/profile.template.md there to personalise Jarvis.",
                path,
            )
            return ""
        return path.read_text(encoding="utf-8")

    def base_prompt(self) -> str:
        return self._base

    def local_extra(self) -> str:
        return self._local

    def profile(self) -> str:
        return self._profile

    def refs_index(self, cwd: str | None) -> list[str]:
        """Return paths of available ref files (global + per-project), excluding profile.md."""
        paths: list[str] = []
        if self._refs_dir.exists():
            for f in sorted(self._refs_dir.glob("*.md")):
                if f.name != "profile.md":
                    paths.append(str(f))
        if cwd:
            key = hashlib.md5(cwd.encode()).hexdigest()
            project_refs = self._projects_dir / key / "refs"
            if project_refs.exists():
                for f in sorted(project_refs.glob("*.md")):
                    paths.append(str(f))
        return paths
