import hashlib
import json
import logging
from pathlib import Path

import httpx

_DISCOVER_SYSTEM_PROMPT = """You are a project analyzer. Read the provided file contents and return ONLY a JSON object — no other text, no markdown.

Return this structure (omit keys you cannot determine):
{
  "project_type": "node|python|rust|go|java|other",
  "build_command": "command to build",
  "test_command": "command to run tests",
  "dev_command": "command to start dev server",
  "important_files": ["list", "of", "key", "files"],
  "notes": "any important notes about this project"
}"""

# Config files that signal project type and commands
_DISCOVERY_FILES = [
    "package.json", "pyproject.toml", "setup.py", "Makefile",
    "Cargo.toml", "go.mod", "build.gradle", "pom.xml",
]


class ProjectMemory:
    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or (Path.home() / ".jarvis" / "projects")
        self._logger = logging.getLogger("jarvis.commands")

    def _project_dir(self, cwd: str) -> Path:
        key = hashlib.md5(cwd.encode()).hexdigest()
        return self._base / key

    def load(self, cwd: str) -> dict:
        path = self._project_dir(cwd) / "memory.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def save(self, cwd: str, data: dict) -> None:
        project_dir = self._project_dir(cwd)
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / "memory.json"
        data["cwd"] = cwd
        path.write_text(json.dumps(data, indent=2))

    def update(self, cwd: str, key: str, value: str) -> None:
        existing = self.load(cwd)
        existing[key] = value
        self.save(cwd, existing)

    def format_context(self, cwd: str) -> str:
        """Return a short context string for injection into agent system prompts."""
        data = self.load(cwd)
        if not data:
            return ""
        parts = []
        if data.get("project_type"):
            parts.append(f"Type: {data['project_type']}")
        if data.get("build_command"):
            parts.append(f"Build: {data['build_command']}")
        if data.get("test_command"):
            parts.append(f"Test: {data['test_command']}")
        if data.get("dev_command"):
            parts.append(f"Dev: {data['dev_command']}")
        if data.get("notes"):
            parts.append(f"Notes: {data['notes']}")
        return " | ".join(parts)

    def discover(self, cwd: str, ollama_host: str, ollama_model: str) -> dict:
        """Auto-discover project metadata by asking Ollama to analyze config files."""
        cwd_path = Path(cwd)
        file_contents = []
        for filename in _DISCOVERY_FILES:
            fpath = cwd_path / filename
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")[:2000]
                    file_contents.append(f"=== {filename} ===\n{content}")
                except Exception:
                    pass

        if not file_contents:
            return {}

        user_msg = "Analyze these project files:\n\n" + "\n\n".join(file_contents)

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{ollama_host}/v1/chat/completions",
                    json={
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": _DISCOVER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Strip markdown code fences if model wraps JSON
                content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                data = json.loads(content)
                self.save(cwd, data)
                return data
        except Exception as e:
            self._logger.warning(f"Memory discovery failed for {cwd}: {e}")
            return {}
