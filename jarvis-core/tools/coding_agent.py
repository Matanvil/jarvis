import hashlib
import logging
import os
import subprocess
import sys

_log = logging.getLogger("jarvis.coding_agent")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai-coding-agent')))

from src.agent_loop import AgentLoop
from src.planner import Planner, PlannerError
from src.reviewer import Reviewer, ReviewerError
from src.indexer import index_repo
from src.llm import ClaudeClient
from src.ollama_client import OllamaClient
from src.hybrid_client import HybridClient
from src.embedder import OllamaEmbedder
from src.store import VectorStore


_CHROMA_PATH = os.path.expanduser("~/.jarvis/coding-agent-chroma")


class CodingAgentTool:
    def __init__(self, config: dict):
        ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
        claude = ClaudeClient(
            model=config.get("models", {}).get("haiku", "claude-haiku-4-5-20251001"),
            api_key=config["anthropic_api_key"],
        )
        local_model = config.get("local_model", "")
        if local_model:
            self._llm = HybridClient(
                ollama=OllamaClient(model=local_model, base_url=ollama_host),
                claude=claude,
            )
        else:
            self._llm = claude
        self._embedder = OllamaEmbedder(
            model="nomic-embed-text",
            base_url=config.get("ollama", {}).get("host", "http://localhost:11434"),
        )
        self._chroma_path = _CHROMA_PATH
        self._stores: dict[str, VectorStore] = {}

    def _ensure_indexed(self, cwd: str) -> VectorStore:
        """Return the VectorStore for cwd, indexing first if needed."""
        if cwd in self._stores:
            return self._stores[cwd]

        collection_name = "repo_" + hashlib.md5(cwd.encode()).hexdigest()[:12]
        store = VectorStore(chroma_path=self._chroma_path, collection_name=collection_name)

        if store.count() == 0:
            index_repo(cwd, self._embedder, store)

        self._stores[cwd] = store
        return store

    def ask(self, question: str, cwd: str) -> dict:
        llm_model = getattr(self._llm, "model", "unknown")
        _log.info("coding_ask starting — model=%s cwd=%s", llm_model, cwd)
        try:
            store = self._ensure_indexed(cwd)
            loop = AgentLoop(self._llm, self._embedder, store, repo_root=cwd)

            def _on_event(event: str, data: dict) -> None:
                if event == "model_fallback":
                    _log.warning("coding_ask model fallback — kind=%s reason=%s", data.get("kind"), data.get("reason"))

            answer = loop.ask(question, on_event=_on_event)
            return {"answer": answer, "error": None}
        except Exception as e:
            _log.error("coding_ask failed — %s", e)
            return {"answer": None, "error": str(e)}

    def plan(self, task: str, cwd: str) -> dict:
        # Planner calls llm.client.messages.create() directly — always uses Claude (not local model)
        llm_model = getattr(self._llm, "model", "unknown")
        _log.info("coding_plan starting — model=%s (direct API, local model bypassed) cwd=%s", llm_model, cwd)
        try:
            store = self._ensure_indexed(cwd)
            planner = Planner(self._llm, self._embedder, store, repo_root=cwd)
            result = planner.plan(task, cwd)
            edits = [
                {
                    "file": e.file,
                    "description": e.description,
                    "old_code": e.old_code,
                    "new_code": e.new_code,
                }
                for e in result.edits
            ]
            summary_lines = [f"Plan: {task}", ""]
            for e in result.edits:
                summary_lines.append(f"• {e.file}: {e.description}")
            return {"plan_summary": "\n".join(summary_lines), "edits": edits, "error": None}
        except PlannerError as e:
            _log.error("coding_plan failed — %s", e)
            return {"plan_summary": None, "edits": None, "error": str(e)}
        except Exception as e:
            _log.error("coding_plan failed — %s", e)
            return {"plan_summary": None, "edits": None, "error": str(e)}

    def review(self, cwd: str, context: str = "") -> dict:
        # Reviewer calls llm.client.messages.create() directly — always uses Claude (not local model)
        llm_model = getattr(self._llm, "model", "unknown")
        _log.info("coding_review starting — model=%s (direct API, local model bypassed) cwd=%s", llm_model, cwd)
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return {"summary": None, "issues": None, "error": proc.stderr.strip() or "git diff failed"}
            diff = proc.stdout
            if not diff.strip():
                return {"summary": None, "issues": None, "error": "No changes to review (git diff HEAD is empty)"}

            store = self._ensure_indexed(cwd)
            reviewer = Reviewer(self._llm, self._embedder, store, repo_root=cwd)
            result = reviewer.review(diff, context)
            issues = [
                {
                    "category": i.category,
                    "description": i.description,
                    "file": i.file,
                    "recommendation": i.recommendation,
                }
                for i in result.issues
            ]
            return {"summary": result.summary, "issues": issues, "error": None}
        except ReviewerError as e:
            _log.error("coding_review failed — %s", e)
            return {"summary": None, "issues": None, "error": str(e)}
        except Exception as e:
            _log.error("coding_review failed — %s", e)
            return {"summary": None, "issues": None, "error": str(e)}
