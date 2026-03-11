import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from memory import ProjectMemory


class CommandStatus(str, Enum):
    RECEIVED = "RECEIVED"
    ROUTING = "ROUTING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JarvisCommand:
    id: str
    source: str           # "voice" | "hotkey" | "telegram" | "ui"
    raw_input: str
    cwd: str | None
    status: CommandStatus
    created_at: float
    completed_at: float | None = None
    result: dict | None = None


class CommandPipeline:
    """Wraps the Router with command lifecycle tracking and a single-command lock.

    The registry is in-memory only (V0). Interface designed for future persistence
    swap-in without touching pipeline logic.
    """

    _MAX_REGISTRY_SIZE = 50

    def __init__(self, router, memory: ProjectMemory | None = None):
        self._router = router
        self._memory = memory
        self._registry: dict[str, JarvisCommand] = {}
        self._executing: bool = False
        self._current_command_id: str | None = None
        self._logger = logging.getLogger("jarvis.commands")

    def _create_command(self, text: str, cwd: str | None, source: str) -> JarvisCommand:
        return JarvisCommand(
            id=str(uuid.uuid4()),
            source=source,
            raw_input=text,
            cwd=cwd,
            status=CommandStatus.RECEIVED,
            created_at=time.time(),
        )

    def _register(self, cmd: JarvisCommand) -> None:
        if len(self._registry) >= self._MAX_REGISTRY_SIZE:
            # Evict oldest completed/failed/cancelled command
            for old_id, old_cmd in list(self._registry.items()):
                if old_cmd.status in (CommandStatus.COMPLETED, CommandStatus.FAILED, CommandStatus.CANCELLED):
                    del self._registry[old_id]
                    break
        self._registry[cmd.id] = cmd

    def submit(self, text: str, cwd: str | None = None, source: str = "hotkey", step_callback=None) -> dict:
        """Submit a command. Returns busy response if another command is executing."""
        if self._executing:
            return {"busy": True, "command_id": self._current_command_id}

        cmd = self._create_command(text, cwd, source)
        self._register(cmd)
        self._executing = True
        self._current_command_id = cmd.id

        # Load project memory context
        memory_context = ""
        if self._memory and cwd:
            memory_context = self._memory.format_context(cwd)
            if not memory_context:
                # Unknown project — trigger discovery (best-effort)
                try:
                    ollama_cfg = self._router._config.get("ollama", {})
                    self._memory.discover(
                        cwd,
                        ollama_host=ollama_cfg.get("host", "http://localhost:11434"),
                        ollama_model=ollama_cfg.get("model", "mistral:latest"),
                    )
                    memory_context = self._memory.format_context(cwd)
                except Exception:
                    pass

        try:
            cmd.status = CommandStatus.ROUTING
            result = self._router.process(text, cwd=cwd, memory_context=memory_context, source=source, step_callback=step_callback)
            cmd.status = CommandStatus.COMPLETED
            cmd.result = result
            cmd.completed_at = time.time()
            return {**result, "command_id": cmd.id}
        except Exception:
            cmd.status = CommandStatus.FAILED
            cmd.completed_at = time.time()
            raise
        finally:
            self._executing = False
            self._current_command_id = None

    def cancel(self, command_id: str) -> dict:
        """Cancel an executing command and release the lock."""
        cmd = self._registry.get(command_id)
        if cmd is None:
            return {"error": "not_found"}
        cmd.status = CommandStatus.CANCELLED
        cmd.completed_at = time.time()
        if self._current_command_id == command_id:
            self._executing = False
            self._current_command_id = None
        return {"cancelled": True, "command_id": command_id}

    def reset_conversation(self) -> None:
        """Clear the router's conversation history."""
        self._router.reset_conversation()

    def abort(self) -> dict:
        """Force-release the lock unconditionally. Emergency escape hatch."""
        self._executing = False
        self._current_command_id = None
        return {"lock_released": True}

    def get(self, command_id: str) -> JarvisCommand | None:
        return self._registry.get(command_id)

    def list_recent(self, limit: int = 50) -> list[dict]:
        cmds = sorted(self._registry.values(), key=lambda c: c.created_at, reverse=True)
        return [
            {
                "id": c.id,
                "source": c.source,
                "raw_input": c.raw_input,
                "status": c.status,
                "created_at": c.created_at,
                "completed_at": c.completed_at,
            }
            for c in cmds[:limit]
        ]
