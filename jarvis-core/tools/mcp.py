import asyncio
import logging
import os
import subprocess
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_log = logging.getLogger("jarvis.mcp")

_PREFIX = "mcp__"


class MCPManager:
    """Manages persistent MCP server connections and exposes their tools to both agents.

    Tool names are prefixed  mcp__<server_name>__<tool_name>  to avoid collisions
    with built-in Jarvis tools.  start_all() is called once at server startup;
    failures are non-fatal — the server logs a warning and continues without that
    MCP server's tools.
    """

    def __init__(self, server_configs: list[dict]):
        self._configs: dict[str, dict] = {c["name"]: c for c in server_configs}
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, list[dict]] = {}  # server_name → list of OpenAI-shaped schemas
        self._transport_cms: dict[str, object] = {}   # keep stdio_client CMs alive
        self._session_cms: dict[str, object] = {}     # keep ClientSession CMs alive
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ── public API ───────────────────────────────────────────────────────────

    def server_names(self) -> list[str]:
        return list(self._configs)

    def tool_schemas(self) -> list[dict]:
        """All MCP tools as OpenAI-compatible tool dicts, ready to append to TOOL_DEFINITIONS."""
        schemas = []
        for tools in self._tools.values():
            schemas.extend(tools)
        return schemas

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith(_PREFIX)

    def parse_mcp_tool_name(self, tool_name: str) -> tuple[str, str]:
        """'mcp__fs__read_file' → ('fs', 'read_file')"""
        without_prefix = tool_name[len(_PREFIX):]
        server, _, tool = without_prefix.partition("__")
        return server, tool

    def guardrail_category(self, server_name: str) -> str:
        return self._configs.get(server_name, {}).get("guardrail", "mcp_tool")

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """Synchronously call a tool on the named MCP server. Returns text output."""
        if server_name not in self._configs:
            raise RuntimeError(f"MCP server '{server_name}' not configured")
        if server_name not in self._sessions:
            raise RuntimeError(f"MCP server '{server_name}' not connected")
        session = self._sessions[server_name]
        coro = session.call_tool(tool_name, arguments)
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            result = future.result(timeout=60)
        else:
            result = asyncio.run(coro)
        if result.isError:
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "error: " + (" ".join(texts) or "MCP tool returned an error")
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)

    def start_all(self) -> None:
        """Connect to all configured MCP servers in a background event loop thread."""
        if not self._configs:
            return
        self._ensure_loop()
        for name, cfg in self._configs.items():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._connect(name, cfg), self._loop
                )
                future.result(timeout=30)
            except Exception as exc:
                _log.warning("[MCP] Failed to start server '%s': %s", name, exc)

    def connect_server(self, cfg: dict) -> None:
        """Add and connect a single server after startup. Idempotent by server name."""
        name = cfg["name"]
        self._configs[name] = cfg
        self._ensure_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._connect(name, cfg), self._loop
            )
            future.result(timeout=30)
        except Exception as exc:
            _log.warning("[MCP] Failed to connect server '%s': %s", name, exc)
            raise

    def stop_all(self) -> None:
        """Disconnect all sessions and clear state."""
        self._sessions.clear()
        self._tools.clear()
        self._transport_cms.clear()
        self._session_cms.clear()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _resolve_env(self, cfg: dict) -> dict:
        """Build the env dict for a server, resolving auth helpers if needed."""
        base = {**os.environ, **(cfg.get("env") or {})}
        auth = cfg.get("auth")
        if auth == "gh_cli":
            try:
                result = subprocess.run(
                    ["gh", "auth", "token"],
                    capture_output=True, text=True,
                )
            except FileNotFoundError:
                raise RuntimeError("gh CLI is not installed. Install it with: brew install gh")
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(
                    "gh CLI is not authenticated. Run: gh auth login"
                )
            base["GITHUB_PERSONAL_ACCESS_TOKEN"] = result.stdout.strip()
        return base

    # ── internal ─────────────────────────────────────────────────────────────

    def _ensure_loop(self) -> None:
        """Start the background event loop thread if it isn't running yet."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

    def _inject_tools(self, server_name: str, mcp_tools: list) -> None:
        """Register discovered tools (called after successful connect or in tests)."""
        schemas = []
        for t in mcp_tools:
            schemas.append({
                "name": f"{_PREFIX}{server_name}__{t.name}",
                "description": t.description or "",
                "input_schema": t.inputSchema,
            })
        self._tools[server_name] = schemas

    async def _connect(self, name: str, cfg: dict) -> None:
        transport = cfg.get("transport", "stdio")
        if transport != "stdio":
            _log.warning("[MCP] Transport '%s' not yet supported for server '%s'", transport, name)
            return

        merged_env = self._resolve_env(cfg)

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env={k: str(v) for k, v in merged_env.items()},
        )

        # Store context managers so the subprocess and session stay alive.
        transport_cm = stdio_client(params)
        read, write = await transport_cm.__aenter__()
        self._transport_cms[name] = transport_cm

        session_cm = ClientSession(read, write)
        await session_cm.__aenter__()
        self._session_cms[name] = session_cm

        await session_cm.initialize()
        result = await session_cm.list_tools()
        self._sessions[name] = session_cm
        self._inject_tools(name, result.tools)
        _log.info("[MCP] Connected to '%s': %d tool(s) registered", name, len(result.tools))
