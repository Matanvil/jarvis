import anthropic
import json
import logging
import os
from dataclasses import dataclass, field
import approval_store
from guardrails import Action, Decision, Guardrails
from tools.shell import ShellTool
from tools.web import WebTool
from tools.code import CodeTool, SUPPORTED_LANGUAGES
from tools.macos import MacOSTool
from tools.coding_agent import CodingAgentTool
from tools._dispatch import execute_tool, format_response, _claude_code_available as claude_code_available
from tools._errors import ApprovalRequiredError
from tools.rag import RAGTool
from memory import ProjectMemory

_BASE_SYSTEM_PROMPT = """You are Jarvis, a macOS AI assistant and coding partner. You help the user by executing tasks directly — think of yourself as a voice-operated Claude Code.

Rules:
- CRITICAL: NEVER claim to have performed an action (created a file, ran a command, opened an app, etc.) without first calling the appropriate tool. If you did not call a tool, you did not do the action.
- CRITICAL: For destructive operations (deleting files, overwriting, sending messages) — if you are not certain of the exact target, use find_files or list_dir first to confirm. NEVER guess. If the command is ambiguous (e.g. "delete this file" with no file named), ask the user to clarify rather than inventing a target.
- Never write "[Tools used: ...]" in your response — this annotation is added automatically by the system from actual tool call records, not from your text.
- Before multi-step tasks, briefly explain what you'll do (1 sentence)
- Use tools to take real action, don't just describe what to do
- Be concise — the user hears your responses via text-to-speech
- For coding tasks: write the code, execute it, report the result
- Never narrate code contents aloud — just say what you did and whether it succeeded
- When working in a project, prefer running commands in that project's directory
- For complex codebase tasks (PR reviews, multi-file edits, debugging, refactoring), use delegate_to_claude_code — it has full file browsing, grep, and surgical editing capabilities
- Always end your response with a line in this exact format (one sentence, ≤ 120 chars, no quotes):
  VOICE: <spoken summary of what you did or answered, natural and conversational>

Tool choice tips:
- For system info (time, date, disk space, battery, uptime), always use shell_run (e.g. `date`, `df -h`, `uptime`) — never run_code.
- Use run_code only when the task genuinely requires code logic (calculations, data processing, scripts).
- CRITICAL: For any question about files, logs, counts, or live system state — always verify with a tool (shell_run, file_read, list_dir, find_files). Never answer from memory or context alone. If you haven't checked, you don't know.

Coding agent tools (prefer these over reading files manually for codebase work):
- coding_ask: Use when the user asks a question about how the codebase works, where something is implemented, or how files/modules relate. Better than reading files one by one — it uses semantic search. Returns a complete answer — finalize immediately after, do not read more files.
- coding_plan: Use when the user asks you to plan, propose, or design a code change, refactor, reorganization, or new feature. Always prefer this over reasoning from a directory listing. Returns a complete plan — finalize immediately after.
- coding_review: Use when the user asks you to review recent changes, check what was modified, or audit a diff. Returns a complete review — finalize immediately after, do not run additional git commands or read files.

macOS file tips:
- Screenshots are named "Screenshot YYYY-MM-DD at HH.MM.SS.png" — use case-insensitive search: find ~/Desktop -iname "*screenshot*"
- Shell glob patterns are case-sensitive on macOS; prefer find with -iname over ls globs for file searches.
- The user's home directory is {home}. Always use this real path — never use placeholder paths like /Users/username or /Users/yourusername.
- The Jarvis project is at {home}/dev/jarvis (Python core: {home}/dev/jarvis/jarvis-core, Swift app: {home}/dev/jarvis/jarvis-swift, docs/plans: {home}/dev/jarvis/docs/plans).
- Project status and remaining tasks are tracked in {home}/dev/jarvis/docs/plans/progress.md.
"""

TOOL_DEFINITIONS = [
    {
        "name": "shell_run",
        "description": "Run a shell command or terminal operation. Use cwd to run in a specific directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"},
                "cwd": {"type": "string", "description": "Working directory (optional, defaults to active project)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 600). Use for long-running builds or test suites."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "file_write",
        "description": "Write content to a file (creates if not exists, overwrites if exists)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": (
            "Make a surgical edit to a file by replacing an exact string with a new one. "
            "Prefer this over file_write for modifying existing files — only the changed portion is touched. "
            "old_string must match exactly (including whitespace). "
            "Fails with a clear error if the string is not found or appears more than once (use replace_all for the latter)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to the file"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false — fails if ambiguous)"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "file_read",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "find_files",
        "description": (
            "Find files or directories by name pattern. Case-insensitive. "
            "Supports wildcards e.g. '*.png', '*screenshot*', 'AudioController.*'. "
            "Prefer this over shell_run for any file search task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Filename pattern with optional wildcards, e.g. '*.png' or '*screenshot*'"},
                "directory": {"type": "string", "description": "Root directory to search in (default: ~)"},
                "file_type": {"type": "string", "enum": ["any", "file", "dir"], "description": "Filter by type (default: any)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_content",
        "description": (
            "Search for text inside files. Returns matching file paths and lines. "
            "Case-insensitive by default. Use file_glob to restrict to certain file types e.g. '*.swift', '*.py'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex to search for"},
                "directory": {"type": "string", "description": "Directory to search in (default: ~)"},
                "file_glob": {"type": "string", "description": "Filename glob to restrict search e.g. '*.swift' (default: *)"},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive search (default: false)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch and extract text content from a URL",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "run_code",
        "description": (
            f"Write and execute a code snippet in any supported language: {', '.join(SUPPORTED_LANGUAGES)}. "
            "Use cwd to run the snippet in the context of an existing project (giving it access to project files)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "description": f"One of: {', '.join(SUPPORTED_LANGUAGES)}"},
                "cwd": {"type": "string", "description": "Project directory to run in (optional)"},
            },
            "required": ["code", "language"],
        },
    },
    {
        "name": "open_app",
        "description": "Open a macOS application by name",
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "notify",
        "description": "Show a macOS notification",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List the contents of a directory. Returns file and subdirectory names with type indicators. "
            "Use this when you need to explore a directory's contents without a full recursive search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to the directory"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "index_codebase",
        "description": (
            "Index a code repository for semantic search. Call this once per project, then use "
            "search_codebase to find relevant code. Re-run after major changes or if the index is stale. "
            "Stores the index at ~/.jarvis/projects/<hash>/rag_store/. "
            "Requires `ollama pull nomic-embed-text` to have been run once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the repository root (defaults to active project directory)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_codebase",
        "description": (
            "Semantically search an indexed codebase. Returns the most relevant code chunks with "
            "file paths and line numbers. Must call index_codebase first if not yet indexed. "
            "Better than file_read or search_content for conceptual questions like "
            "'where is authentication handled' or 'how does caching work'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Repository root (defaults to active project directory)",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chunks to return (default: 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "delegate_to_local",
        "description": (
            "Delegate a simple local sub-task to the local Ollama agent. "
            "Use this for: reading files, listing directories, running quick shell commands, "
            "checking git status, running tests. "
            "Saves API tokens — Ollama handles it locally and returns the result. "
            "Do NOT use for tasks requiring web search or complex reasoning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Clear description of the sub-task for Ollama to execute"},
                "cwd": {"type": "string", "description": "Working directory (optional)"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "delegate_to_claude_code",
        "description": (
            "Delegate a complex codebase task to Claude Code CLI. "
            "Use this for: PR reviews and applying suggested changes, multi-file refactoring, "
            "debugging across files, understanding a codebase, any task requiring file browsing or grep. "
            "Claude Code has full file navigation, surgical editing, and its own agent loop — "
            "it will handle the task end-to-end and return a summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear, complete description of the task for Claude Code to execute",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "create_schedule",
        "description": "Schedule a Jarvis command to run on a recurring or one-time basis. Always generate a short human-readable label. For recurring tasks provide a 5-field cron expression (e.g. '0 9 * * *'). For one-time tasks compute run_at_iso from the current datetime in the system prompt plus any offset the user specified (e.g. 'in 2 minutes' → current time + 2 min as ISO 8601 local time). run_at_iso MUST be a future datetime. IMPORTANT — for the command field: if the user wants to be reminded or notified of something (e.g. 'remind me to call mom', 'send me a smile emoji', 'notify me that X'), store only what needs to be delivered, not how to deliver it — e.g. 'Call mom reminder' or 'smile emoji'. For action tasks (summarise calendar, run build, etc.) store the action as a clear imperative.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The Jarvis command to run (natural language)"},
                "label": {"type": "string", "description": "Short human-readable name, e.g. 'morning calendar summary'"},
                "schedule_type": {"type": "string", "enum": ["recurring", "one_time"]},
                "cron": {"type": ["string", "null"], "description": "5-field cron expression for recurring, e.g. '0 9 * * *'. Null for one_time."},
                "run_at_iso": {"type": ["string", "null"], "description": "ISO 8601 datetime for one_time. Null for recurring."},
            },
            "required": ["command", "label", "schedule_type", "cron", "run_at_iso"],
        },
    },
    {
        "name": "list_schedules",
        "description": "List all scheduled tasks.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "delete_schedule",
        "description": "Delete a scheduled task by ID. Always call list_schedules first to confirm the correct ID. If multiple schedules could match, list them and ask the user which one to delete before proceeding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schedule_id": {"type": "string", "description": "The schedule ID to delete"},
            },
            "required": ["schedule_id"],
        },
    },
    {
        "name": "pause_schedule",
        "description": "Pause a scheduled task without deleting it.",
        "input_schema": {
            "type": "object",
            "properties": {"schedule_id": {"type": "string"}},
            "required": ["schedule_id"],
        },
    },
    {
        "name": "resume_schedule",
        "description": "Resume a paused scheduled task.",
        "input_schema": {
            "type": "object",
            "properties": {"schedule_id": {"type": "string"}},
            "required": ["schedule_id"],
        },
    },
    {
        "name": "coding_ask",
        "description": (
            "Ask a question about the codebase in the given directory. "
            "Uses semantic search over indexed code to answer questions about "
            "architecture, specific functions, data flow, or behavior. "
            "Auto-indexes the codebase on first use. "
            "Returns a complete answer — call finalize immediately after, do NOT read additional files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask about the codebase"},
                "cwd": {"type": "string", "description": "Absolute path to the project directory"},
            },
            "required": ["question", "cwd"],
        },
    },
    {
        "name": "coding_plan",
        "description": (
            "Generate a multi-file edit plan to accomplish a coding task. "
            "Searches the codebase and produces a list of specific file edits "
            "(old_code → new_code) needed to complete the task. "
            "Use this to plan refactors, new features, or bug fixes before applying them. "
            "Returns a complete plan — call finalize immediately after."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The coding task to plan (e.g. 'refactor auth into a service')",
                },
                "cwd": {"type": "string", "description": "Absolute path to the project directory"},
            },
            "required": ["task", "cwd"],
        },
    },
    {
        "name": "coding_review",
        "description": (
            "Review the uncommitted git changes in a project directory. "
            "Runs git diff HEAD, searches the codebase for context, and returns "
            "categorized issues: critical, important, or suggestion. "
            "Returns a complete review — call finalize immediately after, do NOT read additional files or run git commands."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cwd": {"type": "string", "description": "Absolute path to the project directory"},
                "context": {
                    "type": "string",
                    "description": "Optional context about what the changes are for",
                },
            },
            "required": ["cwd"],
        },
    },
]

SCHEDULE_TOOLS = {"create_schedule", "list_schedules", "delete_schedule", "pause_schedule", "resume_schedule"}


def _handle_schedule_tool(tool_name: str, tool_input: dict) -> dict:
    import scheduler as sched_module
    from dataclasses import asdict

    s = sched_module.get_scheduler()
    if s is None:
        return {"error": "Scheduler not running"}

    if tool_name == "create_schedule":
        schedule = s.create(
            command=tool_input["command"],
            label=tool_input["label"],
            schedule_type=tool_input["schedule_type"],
            cron=tool_input.get("cron"),
            run_at_iso=tool_input.get("run_at_iso"),
        )
        return asdict(schedule)

    elif tool_name == "list_schedules":
        return {"schedules": [asdict(x) for x in s.list()]}

    elif tool_name == "delete_schedule":
        ok = s.delete(tool_input["schedule_id"])
        return {"ok": ok, "error": None if ok else "Schedule not found"}

    elif tool_name == "pause_schedule":
        result = s.pause(tool_input["schedule_id"])
        return asdict(result) if result else {"error": "Schedule not found"}

    elif tool_name == "resume_schedule":
        result = s.resume(tool_input["schedule_id"])
        return asdict(result) if result else {"error": "Schedule not found"}

    return {"error": f"Unknown schedule tool: {tool_name}"}


_MILESTONE_TOOLS = {"delegate_to_local", "delegate_to_claude_code"}

_STEP_LABELS: dict[str, str] = {
    "shell_run": "Running command",
    "file_read": "Reading file",
    "file_edit": "Editing file",
    "file_write": "Editing file",
    "web_search": "Searching the web",
    "web_fetch": "Fetching page",
    "find_files": "Searching files",
    "list_dir": "Listing directory",
    "run_code": "Running code",
    "delegate_to_local": "Thinking locally",
    "delegate_to_claude_code": "Delegating to Claude Code",
    "create_schedule": "Creating schedule",
    "list_schedules": "Listing schedules",
    "delete_schedule": "Deleting schedule",
    "pause_schedule": "Pausing schedule",
    "resume_schedule": "Resuming schedule",
    "search_content": "Searching content",
    "notify": "Sending notification",
    "open_app": "Opening app",
    "get_clipboard": "Reading clipboard",
    "set_clipboard": "Writing clipboard",
    "coding_ask": "Asking codebase",
    "coding_plan": "Planning edits",
    "coding_review": "Reviewing changes",
    "finalize": "Done",
}


def _step_label(tool_name: str) -> str:
    """Return a human-readable label for the given tool name, or 'Working…' as fallback."""
    return _STEP_LABELS.get(tool_name, "Working…")


def _is_milestone(tool_name: str, step_index: int) -> bool:
    """True if this step should trigger voice narration."""
    if step_index == 0:
        return True   # first step always narrated
    if tool_name in _MILESTONE_TOOLS:
        return True   # cross-agent handoffs narrated
    return False


@dataclass
class _ClaudeLoopState:
    """Mutable state for one agent run. Carried across a pause/resume so an
    approved run continues instead of replaying from the original command."""
    messages: list
    tool_calls_made: list
    steps: list
    total_steps: int
    last_tool_call: object
    system_prompt: str
    cwd: str | None
    source: str
    local_available: bool
    command_id: str | None
    user_text: str
    # Resume point — set only while paused mid-turn.
    pending_content: list | None = None
    pending_index: int = 0
    pending_results: list | None = None
    wrap_up_nudged: bool = False


class Agent:
    def __init__(self, config: dict, guardrails: Guardrails, local_agent=None,
                 model: str = "claude-haiku-4-5-20251001", mcp_manager=None):
        self._config = config
        self._guardrails = guardrails
        self._model = model
        self._client = anthropic.Anthropic(api_key=config["anthropic_api_key"])
        self._shell = ShellTool()
        self._web = WebTool(brave_api_key=config.get("brave_api_key"))
        self._code = CodeTool()
        self._macos = MacOSTool()
        self._coding = CodingAgentTool(config)
        self._logger = logging.getLogger("jarvis.commands")
        self._local_agent = local_agent
        self._mcp_manager = mcp_manager
        _mem = ProjectMemory()
        self._rag = RAGTool(memory=_mem, ollama_host="http://localhost:11434")

    def _build_tool_list(self) -> list[dict]:
        """Return TOOL_DEFINITIONS plus any MCP tools registered with the manager."""
        if self._mcp_manager is not None:
            return TOOL_DEFINITIONS + self._mcp_manager.tool_schemas()
        return list(TOOL_DEFINITIONS)

    def _build_system_prompt(self, cwd: str | None, memory_context: str = "", source: str = "") -> str:
        prompt = _BASE_SYSTEM_PROMPT.format(home=os.path.expanduser("~"))
        if cwd:
            prompt += f"\nActive project directory: {cwd}\n"
        if memory_context:
            prompt += f"\nProject memory: {memory_context}\n"
        if source == "scheduled":
            prompt += (
                "\n\n=== SCHEDULED TASK MODE ===\n"
                "Rules (strictly enforced):\n"
                "- NEVER call tools to send messages (no curl, no Telegram API, no notify, no create_schedule).\n"
                "- If the command is reminder content (a word, phrase, or emoji) — output ONLY that text, nothing else.\n"
                "  BAD: 'Here is your reminder: Call mom'\n"
                "  BAD: 'I understand, this is a reminder. Call mom'\n"
                "  GOOD: 'Call mom'\n"
                "- If the command is an action task (summarise, check, run, etc.) — use tools and output only the result.\n"
                "=== END SCHEDULED TASK MODE ===\n"
            )
        return prompt

    def run(self, user_text: str, cwd: str | None = None, memory_context: str = "",
            history: list | None = None, local_available: bool = True,
            source: str = "", step_callback=None, command_id: str | None = None,
            system_prompt: str | None = None) -> dict:
        """Run the agent loop. cwd sets the active project directory for all tool calls.
        Returns dict with speak, display, and optional approval_required. When command_id
        is given and the run pauses for approval, a resume callable is registered so the
        run can continue (server-side) without replaying earlier steps."""
        state = _ClaudeLoopState(
            messages=[*(history or []), {"role": "user", "content": user_text}],
            tool_calls_made=[],
            steps=[],
            total_steps=0,
            last_tool_call=None,
            system_prompt=system_prompt if system_prompt is not None else self._build_system_prompt(cwd, memory_context, source),
            cwd=cwd,
            source=source,
            local_available=local_available,
            command_id=command_id,
            user_text=user_text,
        )
        return self._outer_loop(state, step_callback)

    def resume(self, state: "_ClaudeLoopState", step_callback=None) -> dict:
        """Continue a run paused for approval: finish the paused turn (execute the
        now-approved tool and any remaining blocks), then run the outer loop."""
        content = state.pending_content
        start_idx = state.pending_index
        tool_results = state.pending_results or []
        state.pending_content = None
        paused = self._process_turn(state, content, start_idx, tool_results, step_callback)
        if paused is not None:
            return paused  # paused again on a later block in the same turn
        return self._outer_loop(state, step_callback)

    def _outer_loop(self, state: "_ClaudeLoopState", step_callback) -> dict:
        max_steps = self._config.get("reasoning", {}).get("max_steps_claude", 15)

        wrap_up_step = max_steps - 2

        for _ in range(max_steps):
            if state.total_steps >= wrap_up_step and not state.wrap_up_nudged:
                state.wrap_up_nudged = True
                nudge = (
                    "You are approaching your step limit. Based on everything you have found so far, "
                    "call finalize() now with your best answer — include what you discovered and what "
                    "still needs to be done if the task isn't complete."
                )
                state.messages.append({"role": "user", "content": nudge})

            # Omit delegate_to_local when Ollama is down to avoid wasting a step.
            # Omit notify for scheduled tasks — the scheduler fires the notification itself.
            available_tools = [
                t for t in self._build_tool_list()
                if (t["name"] != "delegate_to_local" or state.local_available)
                and (t["name"] != "notify" or state.source != "scheduled")
            ]

            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": state.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=available_tools,
                messages=state.messages,
            )

            if response.stop_reason in ("end_turn", "max_tokens"):
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                result = format_response(text, state.tool_calls_made)
                result["steps"] = state.steps
                return result

            state.messages.append({"role": "assistant", "content": response.content})
            paused = self._process_turn(state, response.content, 0, [], step_callback)
            if paused is not None:
                return paused

        return {**format_response("I ran out of steps. Please try again.", state.tool_calls_made), "steps": state.steps}

    def _process_turn(self, state: "_ClaudeLoopState", content, start_idx: int,
                      tool_results: list, step_callback) -> dict | None:
        """Execute the tool_use blocks of one assistant turn from start_idx onward.
        Returns an approval dict if it paused, else None (turn done, results appended).
        On pause, all loop-state effects of the pending block are reverted so resume
        re-processes it cleanly once the guardrail trusts the action."""
        stall_detection = self._config.get("reasoning", {}).get("stall_detection", True)
        stalled = False

        for idx in range(start_idx, len(content)):
            block = content[idx]
            if block.type != "tool_use":
                continue

            state.total_steps += 1
            prev_last_tool_call = state.last_tool_call

            # Stall detection: same tool + same input as the previous step. Once
            # stalled, answer every remaining tool_use this turn with a warning
            # tool_result instead of executing it (an unanswered tool_use is invalid).
            if stall_detection and not stalled:
                try:
                    current_call = (block.name, frozenset(block.input.items()))
                except TypeError:
                    current_call = (block.name, block.name)
                if current_call == state.last_tool_call:
                    stalled = True
                else:
                    state.last_tool_call = current_call

            if stalled:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "You already tried this exact action. Please try a different approach or conclude with what you know.",
                })
                continue

            step = {
                "tool": block.name,
                "input_summary": str(block.input)[:100],
                "milestone": _is_milestone(block.name, len(state.steps)),
                "result_summary": "",
            }
            state.steps.append(step)

            if step_callback is not None:
                step_callback({
                    "type": "step",
                    "label": _step_label(block.name),
                    "tool": block.name,
                    "milestone": step["milestone"],
                })

            try:
                if block.name in SCHEDULE_TOOLS:
                    result = self._schedule_result(block, state.source)
                else:
                    result = execute_tool(
                        block.name, block.input,
                        self._shell, self._web, self._code, self._macos,
                        self._guardrails,
                        default_cwd=state.cwd,
                        local_agent=self._local_agent,
                        coding=self._coding,
                        mcp_manager=self._mcp_manager,
                        rag=self._rag,
                    )
                step["result_summary"] = result[:200] if isinstance(result, str) else str(result)[:200]
                state.tool_calls_made.append(block.name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            except ApprovalRequiredError as e:
                # Revert this block's loop-state effects so resume re-processes it as
                # a fresh call (executes once the guardrail trusts the category).
                state.steps.pop()
                state.total_steps -= 1
                state.last_tool_call = prev_last_tool_call
                state.pending_content = content
                state.pending_index = idx
                state.pending_results = tool_results
                if state.command_id:
                    approval_store.register(
                        state.command_id,
                        lambda step_callback=None, _s=state: self.resume(_s, step_callback),
                        {"user_text": state.user_text, "agent": "claude", "model": self._model},
                    )
                return {
                    "speak": None,
                    "display": None,
                    "approval_required": {
                        "tool": e.tool_name,
                        "description": e.description,
                        "tool_use_id": block.id,
                        "category": e.category,
                    },
                    "steps": state.steps,
                }

        state.messages.append({"role": "user", "content": tool_results})
        return None

    def _schedule_result(self, block, source: str) -> str:
        """Run a schedule tool, raising ApprovalRequiredError if create_schedule is gated."""
        if block.name == "create_schedule":
            if source == "scheduled":
                return json.dumps({"error": "create_schedule is not allowed inside a scheduled task"})
            cron = block.input.get("cron")
            run_at = block.input.get("run_at_iso")
            timing = f"cron '{cron}'" if cron else f"at {run_at}"
            label = block.input.get("label", "")
            command = block.input.get("command", "")
            action = Action("schedule_create", f"Schedule '{label}' to run '{command}' ({timing})")
            if self._guardrails.classify(action) == Decision.REQUIRE_APPROVAL:
                raise ApprovalRequiredError("create_schedule", action.description, "schedule_create")
            return json.dumps(_handle_schedule_tool(block.name, block.input))
        return json.dumps(_handle_schedule_tool(block.name, block.input))
