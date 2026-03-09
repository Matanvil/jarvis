# Standalone tool dispatch and response formatting shared by Agent and OllamaAgent.
import re
import shlex
import shutil

from guardrails import Decision, Action
from tools._errors import ApprovalRequiredError

# Shell commands that modify the filesystem (create/move/copy/permission changes)
# These are routed to "modify_filesystem" (require_approval) instead of "run_shell" (auto_allow).
_FS_MODIFY_RE = re.compile(r'\b(mkdir|touch|cp|mv|ln|chmod|chown|chgrp|rsync|install)\b')
# rm/rmdir already map to delete_files via TOOL_TO_GUARDRAIL_CATEGORY for file tools,
# but also catch them here for shell_run commands
_FS_DELETE_RE = re.compile(r'\b(rm|rmdir)\b')


def _shell_guardrail_category(command: str) -> str:
    """Return the most specific guardrail category for a shell command."""
    if _FS_DELETE_RE.search(command):
        return "delete_files"
    if _FS_MODIFY_RE.search(command):
        return "modify_filesystem"
    return "run_shell"

# Timeout for delegated Claude Code tasks (complex codebase work can take a while)
_CLAUDE_CODE_TIMEOUT = 300

TOOL_TO_GUARDRAIL_CATEGORY = {
    "shell_run": "run_shell",
    "file_write": "edit_files",
    "file_edit": "edit_files",
    "file_read": "read_files",
    "find_files": "read_files",
    "search_content": "read_files",
    "list_dir": "read_files",
    "web_search": "web_search",
    "web_fetch": "web_search",
    "run_code": "run_code_with_effects",
    "open_app": "open_apps",
    "notify": "open_apps",
    "delegate_to_claude_code": "run_code_with_effects",
    "delegate_to_local": "run_shell",
}


def _claude_code_available() -> bool:
    """Return True if the `claude` CLI is installed and reachable."""
    return shutil.which("claude") is not None


def execute_tool(
    tool_name: str,
    tool_input: dict,
    shell,
    web,
    code,
    macos,
    guardrails,
    default_cwd: str | None = None,
    local_agent=None,
) -> str:
    """Dispatch a tool call. Raises ApprovalRequiredError if guardrails block it."""
    if tool_name == "shell_run":
        category = _shell_guardrail_category(tool_input.get("command", ""))
    else:
        category = TOOL_TO_GUARDRAIL_CATEGORY.get(tool_name, "run_shell")
    description = f"{tool_name}: {tool_input}"
    action = Action(category=category, description=description)
    decision = guardrails.classify(action)

    if decision == Decision.REQUIRE_APPROVAL:
        raise ApprovalRequiredError(tool_name, description, category=category)

    # cwd: tool input overrides the request-level default
    cwd = tool_input.get("cwd", default_cwd)

    if tool_name == "shell_run":
        r = shell.run(tool_input["command"], cwd=cwd)
        return f"exit_code={r['exit_code']}\nstdout={r['stdout']}\nstderr={r['stderr']}"
    elif tool_name == "file_write":
        r = shell.write_file(tool_input["path"], tool_input["content"])
        return f"success={r['success']}" if r["success"] else f"error={r['error']}"
    elif tool_name == "file_edit":
        r = shell.file_edit(
            tool_input["path"],
            tool_input["old_string"],
            tool_input["new_string"],
            replace_all=tool_input.get("replace_all", False),
        )
        if r["success"]:
            return f"success: replaced {r['replaced']} occurrence(s)"
        return f"error: {r['error']}"
    elif tool_name == "file_read":
        r = shell.read_file(tool_input["path"])
        return r["content"] if r["content"] is not None else f"error={r['error']}"
    elif tool_name == "find_files":
        r = shell.find_files(
            tool_input["pattern"],
            directory=tool_input.get("directory", "~"),
            file_type=tool_input.get("file_type", "any"),
        )
        if r["error"]:
            return f"error={r['error']}"
        if not r["matches"]:
            return "No files found matching that pattern."
        return f"{r['count']} match(es):\n" + "\n".join(r["matches"])
    elif tool_name == "search_content":
        r = shell.search_content(
            tool_input["pattern"],
            directory=tool_input.get("directory", "~"),
            file_glob=tool_input.get("file_glob", "*"),
            case_sensitive=tool_input.get("case_sensitive", False),
        )
        if r["error"]:
            return f"error={r['error']}"
        if not r["files"]:
            return "No matches found."
        lines = [f"{r['count']} file(s) matched:"]
        for s in r["snippets"]:
            lines.append(f"\n{s['file']}:")
            lines.extend(f"  {l}" for l in s["lines"])
        return "\n".join(lines)
    elif tool_name == "list_dir":
        r = shell.list_dir(tool_input["path"])
        if r["error"]:
            return f"error={r['error']}"
        entries = r["entries"]
        if not entries:
            return "Directory is empty."
        lines = [f"{'[dir] ' if e['is_dir'] else '[file]'} {e['name']}" for e in sorted(entries, key=lambda e: (not e["is_dir"], e["name"]))]
        return "\n".join(lines)
    elif tool_name == "web_search":
        results = web.search(tool_input["query"])
        return "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)
    elif tool_name == "web_fetch":
        r = web.fetch_page(tool_input["url"])
        return r["text"] if r["text"] is not None else f"error={r['error']}"
    elif tool_name == "run_code":
        r = code.run_snippet(tool_input["code"], tool_input["language"], cwd=cwd)
        return f"exit_code={r['exit_code']}\nstdout={r['stdout']}\nstderr={r['stderr']}"
    elif tool_name == "open_app":
        r = macos.open_app(tool_input["app_name"])
        return "opened" if r["success"] else f"error={r['error']}"
    elif tool_name == "notify":
        r = macos.notify(tool_input["title"], tool_input["body"])
        return "sent" if r["success"] else f"error={r['error']}"
    elif tool_name == "delegate_to_local":
        if local_agent is None:
            return "error: no local agent available for delegation"
        task = tool_input["task"]
        cwd = tool_input.get("cwd", default_cwd)
        try:
            result = local_agent.run(task, cwd=cwd)
        except Exception as e:
            # Ollama unavailable or escalation — return error so Claude can handle it directly
            return f"error: local agent unavailable ({e})"
        return result.get("display") or result.get("speak") or "completed"
    elif tool_name == "delegate_to_claude_code":
        if not _claude_code_available():
            return "error: Claude Code CLI ('claude') is not installed. Run: npm install -g @anthropic-ai/claude-code"
        task = tool_input["task"]
        r = shell.run(
            f"claude --print {shlex.quote(task)}",
            cwd=cwd,
            timeout=_CLAUDE_CODE_TIMEOUT,
        )
        return r["stdout"] or r["stderr"] or f"exit_code={r['exit_code']}"
    return f"unknown tool: {tool_name}"


def format_response(text: str, tool_calls_made: list) -> dict:
    """Format a final agent response for speak and display channels.

    If the model included a VOICE: line, use it as the spoken summary and
    strip it from the displayed text. Falls back to first-sentence heuristic.
    """
    # Extract VOICE: tag if present (model-generated TTS summary)
    voice_line = None
    display_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("VOICE:"):
            voice_line = stripped[len("VOICE:"):].strip()
        else:
            display_lines.append(line)

    display = "\n".join(display_lines).strip()

    if voice_line:
        return {"speak": voice_line, "display": display or voice_line}

    # Fallback: first plain-text sentence heuristic
    has_code = "```" in display
    is_long = len(display) >= 150
    if has_code or is_long:
        plain = display.split("```")[0].strip()
        first = plain.split(".")[0].strip()
        if 10 < len(first) <= 140:
            speak = first + "."
        elif 0 < len(plain) <= 140:
            speak = plain
        else:
            speak = (plain[:140].rsplit(" ", 1)[0] + "…") if len(plain) > 140 else plain
        return {"speak": speak, "display": display}
    return {"speak": display, "display": display}
