# Standalone tool dispatch and response formatting shared by Agent and OllamaAgent.
import re
import shlex
import shutil

from guardrails import Decision, Action
from tools._errors import ApprovalRequiredError

# Shell commands that modify the filesystem (create/move/copy/permission changes)
# These are routed to "modify_filesystem" (require_approval) instead of "run_shell" (auto_allow).
_FS_MODIFY_RE = re.compile(r'\b(mkdir|touch|cp|mv|ln|chmod|chown|chgrp|rsync|install)\b')
# Destructive shell commands → "delete_files" (require_approval). Covers explicit
# removal plus less-obvious destroyers: find -delete, git clean, shred, dd.
_FS_DELETE_RE = re.compile(
    r'\b(rm|rmdir|shred)\b'      # explicit removal
    r'|\bfind\b.*-delete\b'      # find ... -delete
    r'|\bgit\s+clean\b'          # git clean removes untracked files
    r'|\bdd\b'                   # dd can overwrite disks/files
)
# run_code snippets that delete files. Gated like a destructive shell command so a
# Python/JS snippet can't delete files under the auto-allowed run_code path while
# shell "rm" requires approval. (File *writes* stay auto-allowed — parity with file_write.)
_CODE_DELETE_RE = re.compile(
    r'os\.(remove|unlink|rmdir|removedirs)\b'
    r'|shutil\.rmtree\b'
    r'|\.unlink\('                                          # Path(...).unlink()
    r'|fs\.(unlink|rm|rmdir|rmSync|rmdirSync|unlinkSync)\b'  # node fs
    r'|File\.(delete|unlink)\b|FileUtils\.rm\b'             # ruby
    r'|\brm\s+-[rf]|\bshred\b'                              # shelling out from code
)


def _shell_guardrail_category(command: str) -> str:
    """Return the most specific guardrail category for a shell command."""
    if _FS_DELETE_RE.search(command):
        return "delete_files"
    if _FS_MODIFY_RE.search(command):
        return "modify_filesystem"
    return "run_shell"


def _code_guardrail_category(code: str) -> str:
    """Return the guardrail category for a run_code snippet. Deletion-capable code
    is gated as delete_files; everything else is auto-allowed under run_code_with_effects."""
    if _CODE_DELETE_RE.search(code):
        return "delete_files"
    return "run_code_with_effects"

# Timeout for delegated Claude Code tasks (complex codebase work can take a while)
_CLAUDE_CODE_TIMEOUT = 300
_MAX_SHELL_TIMEOUT = 600

TOOL_TO_GUARDRAIL_CATEGORY = {
    "shell_run": "run_shell",
    "file_write": "create_files",
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
    "coding_ask": "read_files",
    "coding_plan": "read_files",
    "coding_review": "read_files",
    "index_codebase": "read_files",
    "search_codebase": "read_files",
    "list_plans": "read_files",
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
    coding=None,
    mcp_manager=None,
    rag=None,
) -> str:
    """Dispatch a tool call. Raises ApprovalRequiredError if guardrails block it."""
    if tool_name == "shell_run":
        category = _shell_guardrail_category(tool_input.get("command", ""))
    elif tool_name == "run_code":
        category = _code_guardrail_category(tool_input.get("code", ""))
    elif mcp_manager and mcp_manager.is_mcp_tool(tool_name):
        server, _ = mcp_manager.parse_mcp_tool_name(tool_name)
        category = mcp_manager.guardrail_category(server)
    else:
        category = TOOL_TO_GUARDRAIL_CATEGORY.get(tool_name, "run_shell")
    description = f"{tool_name}: {tool_input}"
    # MCP server configs can use policy values directly as their guardrail field
    # (e.g. "guardrail": "auto_allow"). Short-circuit before the dict lookup so these work.
    if category == "auto_allow":
        pass  # skip guardrails check
    elif category == "require_approval":
        raise ApprovalRequiredError(tool_name, description, category=category)
    else:
        action = Action(category=category, description=description)
        decision = guardrails.classify(action)
        if decision == Decision.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(tool_name, description, category=category)

    # cwd: tool input overrides the request-level default
    cwd = tool_input.get("cwd", default_cwd)

    if tool_name == "shell_run":
        raw_timeout = tool_input.get("timeout")
        timeout = min(int(raw_timeout), _MAX_SHELL_TIMEOUT) if raw_timeout is not None else 30
        r = shell.run(tool_input["command"], cwd=cwd, timeout=timeout)
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
    elif tool_name == "list_plans":
        r = shell.list_plans(tool_input.get("cwd") or cwd or "~")
        if r["error"]:
            return f"error={r['error']}"
        if not r["plans"]:
            return f"No saved plans for this project. Plans directory: {r['plans_dir']}"
        lines = [f"Plans directory: {r['plans_dir']}", ""]
        lines += [f"• {p['filename']}: {p['title']}" for p in r["plans"]]
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
    elif tool_name == "coding_ask":
        if coding is None:
            return "error: coding agent not available"
        r = coding.ask(tool_input["question"], tool_input["cwd"])
        if r["error"]:
            return f"error: {r['error']}"
        return r["answer"]
    elif tool_name == "coding_plan":
        if coding is None:
            return "error: coding agent not available"
        r = coding.plan(tool_input["task"], tool_input["cwd"])
        if r["error"]:
            return f"error: {r['error']}"
        lines = [r["plan_summary"], ""]
        for edit in r["edits"]:
            lines.append(f"File: {edit['file']}")
            lines.append(f"  {edit['description']}")
            lines.append("--- old ---")
            lines.append(edit["old_code"])
            lines.append("--- new ---")
            lines.append(edit["new_code"])
            lines.append("")
        return "\n".join(lines)
    elif tool_name == "coding_review":
        if coding is None:
            return "error: coding agent not available"
        r = coding.review(tool_input["cwd"], tool_input.get("context", ""))
        if r["error"]:
            return f"error: {r['error']}"
        lines = [r["summary"], ""]
        for issue in r["issues"]:
            lines.append(f"[{issue['category'].upper()}] {issue['file']}: {issue['description']}")
            if issue.get("recommendation"):
                lines.append(f"  → {issue['recommendation']}")
        return "\n".join(lines)
    elif tool_name == "index_codebase":
        if rag is None:
            return "error: RAG tool not available"
        repo = tool_input.get("repo_path") or default_cwd
        if not repo:
            return "error: no repo_path specified and no active project directory"
        force = bool(tool_input.get("force", False))
        if not force:
            existing = rag.status(repo)
            if existing["count"] > 0 and not existing["stale"]:
                return (
                    f"Index already up to date: {existing['count']} chunks indexed "
                    f"(last indexed: {existing['indexed_at']}). "
                    f"Use search_codebase to query it, or pass force=true to re-index."
                )
        r = rag.index(repo)
        if r["error"]:
            return f"error: {r['error']}"
        return f"Indexed {r['indexed']} chunks from {repo}"
    elif tool_name == "search_codebase":
        if rag is None:
            return "error: RAG tool not available"
        repo = tool_input.get("repo_path") or default_cwd
        if not repo:
            return "error: no repo_path specified and no active project directory"
        n = int(tool_input.get("n_results", 5))
        r = rag.search(tool_input["query"], repo, n_results=n)
        if r["error"]:
            return f"error: {r['error']}"
        lines = []
        if r.get("stale"):
            lines.append("[Note: index may be stale — consider running index_codebase to refresh]")
        for chunk in r["chunks"]:
            lines.append(f"\n### {chunk['file']}:{chunk['start_line']} ({chunk['chunk_type']}, score={chunk['score']})")
            lines.append(chunk["text"])
        return "\n".join(lines) if lines else "No relevant results found."
    if mcp_manager and mcp_manager.is_mcp_tool(tool_name):
        server, bare_tool = mcp_manager.parse_mcp_tool_name(tool_name)
        return mcp_manager.call_tool(server, bare_tool, tool_input)
    if mcp_manager is None and tool_name.startswith("mcp__"):
        return f"error: MCP tool '{tool_name}' called but no MCP manager is configured"
    return f"unknown tool: {tool_name}"


def format_response(text: str, tool_calls_made: list) -> dict:
    """Format a final agent response for speak and display channels.

    If the model included a VOICE: line, use it as the spoken summary and
    strip it from the displayed text. Falls back to first-sentence heuristic.
    """
    # Extract VOICE: tag if present (model-generated TTS summary); case-insensitive
    # so "VOICe:", "voice:", etc. are all caught.
    voice_line = None
    display_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("VOICE:"):
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
        # Find the first sentence boundary: sentence-ending punctuation followed by a
        # space and an uppercase letter, but only when the period is NOT preceded by a
        # digit (avoids splitting version numbers like "3.5" or "Python 3.11.2").
        m = re.search(r'(?<!\d)([.!?])\s+(?=[A-Z])', plain)
        first = plain[:m.end(1)].strip() if m else plain
        if 10 < len(first) <= 140:
            speak = first
        elif 0 < len(plain) <= 140:
            speak = plain
        else:
            speak = (plain[:140].rsplit(" ", 1)[0] + "…") if len(plain) > 140 else plain
        return {"speak": speak, "display": display}
    return {"speak": display, "display": display}
