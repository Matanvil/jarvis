"""Gemma 4 31B executor benchmark — 50 cases across all tool types.

Usage:
    cd jarvis-core && source .venv/bin/activate
    python benchmark_gemma.py
    python benchmark_gemma.py --models gemma4:31b,qwen3-coder:30b
    python benchmark_gemma.py --timeout 120 --num-ctx 16384
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx


# ---------------------------------------------------------------------------
# Tools definition (mirrors Jarvis tool set)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell_run",
            "description": "Run a shell command on macOS",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute a code snippet and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "description": "Programming language (python, bash, etc.)"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open a macOS application by name",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Application name"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify",
            "description": "Send a macOS system notification",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["title", "message"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are Jarvis, a macOS AI assistant. You act by calling tools — you NEVER respond with plain text alone.

ABSOLUTE RULES — no exceptions:
1. You MUST call a tool for every single request. A text-only response is always wrong.
2. If a task involves multiple steps, call the FIRST tool immediately. Do not explain the plan.
3. The user cannot execute anything themselves — you are their hands. If you do not call a tool, nothing happens.
4. NEVER say "I would run..." or "You can use..." — just call the tool.

Tool routing — always use the most direct tool:
- Shell commands, file listings, git operations, process info, system info (date, time, disk, memory, uptime), finding files by attribute (size, age, pattern), counting lines, grepping — ALL use shell_run
- Web lookups, current events, documentation, prices, weather → web_search
- Execute code that produces output (Python, JS, TypeScript, Ruby, Swift) → run_code
- Open a macOS application → open_app
- Read a file's contents → file_read  (NEVER use shell_run to cat/head/tail a file — ALWAYS use file_read)
- Write or create a file → file_write
- macOS system notification → notify

Critical examples — call the tool immediately:
- "find Python files modified today" → shell_run(command="find . -name '*.py' -mtime -1")
- "count lines of code" → shell_run(command="find . -name '*.py' | xargs wc -l")
- "show git log" → shell_run(command="git log --oneline -10")
- "what time is it" → shell_run(command="date")
- "write a script to /tmp/x.py" → file_write(path="/tmp/x.py", content="...")
- "run a function that reverses a string" → run_code(code="def reverse(s): return s[::-1]\\nprint(reverse('hello'))")
- "read the requirements.txt file" → file_read(path="requirements.txt")
"""

# ---------------------------------------------------------------------------
# 50 test cases
# Format: (command, expected_tool, category, notes)
# categories: simple | multi_step | reasoning | edge
# ---------------------------------------------------------------------------

CASES = [
    # ── shell_run — simple (10) ─────────────────────────────────────────────
    ("list files in ~/Downloads",                       "shell_run",        "simple",     ""),
    ("run git status in ~/dev/jarvis",                  "shell_run",        "simple",     ""),
    ("run the test suite with pytest",                  "shell_run",        "simple",     ""),
    ("execute this bash script: echo hello",            "shell_run",        "simple",     ""),
    ("show all running Python processes",               "shell_run",        "simple",     ""),
    ("get the current macOS version",                   "shell_run",        "simple",     ""),
    ("show disk usage of my home directory",            "shell_run",        "simple",     ""),
    ("list all git branches in the project",            "shell_run",        "simple",     ""),
    ("check if port 8765 is currently in use",          "shell_run",        "simple",     ""),
    ("show currently installed Homebrew packages",      "shell_run",        "simple",     ""),

    # ── shell_run — multi-step / reasoning (10) ────────────────────────────
    ("find all Python files modified in the last 24 hours",                         "shell_run", "multi_step", "needs find -mtime"),
    ("search for all TODO comments recursively in the src directory",               "shell_run", "multi_step", "grep -r"),
    ("count the total lines of code in jarvis-core, excluding the venv",           "shell_run", "multi_step", "find + wc"),
    ("show the 5 largest files in ~/Downloads sorted by size",                     "shell_run", "multi_step", "find + sort"),
    ("find and list all .log files older than 7 days in ~/.jarvis/logs",           "shell_run", "multi_step", "find -mtime +7"),
    ("show which process is using the most CPU right now",                         "shell_run", "reasoning",  "ps or top"),
    ("check if the Python FastAPI server on port 8765 is responding",              "shell_run", "reasoning",  "curl localhost"),
    ("show the git log of the last 10 commits with author and date",               "shell_run", "reasoning",  "git log format"),
    ("list all environment variables that contain the word PATH",                  "shell_run", "reasoning",  "env | grep"),
    ("show all open network connections on this machine",                          "shell_run", "reasoning",  "netstat or lsof"),

    # ── web_search (8) ─────────────────────────────────────────────────────
    ("search the web for FastAPI documentation",                    "web_search", "simple",   ""),
    ("check the weather online",                                    "web_search", "simple",   ""),
    ("search for Python list sorting best practices",               "web_search", "simple",   ""),
    ("find the latest Claude API pricing from Anthropic",           "web_search", "simple",   ""),
    ("look up how to configure Ollama to use a custom model",       "web_search", "reasoning",""),
    ("search for how to resolve a git merge conflict",              "web_search", "simple",   ""),
    ("find out what's new in Python 3.13",                         "web_search", "simple",   ""),
    ("search for benchmarks comparing llama3 vs gemma on coding",  "web_search", "reasoning",""),

    # ── run_code (7) ────────────────────────────────────────────────────────
    ("run this Python snippet: print(1+1)",                                         "run_code", "simple",   ""),
    ("calculate the fibonacci sequence up to index 10 in Python",                  "run_code", "reasoning",""),
    ("run a Python script that prints all even numbers between 1 and 20",          "run_code", "simple",   ""),
    ("execute a quick Python check of the current Python version",                 "run_code", "simple",   ""),
    ("run: import json; print(json.dumps({'status': 'ok', 'version': 3}))",        "run_code", "simple",   ""),
    ("write and run a Python function that reverses a string and tests it",        "run_code", "reasoning",""),
    ("run a bash snippet that prints the current date and hostname",               "shell_run", "simple",   "bash → shell_run"),

    # ── open_app (5) ───────────────────────────────────────────────────────
    ("open Safari",                    "open_app", "simple",   ""),
    ("open the Finder app",            "open_app", "simple",   ""),
    ("open VS Code",                   "open_app", "simple",   ""),
    ("launch the Terminal application","open_app", "simple",   ""),
    ("open System Preferences",        "open_app", "simple",   ""),

    # ── file_read (5) ──────────────────────────────────────────────────────
    ("read the file README.md",                         "file_read", "simple",   ""),
    ("show the contents of /tmp/test.txt",              "file_read", "simple",   ""),
    ("read my Jarvis config file at ~/.jarvis/config.json","file_read","simple",  ""),
    ("show me the contents of jarvis-core/server.py",  "file_read", "simple",   ""),
    ("read the requirements.txt file",                 "file_read", "simple",   ""),

    # ── file_write (5) ─────────────────────────────────────────────────────
    ("write 'hello world' to /tmp/test.txt",                                    "file_write", "simple",   ""),
    ("create a new file called notes.md with content 'my notes'",              "file_write", "simple",   ""),
    ("save the text 'benchmark run complete' to /tmp/benchmark_log.txt",       "file_write", "simple",   ""),
    ("create a file at /tmp/ideas.txt with the content 'brainstorm session'",  "file_write", "simple",   ""),
    ("write a simple Python hello world script to /tmp/hello.py",              "file_write", "reasoning","needs code content"),

    # ── notify (3) ─────────────────────────────────────────────────────────
    ("notify me that the build finished",               "notify", "simple",   ""),
    ("send a notification that deployment is complete", "notify", "simple",   ""),
    ("alert me with a macOS notification that the tests passed", "notify", "simple", ""),

    # ── edge / ambiguous (7) ───────────────────────────────────────────────
    ("what time is it?",                                "shell_run",   "edge", "date command"),
    ("show me today's date and day of the week",       "shell_run",   "edge", "date"),
    ("check if ollama is running on this machine",     "shell_run",   "edge", "ps or curl"),
    ("what's my current working directory?",           "shell_run",   "edge", "pwd"),
    ("show me the memory usage of the system",         "shell_run",   "edge", "vm_stat or free"),
    ("how much free disk space do I have?",            "shell_run",   "edge", "df -h"),
    ("is the jarvis-core server currently running?",   "shell_run",   "edge", "ps or curl"),
]

assert len(CASES) >= 50, f"Expected at least 50 cases, got {len(CASES)}"

# Required args per tool — used for argument quality check
REQUIRED_ARGS = {
    "shell_run":   ["command"],
    "web_search":  ["query"],
    "run_code":    ["code"],
    "open_app":    ["name"],
    "file_read":   ["path"],
    "file_write":  ["path", "content"],
    "notify":      ["title", "message"],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    num_ctx: int
    total: int = 0
    tool_correct: int = 0
    args_valid: int = 0          # required args present + non-empty
    completed: int = 0           # got any tool call at all
    bad_json: int = 0
    errors: int = 0
    latencies: list = field(default_factory=list)
    completion_tok_s: list = field(default_factory=list)
    # per-category breakdown
    category_correct: dict = field(default_factory=lambda: {
        "simple": [0, 0], "multi_step": [0, 0], "reasoning": [0, 0], "edge": [0, 0]
    })

    @property
    def tool_acc_pct(self) -> float:
        return (self.tool_correct / self.total * 100) if self.total else 0

    @property
    def args_acc_pct(self) -> float:
        return (self.args_valid / self.total * 100) if self.total else 0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def avg_tok_s(self) -> float:
        return statistics.mean(self.completion_tok_s) if self.completion_tok_s else 0

    @property
    def p50_tok_s(self) -> float:
        return statistics.median(self.completion_tok_s) if self.completion_tok_s else 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "num_ctx": self.num_ctx,
            "total": self.total,
            "tool_correct": self.tool_correct,
            "args_valid": self.args_valid,
            "completed": self.completed,
            "bad_json": self.bad_json,
            "errors": self.errors,
            "tool_acc_pct": self.tool_acc_pct,
            "args_acc_pct": self.args_acc_pct,
            "avg_latency_ms": self.avg_latency,
            "p50_latency_ms": self.p50,
            "p95_latency_ms": self.p95,
            "avg_completion_tok_s": self.avg_tok_s,
            "p50_completion_tok_s": self.p50_tok_s,
            "category_correct": self.category_correct,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model: str,
    host: str,
    timeout: float,
    num_ctx: int,
    cases: list[tuple[str, str, str, str]],
    chat_template_kwargs: Optional[dict] = None,
) -> Result:
    result = Result(model=model, num_ctx=num_ctx, total=len(cases))
    width = max(len(c[0]) for c in cases)

    with httpx.Client(timeout=timeout) as client:
        for idx, (command, expected_tool, category, _notes) in enumerate(cases, 1):
            t0 = time.monotonic()
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": command},
                    ],
                    "tools": TOOLS,
                    "options": {"num_ctx": num_ctx},
                }
                if chat_template_kwargs is not None:
                    payload["chat_template_kwargs"] = chat_template_kwargs

                resp = client.post(
                    f"{host}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
                message = body["choices"][0]["message"]
                tool_calls = message.get("tool_calls")
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                result.latencies.append(elapsed_ms)
                result.errors += 1
                print(f"  [{idx:02d}/{len(cases):02d}] ERR  {command[:60]:<60}  ({elapsed_ms:.0f}ms)  error: {exc}")
                continue

            elapsed_ms = (time.monotonic() - t0) * 1000
            result.latencies.append(elapsed_ms)
            usage = body.get("usage") or {}
            completion_tokens = usage.get("completion_tokens") or 0
            tok_s = (completion_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 and completion_tokens else 0
            if tok_s:
                result.completion_tok_s.append(tok_s)

            # Normalise text-based tool calls (qwen-style fallback)
            if not tool_calls:
                raw = (message.get("content") or "").strip()
                if raw.startswith("{"):
                    try:
                        parsed = json.loads(raw)
                        if "name" in parsed:
                            tool_calls = [{"function": {
                                "name": parsed["name"],
                                "arguments": json.dumps(parsed.get("arguments", {})),
                            }}]
                    except json.JSONDecodeError:
                        pass

            if not tool_calls:
                result.bad_json += 1
                print(f"  [{idx:02d}/{len(cases):02d}] MISS {command[:60]:<60}  ({elapsed_ms:.0f}ms)  no tool call (expected {expected_tool})")
                continue

            result.completed += 1
            got_tool = tool_calls[0]["function"]["name"]
            tool_ok = got_tool == expected_tool

            # Argument quality check
            try:
                args = json.loads(tool_calls[0]["function"]["arguments"])
                required = REQUIRED_ARGS.get(got_tool, [])
                args_ok = all(args.get(k, "").strip() for k in required)
            except (json.JSONDecodeError, AttributeError):
                args = {}
                args_ok = False

            if tool_ok:
                result.tool_correct += 1
                result.category_correct[category][0] += 1
            if tool_ok and args_ok:
                result.args_valid += 1

            result.category_correct[category][1] += 1

            status = "OK  " if tool_ok else "FAIL"
            arg_flag = "args✓" if args_ok else "args✗"
            tok_s_text = f"{tok_s:.1f} tok/s" if tok_s else "tok/s n/a"
            print(
                f"  [{idx:02d}/{len(cases):02d}] {status} {command[:60]:<60}  ({elapsed_ms:.0f}ms)  "
                f"got={got_tool}  exp={expected_tool}  {arg_flag}  {tok_s_text}"
            )

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(results: list[Result], total_cases: int) -> None:
    print("\n" + "=" * 90)
    print(f"EXECUTOR BENCHMARK — {total_cases} CASES")
    print("=" * 90)

    for r in sorted(results, key=lambda x: x.tool_acc_pct, reverse=True):
        print(f"\nModel: {r.model}  (num_ctx={r.num_ctx})")
        print(f"  Tool accuracy:   {r.tool_correct}/{r.total}  ({r.tool_acc_pct:.1f}%)")
        print(f"  Args quality:    {r.args_valid}/{r.total}   ({r.args_acc_pct:.1f}%)")
        print(f"  Completed calls: {r.completed}/{r.total}")
        print(f"  Bad/no tool:     {r.bad_json}   Errors: {r.errors}")
        print(f"  Latency:         avg={r.avg_latency:.0f}ms  p50={r.p50:.0f}ms  p95={r.p95:.0f}ms")
        print(f"  Throughput:      avg={r.avg_tok_s:.1f} tok/s  p50={r.p50_tok_s:.1f} tok/s")
        print()
        print("  Category breakdown:")
        for cat, (correct, total) in r.category_correct.items():
            bar = "█" * correct + "░" * (total - correct)
            pct = (correct / total * 100) if total else 0
            print(f"    {cat:<12} {correct:>2}/{total:<2}  {pct:5.1f}%  {bar}")

    print("\n" + "=" * 90)
    if results:
        best = max(results, key=lambda x: x.tool_acc_pct)
        print(f"Best: {best.model} — {best.tool_acc_pct:.1f}% tool accuracy, {best.p50:.0f}ms p50")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 31B executor benchmark (50 cases)")
    parser.add_argument(
        "--models",
        default="gemma4:31b",
        help="Comma-separated model tags (default: gemma4:31b)",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--num-ctx", type=int, default=8192,
        help="Context window size passed to Ollama (default: 8192). "
             "Use 32768 for coding/multi-step tasks.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Run only the first N benchmark cases (default: all cases).",
    )
    parser.add_argument(
        "--chat-template-args", default="",
        help='Optional JSON passed as chat_template_kwargs, e.g. \'{"enable_thinking": false}\'',
    )
    parser.add_argument(
        "--output", default="",
        help="Optional path to save benchmark results JSON.",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = CASES[: args.limit] if args.limit > 0 else CASES
    chat_template_kwargs = json.loads(args.chat_template_args) if args.chat_template_args else None

    results = []
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model}  (num_ctx={args.num_ctx}, cases={len(cases)})")
        print(f"{'=' * 60}")
        results.append(
            run_benchmark(
                model,
                args.host,
                args.timeout,
                args.num_ctx,
                cases,
                chat_template_kwargs=chat_template_kwargs,
            )
        )

    print_report(results, total_cases=len(cases))

    if args.output:
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": args.host,
            "limit": args.limit,
            "num_ctx": args.num_ctx,
            "chat_template_kwargs": chat_template_kwargs,
            "results": [r.to_dict() for r in results],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
