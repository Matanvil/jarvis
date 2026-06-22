import json
import os
import re
import time as _time
from dataclasses import dataclass, field
import httpx
import approval_store
from guardrails import Guardrails
from tools.shell import ShellTool
from tools.web import WebTool
from tools.code import CodeTool
from tools.macos import MacOSTool
from tools._dispatch import execute_tool, format_response
from agent import TOOL_DEFINITIONS, _BASE_SYSTEM_PROMPT, _step_label
from tools._errors import ApprovalRequiredError
from tools.rag import RAGTool
from memory import ProjectMemory

DPO_LOG_PATH = os.path.expanduser("~/.jarvis/logs/dpo_data.jsonl")


def _flush_dpo(record: dict) -> None:
    """Append a DPO record to the log file, creating it if needed."""
    try:
        os.makedirs(os.path.dirname(DPO_LOG_PATH), exist_ok=True)
        with open(DPO_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

_PLANNING_RE = re.compile(
    r"^(now\s+|ok(ay)?[,.]?\s+|sure[,.]?\s+|well[,.]?\s+|alright[,.]?\s+|great[,.]?\s+|perfect[,.]?\s+)?"
    r"(let me\b|i('ll| will)\b|i'm going to\b|i need to\b|i'll start\b|"
    r"i'll check\b|i'll look\b|i'll fetch\b|i'll get\b|i'll find\b|i'll run\b|i'll read\b|"
    r"to do this\b|here's what i|first[,\s]i)",
    re.IGNORECASE,
)


def _is_planning_text(text: str) -> bool:
    """Return True if this looks like planning/intent text without actual content.
    Used to decide whether to nudge the model back toward tool use.
    Empty text is also treated as non-final — the model gave up silently."""
    if not text:
        return True
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    # Check first line (intent at start) or last line (intent at end, e.g. "...Let me start that for you.")
    return bool(_PLANNING_RE.match(lines[0]) or _PLANNING_RE.match(lines[-1]))


_ACTION_TRACE_RE = re.compile(r'^actions\s*:', re.IGNORECASE)


def _is_action_trace(text: str) -> bool:
    """Return True if the model echoed tool results as 'Actions: ...' text instead
    of calling finalize(). This leaks implementation details into the response and
    poisons conversational history — trigger a nudge to call finalize properly."""
    first_line = text.strip().split("\n")[0].strip()
    return bool(_ACTION_TRACE_RE.match(first_line))


# Appended to the base system prompt for local models that need extra guidance
_LOCAL_EXTRA = """
CRITICAL RULES FOR THIS MODEL:
- Respond ONLY in English. Never use Thai, Chinese, Arabic, or any non-English language.
- ALWAYS use the provided tool/function calls to take action. NEVER write tool calls as JSON text in your response.
- If you need to run a command or write a file, call the tool — do not describe it in text.
- NEVER claim to have performed an action without first calling the tool. No tool call = no action taken.
- NEVER respond with "Let me check...", "I'll do...", "First I need to..." or any planning sentence without ALSO calling a tool in the same response. If you need information, call the tool NOW — do not announce that you will.
- To read a file always call file_read — NEVER use shell_run to cat/head/tail a file.
- Be efficient: once you have enough information to answer, stop calling tools and respond. Do NOT keep gathering extra data beyond what the user asked for.
- You have a limited number of tool calls. Use only what is needed — typically 1-3 calls. Do not explore tangents.
- NEVER call mkdir, file_write, or any filesystem-modifying command unless the user explicitly asked you to create or write something. Answering a question does not require creating directories or files.
- NEVER write "Actions:" or echo tool call results as text in your response. After using a tool, call finalize() with a clean answer — do not describe what you did.
"""


def _clean_local_text(text: str) -> str:
    """Strip non-English garbage common in qwen/mistral outputs."""
    # Remove embedded tool call JSON blocks the model leaked into text
    text = re.sub(r'\{["\s]*name["\s]*:[^}]+\}', '', text)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # Remove lines that are mostly non-Latin characters (Thai, Chinese, etc.)
    clean = []
    for line in text.splitlines():
        non_latin = sum(1 for c in line if ord(c) > 0x2E7F)
        if non_latin > 3:
            continue
        clean.append(line)
    return '\n'.join(clean).strip()


class EscalateToCloud(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Escalate to Claude: {reason}")


def _anthropic_to_local_tools(anthropic_tools: list) -> list:
    """Convert Anthropic tool schema format to OpenAI/Ollama format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        }
        for t in anthropic_tools
    ]


# Exclude delegation tools — LocalAgent is the delegate, not the delegator.
# Routing decisions are made by the pre-flight classifier in Router, not by tool calls.
_DELEGATION_TOOLS = {"delegate_to_claude_code", "delegate_to_local", "coding_ask", "coding_plan", "coding_review"}
_LOCAL_SAFE_TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["name"] not in _DELEGATION_TOOLS]
_FINALIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize",
        "description": (
            "Call this when you have enough information to answer the user's request. "
            "Use this instead of continuing to search or gather more data. "
            "Pass your complete answer as the 'answer' parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your complete answer to the user's request.",
                }
            },
            "required": ["answer"],
        },
    },
}

_LOCAL_TOOLS = _anthropic_to_local_tools(_LOCAL_SAFE_TOOL_DEFINITIONS) + [_FINALIZE_TOOL]


_FINALIZE_ANSWER_RE = re.compile(r'"answer"\s*:\s*"')

_JSON_ESCAPE_MAP = {'"': '"', '\\': '\\', '/': '/', 'b': '\b',
                    'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}


def _decode_json_string_chunk(chunk: str, state: dict) -> tuple[str, bool]:
    """Incrementally decode a chunk of a JSON string value.

    state must have 'in_escape' (bool). Mutated in-place.
    Returns (decoded_output, is_done) where is_done=True when the closing
    unescaped '"' is found. Handles \\uXXXX by passing through literally.
    """
    out: list[str] = []
    for ch in chunk:
        if state["in_escape"]:
            out.append(_JSON_ESCAPE_MAP.get(ch, ch))
            state["in_escape"] = False
        elif ch == '\\':
            state["in_escape"] = True
        elif ch == '"':
            return ''.join(out), True  # hit closing quote of the JSON string value
        else:
            out.append(ch)
    return ''.join(out), False


def _stream_call(client: "httpx.Client", url: str, payload: dict,
                 step_callback, composing_state: list | None = None,
                 metrics_out: dict | None = None) -> tuple[dict, str | None]:
    """Make a streaming LLM call. Returns (msg, finish_reason) in same shape as non-streaming.

    Branches on first meaningful chunk:
    - tool_calls: accumulate fragments silently; finalize(answer=...) streams decoded answer tokens
    - content: fire step_callback({"type": "token", "text": token}) per fragment

    If metrics_out is provided it is populated with: ttft_ms, tokens, gen_ms, tok_s.
    Tokens are counted for both text and finalize-answer paths.
    Falls back to non-streaming POST on any exception.
    """
    try:
        with client.stream("POST", url, json={**payload, "stream": True}) as resp:
            resp.raise_for_status()
            full_content = ""
            accumulated: dict[int, dict] = {}  # index → {id, name, arguments}
            finish_reason: str | None = None
            is_text: bool | None = None  # None until first meaningful chunk
            # Per-index state for streaming finalize answer content as decoded tokens.
            # Uses an escape-aware JSON string decoder — no tail buffer needed.
            _fin: dict[int, dict] = {}
            # Latency / throughput metrics (tracked for both text and finalize paths)
            _t_start: float = _time.monotonic()
            _t_first_token: float | None = None
            _token_count: int = 0
            _t_last_token: float = _t_start

            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # Determine branch on first chunk that has content or tool_calls
                if is_text is None:
                    if delta.get("content"):
                        is_text = True
                    elif delta.get("tool_calls"):
                        is_text = False

                if is_text:
                    token = delta.get("content", "")
                    if token:
                        full_content += token
                        _now = _time.monotonic()
                        if _t_first_token is None:
                            _t_first_token = _now
                        _token_count += 1
                        _t_last_token = _now
                        if step_callback:
                            if composing_state is not None and not composing_state[0]:
                                composing_state[0] = True
                                step_callback({"type": "step", "label": "Composing response…", "tool": "text", "milestone": False})
                            step_callback({"type": "token", "text": token})
                elif is_text is False:
                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in accumulated:
                            accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.get("id"):
                            accumulated[idx]["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            accumulated[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            arg_chunk = func["arguments"]
                            accumulated[idx]["arguments"] += arg_chunk

                            # Stream finalize answer as decoded tokens so the HUD
                            # shows the answer building up during generation.
                            if step_callback and accumulated[idx].get("name") == "finalize":
                                if idx not in _fin:
                                    _fin[idx] = {"started": False, "pending": "",
                                                 "in_escape": False, "done": False}
                                fb = _fin[idx]
                                if fb["done"]:
                                    continue
                                if not fb["started"]:
                                    fb["pending"] += arg_chunk
                                    m = _FINALIZE_ANSWER_RE.search(fb["pending"])
                                    if m:
                                        fb["started"] = True
                                        if composing_state is not None and not composing_state[0]:
                                            composing_state[0] = True
                                            step_callback({"type": "step", "label": "Composing response…", "tool": "finalize", "milestone": False})
                                        decoded, done = _decode_json_string_chunk(fb["pending"][m.end():], fb)
                                        if decoded:
                                            step_callback({"type": "token", "text": decoded})
                                            _now = _time.monotonic()
                                            if _t_first_token is None:
                                                _t_first_token = _now
                                            _token_count += 1
                                            _t_last_token = _now
                                        fb["done"] = done
                                else:
                                    decoded, done = _decode_json_string_chunk(arg_chunk, fb)
                                    if decoded:
                                        step_callback({"type": "token", "text": decoded})
                                        _now = _time.monotonic()
                                        if _t_first_token is None:
                                            _t_first_token = _now
                                        _token_count += 1
                                        _t_last_token = _now
                                    fb["done"] = done

        # Populate metrics (captured for both text and finalize paths)
        if metrics_out is not None and _token_count > 0:
            gen_ms = int((_t_last_token - (_t_first_token or _t_start)) * 1000) or 1
            metrics_out["ttft_ms"] = int((_t_first_token - _t_start) * 1000) if _t_first_token else None
            metrics_out["tokens"] = _token_count
            metrics_out["gen_ms"] = gen_ms
            metrics_out["tok_s"] = round(_token_count / (gen_ms / 1000), 1)

        # Reconstruct msg in non-streaming shape
        if accumulated:
            tool_calls = [
                {
                    "id": accumulated[i]["id"],
                    "type": "function",
                    "function": {
                        "name": accumulated[i]["name"],
                        "arguments": accumulated[i]["arguments"],
                    },
                }
                for i in sorted(accumulated)
            ]
            return {"role": "assistant", "content": full_content or "", "tool_calls": tool_calls}, finish_reason
        else:
            return {"role": "assistant", "content": full_content, "tool_calls": None}, finish_reason

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        raise
    except Exception:
        # Fallback: non-streaming retry on parse/streaming errors
        fallback_payload = {k: v for k, v in payload.items() if k != "stream"}
        resp = client.post(url, json=fallback_payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return choice["message"], choice.get("finish_reason")


@dataclass
class _LocalLoopState:
    """Mutable state for one local run, carried across a pause/resume so an
    approved run continues instead of replaying from the original command."""
    messages: list
    tool_calls_made: list
    steps: list
    max_steps: int
    steps_used: int
    last_tool_call: object
    recent_tools: list
    stall_detection: bool
    cwd: str | None
    command_id: str | None
    user_text: str
    pending_tool_calls: list | None = None
    pending_index: int = 0
    no_tool_retries: int = 0
    composing_emitted: bool = False
    gen_metrics: dict = field(default_factory=dict)
    wrap_up_nudged: bool = False
    intent_class: str | None = None
    thinking_disabled: bool = False
    finalize_nudge_count: int = 0
    pending_dpo: dict | None = None


class LocalAgent:
    def __init__(self, config: dict, guardrails: Guardrails, mcp_manager=None):
        self._config = config
        self._guardrails = guardrails
        self._shell = ShellTool()
        self._web = WebTool(brave_api_key=config.get("brave_api_key"))
        self._code = CodeTool()
        self._macos = MacOSTool()
        self._mcp_manager = mcp_manager
        _mem = ProjectMemory()
        self._rag = RAGTool(memory=_mem, ollama_host=config.get("local", {}).get("host", "http://localhost:11434"))
        # Shared client reuses TCP connection to the persistent Ollama process.
        # Note: if timeout is updated via POST /config after construction, the
        # existing client will not pick up the new value — acceptable for now.
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=30.0, pool=5.0)
        )

    def _build_tool_list(self) -> list[dict]:
        """Return local-format tool list with built-ins plus any MCP tools."""
        base = list(_LOCAL_TOOLS)
        if self._mcp_manager is not None:
            mcp_schemas = self._mcp_manager.tool_schemas()
            base = base + _anthropic_to_local_tools(mcp_schemas)
        return base

    def close(self) -> None:
        self._http_client.close()

    def __del__(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass

    @property
    def _host(self) -> str:
        import config as cfg
        return cfg.executor_backend(self._config)[0]

    @property
    def _model(self) -> str:
        import config as cfg
        return cfg.executor_backend(self._config)[1]

    @property
    def _chat_template_kwargs(self) -> dict | None:
        # rapid-mlx handles chat templating internally — don't inject local-specific kwargs
        if self._config.get("local", {}).get("executor_rapid_mlx"):
            return None
        return self._config.get("local", {}).get("executor_chat_template_kwargs")

    @property
    def _routing_mode(self) -> str:
        return self._config.get("local", {}).get("routing_mode", "automatic")

    @property
    def _timeout(self) -> float:
        return float(self._config.get("local", {}).get("timeout_seconds", 300))

    _WINDOW = 5
    _NEAR_DUP_THRESHOLD = 3     # same tool ≥ 3 times in window → redundant

    def run(self, user_text: str, cwd: str | None = None, memory_context: str = "",
            history: list | None = None, step_callback=None,
            intent_class: str | None = None, command_id: str | None = None,
            system_prompt: str | None = None) -> dict:
        """Run local tool-use loop. Raises EscalateToCloud if local executor can't handle it.
        Returns same dict shape as Agent.run(). When command_id is given and the run
        pauses for approval, a resume callable is registered so it can continue
        server-side without replaying earlier steps."""
        if system_prompt is not None:
            system_msg = system_prompt
        else:
            system_msg = _BASE_SYSTEM_PROMPT.format(home=os.path.expanduser("~")) + _LOCAL_EXTRA
            if cwd:
                system_msg += f"\nActive project directory: {cwd}\n"
            if memory_context:
                system_msg += f"\nProject memory: {memory_context}\n"

        max_steps = int(self._config.get("reasoning", {}).get("max_steps_local", 15))

        state = _LocalLoopState(
            messages=[
                {"role": "system", "content": system_msg},
                *(history or []),
                {"role": "user", "content": user_text},
            ],
            tool_calls_made=[],
            steps=[],
            max_steps=max_steps,
            steps_used=0,
            last_tool_call=None,
            recent_tools=[],
            stall_detection=self._config.get("reasoning", {}).get("stall_detection", True),
            cwd=cwd,
            command_id=command_id,
            user_text=user_text,
            intent_class=intent_class,
        )
        return self._outer_loop(state, step_callback)

    def resume(self, state: "_LocalLoopState", step_callback=None) -> dict:
        """Continue a run paused for approval: finish the paused tool batch (execute
        the now-approved tool and any remaining calls), then run the outer loop."""
        tool_calls = state.pending_tool_calls
        start_idx = state.pending_index
        state.pending_tool_calls = None
        kind, result = self._process_tools(state, tool_calls, start_idx, step_callback)
        if kind in ("final", "paused"):
            return result
        return self._outer_loop(state, step_callback)

    def _outer_loop(self, state: "_LocalLoopState", step_callback) -> dict:
        url = f"{self._host}/v1/chat/completions"
        wrap_up_step = state.max_steps - 2
        try:
            while state.steps_used < state.max_steps:
                if state.steps_used >= wrap_up_step and not state.wrap_up_nudged:
                    state.wrap_up_nudged = True
                    nudge = (
                        "You are approaching your step limit. Based on everything you have found so far, "
                        "call finalize() now with your best answer — include what you discovered and what "
                        "still needs to be done if the task isn't complete."
                    )
                    state.messages.append({"role": "user", "content": nudge})

                state.steps_used += 1
                payload = {"model": self._model, "messages": state.messages, "tools": self._build_tool_list()}
                if self._chat_template_kwargs:
                    payload["chat_template_kwargs"] = self._chat_template_kwargs
                thinking_on = state.intent_class == "complex_reasoning" and not state.thinking_disabled
                payload["enable_thinking"] = thinking_on

                composing_state = [state.composing_emitted]
                call_metrics: dict = {}
                msg, finish = _stream_call(self._http_client, url, payload, step_callback, composing_state, call_metrics)
                state.composing_emitted = composing_state[0]
                if call_metrics:
                    state.gen_metrics = call_metrics  # keep last text-generating call's metrics

                if finish == "stop" or not msg.get("tool_calls"):
                    text = _clean_local_text(msg.get("content") or "")
                    if not text and not state.tool_calls_made and finish != "stop":
                        # Thinking may have exhausted the token budget leaving empty content.
                        # Retry once with thinking disabled before escalating.
                        if thinking_on and not state.thinking_disabled:
                            state.thinking_disabled = True
                            state.steps_used -= 1
                            state.composing_emitted = False
                            continue
                        raise EscalateToCloud("Empty response from local executor — server may have crashed")

                    # Genuine final answer: text is not planning-intent or an action trace. Return now.
                    if not _is_planning_text(text) and not _is_action_trace(text):
                        if state.pending_dpo is not None:
                            state.pending_dpo["chosen"] = {"role": "assistant", "content": text}
                            _flush_dpo(state.pending_dpo)
                            state.pending_dpo = None
                        result = format_response(text, state.tool_calls_made)
                        result["steps"] = state.steps
                        result.update(state.gen_metrics)
                        return result

                    # Planning text or action trace returned — nudge.
                    if state.no_tool_retries < 2:
                        state.no_tool_retries += 1
                        state.steps_used -= 1  # don't burn a step on the nudge
                        state.composing_emitted = False  # allow "Composing response…" to refire
                        # Capture DPO record: context before bad response + rejected text.
                        # chosen is filled in when the retry succeeds.
                        if text and state.pending_dpo is None:
                            state.pending_dpo = {
                                "ts": _time.time(),
                                "command_id": state.command_id,
                                "intent_class": state.intent_class,
                                "context": list(state.messages),
                                "rejected": text,
                                "chosen": None,
                            }
                        if text:  # never append an empty assistant message — some backends reject it
                            state.messages.append({"role": "assistant", "content": text})
                        # If thinking was on and produced nothing, disable it for the retry
                        if not text and thinking_on:
                            state.thinking_disabled = True
                        if _is_action_trace(text):
                            nudge = (
                                "Your response echoes tool output as 'Actions: ...' text. "
                                "Do NOT write tool call results in your response. "
                                "Call finalize() now with a clean, direct answer to the user's question."
                            )
                        elif not text:
                            nudge = (
                                "You returned an empty response. Based on the tool results so far, "
                                "either call the next tool to continue, or call finalize() with your answer."
                            )
                        elif state.tool_calls_made:
                            nudge = (
                                "You've made progress but returned planning text instead of calling a tool. "
                                "Continue by calling the next tool now — do not write a text response."
                            )
                        else:
                            nudge = (
                                "You returned text but made no tool calls. "
                                "You MUST call the appropriate tool now — do not write a text response."
                            )
                        if step_callback:
                            step_callback({"type": "clear"})
                        state.messages.append({"role": "user", "content": nudge})
                        continue

                    # 2 retries exhausted — if we did useful work, return it; otherwise escalate.
                    if state.tool_calls_made:
                        result = format_response(text, state.tool_calls_made)
                        result["steps"] = state.steps
                        result.update(state.gen_metrics)
                        return result
                    raise EscalateToCloud("Model returned text without tool calls after 2 retries")

                state.messages.append(msg)
                if state.pending_dpo is not None:
                    state.pending_dpo["chosen"] = msg
                    _flush_dpo(state.pending_dpo)
                    state.pending_dpo = None
                kind, result = self._process_tools(state, msg["tool_calls"], 0, step_callback)
                if kind in ("final", "paused"):
                    return result

        except httpx.ConnectError as e:
            raise EscalateToCloud(f"Local executor unavailable: {e}")
        except httpx.TimeoutException as e:
            raise EscalateToCloud(f"Local executor timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise EscalateToCloud(f"Local executor HTTP error: {e}")

        # Salvage: one final no-tools call to emit whatever the model gathered
        try:
            salvage_resp = self._http_client.post(
                url, json={"model": self._model, "messages": state.messages, "tool_choice": "none"},
            )
            salvage_resp.raise_for_status()
            salvage_text = _clean_local_text(
                salvage_resp.json()["choices"][0]["message"].get("content") or ""
            )
            if salvage_text:
                result = format_response(salvage_text, state.tool_calls_made)
                result["steps"] = state.steps
                return result
        except Exception:
            pass

        result = format_response("I ran out of steps. Please try again.", state.tool_calls_made)
        result["steps"] = state.steps
        return result

    def _process_tools(self, state: "_LocalLoopState", tool_calls: list, start_idx: int,
                       step_callback) -> tuple[str, dict | None]:
        """Process the tool calls of one assistant turn from start_idx. Returns
        ("final", result) if it concluded, ("paused", approval) if it stopped for
        approval, or ("continue", None) when the batch finished (continue the loop).
        On pause, the pending call's loop-state effects are reverted so resume
        re-processes it cleanly once the guardrail trusts the action."""
        url = f"{self._host}/v1/chat/completions"
        for j in range(start_idx, len(tool_calls)):
            tc = tool_calls[j]
            name = tc["function"]["name"]
            call_id = tc["id"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                state.messages.append({"role": "tool", "tool_call_id": call_id,
                                       "content": f"error: malformed tool arguments — {e}. Please retry with valid JSON."})
                continue

            # finalize: model signals it has enough info — return immediately
            if name == "finalize":
                answer = args.get("answer", "") if isinstance(args, dict) else ""
                # Reject finalize() with planning text — model called tools but still
                # wrote an intent sentence instead of a real answer.
                if _is_planning_text(answer) and state.finalize_nudge_count < 2:
                    state.finalize_nudge_count += 1
                    state.steps_used -= 1  # don't count this against the budget
                    if step_callback:
                        step_callback({"type": "clear"})
                    state.messages.append({"role": "user", "content": (
                        "finalize() was called with planning text instead of an actual answer. "
                        "You already ran the tools — call finalize() NOW with the real result "
                        "from those tool calls. Do NOT write 'Let me...', 'I'll...', or any "
                        "planning sentence. Give the user the actual answer."
                    )})
                    continue
                step = {"tool": "finalize", "input_summary": answer[:100],
                        "result_summary": "finalized", "milestone": len(state.steps) == 0}
                state.steps.append(step)
                result = format_response(answer, state.tool_calls_made)
                result["steps"] = state.steps
                return ("final", result)

            # Snapshot for clean revert if this call pauses for approval.
            prev_last_tool_call = state.last_tool_call
            recent_len = len(state.recent_tools)

            # Near-duplicate detection: same tool dominates recent window → salvage
            state.recent_tools.append(name)
            if len(state.recent_tools) > self._WINDOW:
                state.recent_tools.pop(0)
            if state.stall_detection and state.recent_tools.count(name) >= self._NEAR_DUP_THRESHOLD and state.steps:
                state.messages.append({
                    "role": "user",
                    "content": f"You have called '{name}' {state.recent_tools.count(name)} times with similar inputs and similar results. Stop and respond with what you already know.",
                })
                try:
                    nr = self._http_client.post(
                        url, json={"model": self._model, "messages": state.messages, "tool_choice": "none"},
                    )
                    nr.raise_for_status()
                    nr_text = _clean_local_text(nr.json()["choices"][0]["message"].get("content") or "")
                    if nr_text:
                        result = format_response(nr_text, state.tool_calls_made)
                        result["steps"] = state.steps
                        return ("final", result)
                except Exception:
                    pass

            # Exact stall detection: same tool + same args twice in a row → conclude
            if state.stall_detection:
                try:
                    current_call = (name, frozenset(args.items()) if isinstance(args, dict) else str(args))
                except TypeError:
                    current_call = (name, str(args))
                if current_call == state.last_tool_call:
                    state.messages.append({
                        "role": "user",
                        "content": "You already tried this exact action. Please try a different approach or conclude with what you know.",
                    })
                    resp2 = self._http_client.post(
                        url, json={"model": self._model, "messages": state.messages},
                    )
                    resp2.raise_for_status()
                    text = _clean_local_text(resp2.json()["choices"][0]["message"].get("content") or "")
                    result = format_response(text, state.tool_calls_made)
                    result["steps"] = state.steps
                    return ("final", result)
                state.last_tool_call = current_call

            step = {"tool": name, "input_summary": str(args)[:100], "result_summary": "", "milestone": len(state.steps) == 0}
            state.steps.append(step)
            if step_callback is not None:
                step_callback({"type": "step", "label": _step_label(name), "tool": name, "milestone": step["milestone"]})
            try:
                result = execute_tool(name, args, self._shell, self._web, self._code, self._macos, self._guardrails, default_cwd=state.cwd, coding=None, mcp_manager=self._mcp_manager, rag=self._rag)
                step["result_summary"] = result[:200] if isinstance(result, str) else str(result)[:200]
                state.tool_calls_made.append(name)
                state.no_tool_retries = 0  # successful tool call resets retry budget
            except ApprovalRequiredError as e:
                # Revert this call's loop-state effects so resume re-processes it cleanly.
                state.steps.pop()
                state.last_tool_call = prev_last_tool_call
                del state.recent_tools[recent_len:]
                state.pending_tool_calls = tool_calls
                state.pending_index = j
                if state.command_id:
                    approval_store.register(
                        state.command_id,
                        lambda step_callback=None, _s=state: self.resume(_s, step_callback),
                        {"user_text": state.user_text, "agent": "local", "model": self._model},
                    )
                return ("paused", {
                    "speak": None, "display": None,
                    "approval_required": {
                        "tool": e.tool_name,
                        "description": e.description,
                        "tool_use_id": call_id,
                        "category": e.category,
                    },
                    "steps": state.steps,
                })

            state.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        return ("continue", None)
