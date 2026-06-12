import json
import os
import re
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

# Appended to the base system prompt for local models that need extra guidance
_OLLAMA_EXTRA = """
CRITICAL RULES FOR THIS MODEL:
- Respond ONLY in English. Never use Thai, Chinese, Arabic, or any non-English language.
- ALWAYS use the provided tool/function calls to take action. NEVER write tool calls as JSON text in your response.
- If you need to run a command or write a file, call the tool — do not describe it in text.
- NEVER claim to have performed an action without first calling the tool. No tool call = no action taken.
- To read a file always call file_read — NEVER use shell_run to cat/head/tail a file.
- Be efficient: once you have enough information to answer, stop calling tools and respond. Do NOT keep gathering extra data beyond what the user asked for.
- You have a limited number of tool calls. Use only what is needed — typically 1-3 calls. Do not explore tangents.
- CODING TOOLS RULE: After calling coding_ask, coding_plan, or coding_review — call finalize IMMEDIATELY. These tools return complete answers. Do NOT call shell_run, file_read, list_dir, find_files, or any other tool after them. One coding tool call → finalize. That is the entire sequence.
- NEVER call mkdir, file_write, or any filesystem-modifying command unless the user explicitly asked you to create or write something. Answering a question does not require creating directories or files.
"""


def _clean_ollama_text(text: str) -> str:
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


def _anthropic_to_ollama_tools(anthropic_tools: list) -> list:
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


# Exclude delegation tools — OllamaAgent is the delegate, not the delegator.
# Routing decisions are made by the pre-flight classifier in Router, not by tool calls.
_DELEGATION_TOOLS = {"delegate_to_claude_code", "delegate_to_local"}
_OLLAMA_SAFE_TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["name"] not in _DELEGATION_TOOLS]
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

_OLLAMA_TOOLS = _anthropic_to_ollama_tools(_OLLAMA_SAFE_TOOL_DEFINITIONS) + [_FINALIZE_TOOL]


def _stream_call(client: "httpx.Client", url: str, payload: dict,
                 step_callback) -> tuple[dict, str | None]:
    """Make a streaming LLM call. Returns (msg, finish_reason) in same shape as non-streaming.

    Branches on first meaningful chunk:
    - tool_calls: accumulate fragments silently, no visual output
    - content: fire step_callback({"type": "token", "text": token}) for each token

    Falls back to non-streaming POST on any exception.
    """
    try:
        with client.stream("POST", url, json={**payload, "stream": True}) as resp:
            resp.raise_for_status()
            full_content = ""
            accumulated: dict[int, dict] = {}  # index → {id, name, arguments}
            finish_reason: str | None = None
            is_text: bool | None = None  # None until first meaningful chunk

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
                        if step_callback:
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
                            accumulated[idx]["arguments"] += func["arguments"]

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
class _OllamaLoopState:
    """Mutable state for one Ollama run, carried across a pause/resume so an
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


class OllamaAgent:
    def __init__(self, config: dict, guardrails: Guardrails, mcp_manager=None):
        self._config = config
        self._guardrails = guardrails
        self._shell = ShellTool()
        self._web = WebTool(brave_api_key=config.get("brave_api_key"))
        self._code = CodeTool()
        self._macos = MacOSTool()
        from tools.coding_agent import CodingAgentTool
        self._coding = CodingAgentTool(config)
        self._mcp_manager = mcp_manager
        # Shared client reuses TCP connection to the persistent Ollama process.
        # Note: if timeout is updated via POST /config after construction, the
        # existing client will not pick up the new value — acceptable for now.
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=30.0, pool=5.0)
        )

    def _build_tool_list(self) -> list[dict]:
        """Return Ollama-format tool list with built-ins plus any MCP tools."""
        base = list(_OLLAMA_TOOLS)
        if self._mcp_manager is not None:
            mcp_schemas = self._mcp_manager.tool_schemas()
            base = base + _anthropic_to_ollama_tools(mcp_schemas)
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
        # rapid-mlx handles chat templating internally — don't inject Ollama-specific kwargs
        if self._config.get("ollama", {}).get("executor_rapid_mlx"):
            return None
        return self._config.get("ollama", {}).get("executor_chat_template_kwargs")

    @property
    def _routing_mode(self) -> str:
        return self._config.get("ollama", {}).get("routing_mode", "ollama_first")

    @property
    def _timeout(self) -> float:
        return float(self._config.get("ollama", {}).get("timeout_seconds", 300))

    _WINDOW = 5
    _NEAR_DUP_THRESHOLD = 3     # same tool ≥ 3 times in window → redundant

    def run(self, user_text: str, cwd: str | None = None, memory_context: str = "",
            history: list | None = None, step_callback=None,
            intent_class: str | None = None, command_id: str | None = None) -> dict:
        """Run Ollama tool-use loop. Raises EscalateToCloud if Ollama can't handle it.
        Returns same dict shape as Agent.run(). When command_id is given and the run
        pauses for approval, a resume callable is registered so it can continue
        server-side without replaying earlier steps."""
        system_msg = _BASE_SYSTEM_PROMPT.format(home=os.path.expanduser("~")) + _OLLAMA_EXTRA
        if cwd:
            system_msg += f"\nActive project directory: {cwd}\n"
        if memory_context:
            system_msg += f"\nProject memory: {memory_context}\n"

        budgets = self._config.get("reasoning", {}).get("step_budgets", {})
        if intent_class and intent_class in budgets:
            max_steps = int(budgets[intent_class])
        else:
            max_steps = int(self._config.get("reasoning", {}).get("max_steps_ollama", 10))

        state = _OllamaLoopState(
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
        )
        return self._outer_loop(state, step_callback)

    def resume(self, state: "_OllamaLoopState", step_callback=None) -> dict:
        """Continue a run paused for approval: finish the paused tool batch (execute
        the now-approved tool and any remaining calls), then run the outer loop."""
        tool_calls = state.pending_tool_calls
        start_idx = state.pending_index
        state.pending_tool_calls = None
        kind, result = self._process_tools(state, tool_calls, start_idx, step_callback)
        if kind in ("final", "paused"):
            return result
        return self._outer_loop(state, step_callback)

    def _outer_loop(self, state: "_OllamaLoopState", step_callback) -> dict:
        url = f"{self._host}/v1/chat/completions"
        try:
            while state.steps_used < state.max_steps:
                state.steps_used += 1
                payload = {"model": self._model, "messages": state.messages, "tools": self._build_tool_list()}
                if self._chat_template_kwargs:
                    payload["chat_template_kwargs"] = self._chat_template_kwargs

                msg, finish = _stream_call(self._http_client, url, payload, step_callback)

                if finish == "stop" or not msg.get("tool_calls"):
                    text = _clean_ollama_text(msg.get("content") or "")
                    if not text and not state.tool_calls_made and finish != "stop":
                        raise EscalateToCloud("Empty response from local executor — server may have crashed")
                    result = format_response(text, state.tool_calls_made)
                    result["steps"] = state.steps
                    return result

                state.messages.append(msg)
                kind, result = self._process_tools(state, msg["tool_calls"], 0, step_callback)
                if kind in ("final", "paused"):
                    return result

        except httpx.ConnectError as e:
            raise EscalateToCloud(f"Ollama unavailable: {e}")
        except httpx.TimeoutException as e:
            raise EscalateToCloud(f"Ollama timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise EscalateToCloud(f"Ollama HTTP error: {e}")

        # Salvage: one final no-tools call to emit whatever the model gathered
        try:
            salvage_resp = self._http_client.post(
                url, json={"model": self._model, "messages": state.messages, "tool_choice": "none"},
            )
            salvage_resp.raise_for_status()
            salvage_text = _clean_ollama_text(
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

    def _process_tools(self, state: "_OllamaLoopState", tool_calls: list, start_idx: int,
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
                    nr_text = _clean_ollama_text(nr.json()["choices"][0]["message"].get("content") or "")
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
                    text = _clean_ollama_text(resp2.json()["choices"][0]["message"].get("content") or "")
                    result = format_response(text, state.tool_calls_made)
                    result["steps"] = state.steps
                    return ("final", result)
                state.last_tool_call = current_call

            step = {"tool": name, "input_summary": str(args)[:100], "result_summary": "", "milestone": len(state.steps) == 0}
            state.steps.append(step)
            if step_callback is not None:
                step_callback({"type": "step", "label": _step_label(name), "tool": name, "milestone": step["milestone"]})
            try:
                result = execute_tool(name, args, self._shell, self._web, self._code, self._macos, self._guardrails, default_cwd=state.cwd, coding=self._coding)
                step["result_summary"] = result[:120] if isinstance(result, str) else str(result)[:120]
                state.tool_calls_made.append(name)
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
                        {"user_text": state.user_text, "agent": "ollama", "model": self._model},
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
