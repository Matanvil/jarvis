import json
import os
import re
import httpx
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


class OllamaAgent:
    def __init__(self, config: dict, guardrails: Guardrails):
        self._config = config
        self._guardrails = guardrails
        self._shell = ShellTool()
        self._web = WebTool(brave_api_key=config.get("brave_api_key"))
        self._code = CodeTool()
        self._macos = MacOSTool()
        from tools.coding_agent import CodingAgentTool
        self._coding = CodingAgentTool(config)
        # Shared client reuses TCP connection to the persistent Ollama process.
        # Note: if timeout is updated via POST /config after construction, the
        # existing client will not pick up the new value — acceptable for now.
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=30.0, pool=5.0)
        )

    def close(self) -> None:
        self._http_client.close()

    def __del__(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass

    @property
    def _host(self) -> str:
        ollama = self._config.get("ollama", {})
        return ollama.get("executor_host") or ollama.get("host", "http://localhost:11434")

    @property
    def _model(self) -> str:
        ollama = self._config.get("ollama", {})
        return ollama.get("executor_model") or ollama.get("model", "mistral:latest")

    @property
    def _chat_template_kwargs(self) -> dict | None:
        return self._config.get("ollama", {}).get("executor_chat_template_kwargs")

    @property
    def _routing_mode(self) -> str:
        return self._config.get("ollama", {}).get("routing_mode", "ollama_first")

    @property
    def _timeout(self) -> float:
        return float(self._config.get("ollama", {}).get("timeout_seconds", 300))

    def run(self, user_text: str, cwd: str | None = None, memory_context: str = "",
            history: list | None = None, step_callback=None,
            intent_class: str | None = None) -> dict:
        """Run Ollama tool-use loop. Raises EscalateToCloud if Ollama can't handle it.
        Returns same dict shape as Agent.run()."""
        system_msg = _BASE_SYSTEM_PROMPT.format(home=os.path.expanduser("~")) + _OLLAMA_EXTRA
        if cwd:
            system_msg += f"\nActive project directory: {cwd}\n"
        if memory_context:
            system_msg += f"\nProject memory: {memory_context}\n"

        messages = [
            {"role": "system", "content": system_msg},
            *(history or []),
            {"role": "user", "content": user_text},
        ]
        tool_calls_made = []
        steps = []
        budgets = self._config.get("reasoning", {}).get("step_budgets", {})
        if intent_class and intent_class in budgets:
            max_steps = int(budgets[intent_class])
        else:
            max_steps = int(self._config.get("reasoning", {}).get("max_steps_ollama", 10))
        stall_detection = self._config.get("reasoning", {}).get("stall_detection", True)
        last_tool_call = None        # (tool_name, args_key) for exact stall detection
        recent_tools: list = []      # sliding window of last 5 tool names for near-duplicate detection
        _WINDOW = 5
        _NEAR_DUP_THRESHOLD = 3     # same tool ≥ 3 times in window → redundant

        try:
            for step_idx in range(max_steps):
                payload = {"model": self._model, "messages": messages, "tools": _OLLAMA_TOOLS}
                if self._chat_template_kwargs:
                    payload["chat_template_kwargs"] = self._chat_template_kwargs

                msg, finish = _stream_call(
                    self._http_client,
                    f"{self._host}/v1/chat/completions",
                    payload,
                    step_callback,
                )

                if finish == "stop" or not msg.get("tool_calls"):
                    text = _clean_ollama_text(msg.get("content") or "")
                    if not text and not tool_calls_made and finish != "stop":
                        raise EscalateToCloud("Empty response from local executor — server may have crashed")
                    result = format_response(text, tool_calls_made)
                    result["steps"] = steps
                    return result

                # Append assistant message with tool calls
                messages.append(msg)

                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    call_id = tc["id"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        messages.append({"role": "tool", "tool_call_id": call_id,
                                         "content": f"error: malformed tool arguments — {e}. Please retry with valid JSON."})
                        continue

                    # finalize: model signals it has enough info — return immediately
                    if name == "finalize":
                        answer = args.get("answer", "") if isinstance(args, dict) else ""
                        step = {"tool": "finalize", "input_summary": answer[:100],
                                "result_summary": "finalized", "milestone": len(steps) == 0}
                        steps.append(step)
                        result = format_response(answer, tool_calls_made)
                        result["steps"] = steps
                        return result

                    # Near-duplicate detection: same tool dominates recent window → salvage
                    recent_tools.append(name)
                    if len(recent_tools) > _WINDOW:
                        recent_tools.pop(0)
                    if stall_detection and recent_tools.count(name) >= _NEAR_DUP_THRESHOLD and steps:
                        messages.append({
                            "role": "user",
                            "content": f"You have called '{name}' {recent_tools.count(name)} times with similar inputs and similar results. Stop and respond with what you already know.",
                        })
                        try:
                            nr = self._http_client.post(
                                f"{self._host}/v1/chat/completions",
                                json={"model": self._model, "messages": messages, "tool_choice": "none"},
                            )
                            nr.raise_for_status()
                            nr_text = _clean_ollama_text(nr.json()["choices"][0]["message"].get("content") or "")
                            if nr_text:
                                result = format_response(nr_text, tool_calls_made)
                                result["steps"] = steps
                                return result
                        except Exception:
                            pass

                    # Exact stall detection: same tool + same args twice in a row → break
                    if stall_detection:
                        try:
                            current_call = (name, frozenset(args.items()) if isinstance(args, dict) else str(args))
                        except TypeError:
                            current_call = (name, str(args))
                        if current_call == last_tool_call:
                            messages.append({
                                "role": "user",
                                "content": "You already tried this exact action. Please try a different approach or conclude with what you know.",
                            })
                            # Force one more completion without tools to get a response
                            resp2 = self._http_client.post(
                                f"{self._host}/v1/chat/completions",
                                json={"model": self._model, "messages": messages},
                            )
                            resp2.raise_for_status()
                            text = _clean_ollama_text(resp2.json()["choices"][0]["message"].get("content") or "")
                            result = format_response(text, tool_calls_made)
                            result["steps"] = steps
                            return result
                        last_tool_call = current_call

                    step = {"tool": name, "input_summary": str(args)[:100], "result_summary": "", "milestone": len(steps) == 0}
                    steps.append(step)
                    if step_callback is not None:
                        step_callback({"type": "step", "label": _step_label(name), "tool": name, "milestone": step["milestone"]})
                    try:
                        result = execute_tool(name, args, self._shell, self._web, self._code, self._macos, self._guardrails, default_cwd=cwd, coding=self._coding)
                        step["result_summary"] = result[:120] if isinstance(result, str) else str(result)[:120]
                        tool_calls_made.append(name)
                    except ApprovalRequiredError as e:
                        step["result_summary"] = "approval_required"
                        return {
                            "speak": None, "display": None,
                            "approval_required": {
                                "tool": e.tool_name,
                                "description": e.description,
                                "tool_use_id": call_id,
                                "category": e.category,
                            },
                            "steps": steps,
                        }

                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        except httpx.ConnectError as e:
            raise EscalateToCloud(f"Ollama unavailable: {e}")
        except httpx.TimeoutException as e:
            raise EscalateToCloud(f"Ollama timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise EscalateToCloud(f"Ollama HTTP error: {e}")

        # Salvage: one final no-tools call to emit whatever the model gathered
        try:
            salvage_resp = self._http_client.post(
                f"{self._host}/v1/chat/completions",
                json={"model": self._model, "messages": messages, "tool_choice": "none"},
            )
            salvage_resp.raise_for_status()
            salvage_text = _clean_ollama_text(
                salvage_resp.json()["choices"][0]["message"].get("content") or ""
            )
            if salvage_text:
                result = format_response(salvage_text, tool_calls_made)
                result["steps"] = steps
                return result
        except Exception:
            pass

        result = format_response("I ran out of steps. Please try again.", tool_calls_made)
        result["steps"] = steps
        return result
