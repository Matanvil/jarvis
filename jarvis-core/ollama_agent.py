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
from agent import TOOL_DEFINITIONS, _BASE_SYSTEM_PROMPT
from tools._errors import ApprovalRequiredError

# Appended to the base system prompt for local models that need extra guidance
_OLLAMA_EXTRA = """
CRITICAL RULES FOR THIS MODEL:
- Respond ONLY in English. Never use Thai, Chinese, Arabic, or any non-English language.
- ALWAYS use the provided tool/function calls to take action. NEVER write tool calls as JSON text in your response.
- If you need to run a command or write a file, call the tool — do not describe it in text.
- NEVER claim to have performed an action without first calling the tool. No tool call = no action taken.
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
_OLLAMA_TOOLS = _anthropic_to_ollama_tools(_OLLAMA_SAFE_TOOL_DEFINITIONS)


class OllamaAgent:
    def __init__(self, config: dict, guardrails: Guardrails):
        self._config = config
        self._guardrails = guardrails
        self._shell = ShellTool()
        self._web = WebTool(brave_api_key=config.get("brave_api_key"))
        self._code = CodeTool()
        self._macos = MacOSTool()
        # Shared client reuses TCP connection to the persistent Ollama process.
        # Note: if timeout is updated via POST /config after construction, the
        # existing client will not pick up the new value — acceptable for now.
        self._http_client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        self._http_client.close()

    def __del__(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass

    @property
    def _host(self) -> str:
        return self._config.get("ollama", {}).get("host", "http://localhost:11434")

    @property
    def _model(self) -> str:
        return self._config.get("ollama", {}).get("model", "mistral:latest")

    @property
    def _routing_mode(self) -> str:
        return self._config.get("ollama", {}).get("routing_mode", "ollama_first")

    @property
    def _timeout(self) -> float:
        return float(self._config.get("ollama", {}).get("timeout_seconds", 30))

    def run(self, user_text: str, cwd: str | None = None, memory_context: str = "",
            history: list | None = None) -> dict:
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

        try:
            for _ in range(10):
                resp = self._http_client.post(
                    f"{self._host}/v1/chat/completions",
                    json={"model": self._model, "messages": messages, "tools": _OLLAMA_TOOLS},
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                finish = choice["finish_reason"]

                if finish == "stop" or not msg.get("tool_calls"):
                    text = _clean_ollama_text(msg.get("content") or "")
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

                    step = {"tool": name, "input_summary": str(args)[:100], "result_summary": ""}
                    steps.append(step)
                    try:
                        result = execute_tool(name, args, self._shell, self._web, self._code, self._macos, self._guardrails, default_cwd=cwd)
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

        result = format_response("I ran out of steps. Please try again.", tool_calls_made)
        result["steps"] = steps
        return result
