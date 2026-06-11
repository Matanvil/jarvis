import json
import logging
import time
import httpx
from guardrails import Guardrails
from ollama_agent import OllamaAgent, EscalateToCloud
from agent import Agent, claude_code_available

_VALID_ROUTING_MODES = {"ollama_first", "claude_only", "ollama_only", "haiku_first", "local_first"}

_CLASSIFY_SYSTEM_PROMPT = """You are a command classifier. Analyze the user request and return ONLY a JSON object — no other text.

Return exactly this structure:
{
  "can_handle_locally": true or false,
  "intent_class": "read_only" or "prepare" or "destructive" or "complex_reasoning",
  "reason": "one sentence explanation"
}

Rules for can_handle_locally:
- true: task needs only file ops, shell commands, app control, code execution, OS queries
- false: task needs web search, current news/prices, advanced code generation, deep reasoning

Rules for intent_class:
- read_only: just reading or querying, no changes (list files, read a file, show git log, check dependencies, explain code, search a codebase)
- prepare: will make changes the user should preview (generate code, draft text, move files)
- destructive: deletes files, sends messages, modifies system settings, irreversible actions
- complex_reasoning: needs web search, real-time data, or external information NOT available on this machine (news, prices, current events, remote APIs, live documentation). Git history, local files, installed packages, running processes, and codebase questions are NOT complex_reasoning — they are read_only."""


class Router:
    """Routes commands via pre-flight Ollama classifier, then to the right agent.
    Returns the same response dict shape as both agents, plus routing metadata.
    """

    _MAX_HISTORY = 10  # 5 turns (user + assistant per turn)

    def __init__(self, config: dict, guardrails: Guardrails):
        self._config = config
        self._ollama = OllamaAgent(config=config, guardrails=guardrails)
        haiku_model = config.get("models", {}).get("haiku", "claude-haiku-4-5-20251001")
        sonnet_model = config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
        self._haiku = Agent(config=config, guardrails=guardrails,
                            local_agent=self._ollama, model=haiku_model)
        self._sonnet = Agent(config=config, guardrails=guardrails,
                             local_agent=self._ollama, model=sonnet_model)
        self._claude = self._sonnet   # legacy alias for claude_only / ollama_first modes
        self._history: list[dict] = []

    def reset_conversation(self) -> None:
        self._history = []

        mode = self._config.get("ollama", {}).get("routing_mode", "ollama_first")
        if mode not in _VALID_ROUTING_MODES:
            logging.getLogger("jarvis.errors").warning(
                f"Invalid routing_mode '{mode}'. Must be one of {sorted(_VALID_ROUTING_MODES)}. "
                f"Falling back to 'ollama_first'."
            )

        if not claude_code_available():
            logging.getLogger("jarvis.errors").warning(
                "Claude Code CLI not found. Complex codebase delegation will be unavailable. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

    @property
    def _routing_mode(self) -> str:
        return self._config.get("ollama", {}).get("routing_mode", "ollama_first")

    @property
    def _ollama_model(self) -> str:
        return self._config.get("ollama", {}).get("model", "mistral:latest")

    @property
    def _classifier_model(self) -> str:
        import config as cfg
        return cfg.classifier_backend(self._config)[1]

    @property
    def _classifier_host(self) -> str:
        import config as cfg
        return cfg.classifier_backend(self._config)[0]

    @property
    def _ollama_host(self) -> str:
        return self._config.get("ollama", {}).get("host", "http://localhost:11434")

    @property
    def _ollama_timeout(self) -> float:
        return float(self._config.get("ollama", {}).get("timeout_seconds", 30))

    def _classify(self, text: str) -> dict:
        """Ask Ollama to classify intent. Returns classification dict.
        Raises on any error — caller handles gracefully."""
        with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=self._ollama_timeout, write=30.0, pool=5.0)) as client:
            resp = client.post(
                f"{self._classifier_host}/v1/chat/completions",
                json={
                    "model": self._classifier_model,
                    "messages": [
                        {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            # Extract JSON robustly — model may wrap output in tags or non-Latin text
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in classifier response: {content!r}")
            return json.loads(content[start:end + 1])

    def resume(self, command_id: str, step_callback=None) -> dict | None:
        """Continue a run paused for approval. Returns the annotated final result,
        or None if there is no paused run for this command_id (caller falls back to
        replaying the original command)."""
        import approval_store
        entry = approval_store.pop(command_id)
        if entry is None:
            return None
        start = time.time()
        result = entry["resume"](step_callback)
        meta = entry["meta"]
        self._update_history(meta.get("user_text", ""), result)
        return self._annotate(result, agent=meta.get("agent", "claude"),
                              model=meta.get("model", ""), escalated=False,
                              escalation_reason=None, intent_class=meta.get("intent_class"),
                              start=start)

    def process(self, text: str, cwd: str | None = None, memory_context: str = "",
                source: str = "", step_callback=None, command_id: str | None = None) -> dict:
        """Route a command via pre-flight classifier and return response with metadata."""
        start = time.time()
        mode = self._routing_mode

        # haiku_first: pre-flight classifies → Haiku for non-complex, Sonnet for complex_reasoning
        if mode == "haiku_first":
            classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
            try:
                classification = self._classify(text)
            except Exception as e:
                logging.getLogger("jarvis.errors").warning(
                    f"Pre-flight classifier failed: {e} — using haiku"
                )

            intent_class = classification.get("intent_class", "read_only")
            use_sonnet = intent_class == "complex_reasoning"
            agent = self._sonnet if use_sonnet else self._haiku
            model_name = (self._config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
                          if use_sonnet else
                          self._config.get("models", {}).get("haiku", "claude-haiku-4-5-20251001"))

            result = agent.run(text, cwd=cwd, memory_context=memory_context, history=self._history, source=source,
                               step_callback=step_callback, command_id=command_id)
            self._update_history(text, result)
            return self._annotate(result, agent="claude", model=model_name,
                                  escalated=False, escalation_reason=classification.get("reason"),
                                  intent_class=intent_class, start=start)

        # local_first: pre-flight classifies → OllamaAgent for non-complex, Sonnet for complex_reasoning
        if mode == "local_first":
            classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
            try:
                classification = self._classify(text)
            except Exception as e:
                logging.getLogger("jarvis.errors").warning(
                    f"Pre-flight classifier failed: {e} — using local executor"
                )

            intent_class = classification.get("intent_class", "read_only")
            if intent_class == "complex_reasoning":
                result = self._sonnet.run(text, cwd=cwd, memory_context=memory_context, history=self._history,
                                          source=source, step_callback=step_callback, command_id=command_id)
                self._update_history(text, result)
                model_name = self._config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
                return self._annotate(result, agent="claude", model=model_name,
                                      escalated=False, escalation_reason=classification.get("reason"),
                                      intent_class=intent_class, start=start)

            # Non-complex: use local OllamaAgent; escalate to Sonnet on failure
            try:
                result = self._ollama.run(text, cwd=cwd, memory_context=memory_context, history=self._history, step_callback=step_callback, intent_class=intent_class, command_id=command_id)
                self._update_history(text, result)
                executor_model = self._config.get("ollama", {}).get("executor_model") or self._ollama_model
                return self._annotate(result, agent="ollama", model=executor_model,
                                      escalated=False, escalation_reason=None,
                                      intent_class=intent_class, start=start)
            except EscalateToCloud as e:
                logging.getLogger("jarvis.errors").warning(
                    f"Local executor escalated to Sonnet: {e.reason}"
                )
                result = self._sonnet.run(text, cwd=cwd, memory_context=memory_context, history=self._history,
                                          ollama_available=False, source=source, step_callback=step_callback, command_id=command_id)
                self._update_history(text, result)
                model_name = self._config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
                return self._annotate(result, agent="claude", model=model_name,
                                      escalated=True, escalation_reason=e.reason,
                                      intent_class=intent_class, start=start)

        # claude_only: skip classifier entirely
        if mode == "claude_only":
            result = self._claude.run(text, cwd=cwd, memory_context=memory_context, history=self._history, source=source,
                                      step_callback=step_callback, command_id=command_id)
            self._update_history(text, result)
            return self._annotate(result, agent="claude", model="claude-sonnet-4-6",
                                  escalated=False, escalation_reason=None,
                                  intent_class=None, start=start)

        # Pre-flight classification (ollama_first / ollama_only)
        classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
        try:
            classification = self._classify(text)
        except Exception as e:
            logging.getLogger("jarvis.errors").warning(
                f"Pre-flight classifier failed: {e} — using ollama_first fallback"
            )

        intent_class = classification.get("intent_class", "read_only")
        can_handle_locally = classification.get("can_handle_locally", True)

        escalation_reason = None
        if mode == "ollama_only" or can_handle_locally:
            try:
                result = self._ollama.run(text, cwd=cwd, memory_context=memory_context, history=self._history, step_callback=step_callback, intent_class=intent_class, command_id=command_id)
                self._update_history(text, result)
                return self._annotate(result, agent="ollama", model=self._ollama_model,
                                      escalated=False, escalation_reason=None,
                                      intent_class=intent_class, start=start)
            except EscalateToCloud as e:
                if mode == "ollama_only":
                    msg = f"Local model unavailable: {e.reason}"
                    result = {"speak": msg, "display": msg, "steps": []}
                    return self._annotate(result, agent="ollama", model=self._ollama_model,
                                          escalated=True, escalation_reason=f"suppressed:ollama_only:{e.reason}",
                                          intent_class=intent_class, start=start)
                # ollama_first: Ollama unavailable or escalated — fall through to Claude
                escalation_reason = e.reason
                logging.getLogger("jarvis.errors").warning(
                    f"Ollama escalated to Claude: {e.reason}"
                )

        # can_handle_locally=False OR Ollama escalated: route to Claude
        # Pass ollama_available=False when escalated so Claude doesn't waste a step on delegate_to_local
        result = self._claude.run(text, cwd=cwd, memory_context=memory_context, history=self._history,
                                  ollama_available=(escalation_reason is None), source=source,
                                  step_callback=step_callback, command_id=command_id)
        self._update_history(text, result)
        return self._annotate(result, agent="claude", model="claude-sonnet-4-6",
                              escalated=escalation_reason is not None,
                              escalation_reason=escalation_reason or classification.get("reason"),
                              intent_class=intent_class, start=start)

    def _update_history(self, user_text: str, result: dict) -> None:
        """Append the exchange to history. Skip if approval_required (command not completed)."""
        if result.get("approval_required"):
            return
        assistant_text = result.get("display") or result.get("speak") or ""
        if not assistant_text:
            return
        steps = result.get("steps") or []
        if steps:
            tools_used = ", ".join(s["tool"] for s in steps)
            assistant_text = f"{assistant_text}\n\n[Tools used: {tools_used}]"
        self._history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def _annotate(self, result: dict, agent: str, model: str,
                  escalated: bool, escalation_reason: str | None,
                  intent_class: str | None, start: float) -> dict:
        response_ms = int((time.time() - start) * 1000)
        return {
            **result,
            "_agent": agent,
            "_model": model,
            "_escalated": escalated,
            "_escalation_reason": escalation_reason,
            "_intent_class": intent_class,
            "_response_ms": response_ms,
        }
