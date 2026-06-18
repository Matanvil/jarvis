import anthropic as _anthropic
import json
import logging
import os
import time
import httpx
from datetime import date as _date
from classifier_prompt import CLASSIFY_SYSTEM_PROMPT
from git_context import get_git_context
from guardrails import Guardrails
from ollama_agent import OllamaAgent, EscalateToCloud
from agent import Agent, claude_code_available
from prompt_loader import PromptLoader

_VALID_ROUTING_MODES = {"ollama_first", "claude_only", "ollama_only", "haiku_first", "local_first"}


class Router:
    """Routes commands via pre-flight Ollama classifier, then to the right agent.
    Returns the same response dict shape as both agents, plus routing metadata.
    """

    def __init__(self, config: dict, guardrails: Guardrails, prompt_loader: PromptLoader | None = None, mcp_manager=None):
        self._config = config
        self._http_client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0))
        self._ollama = OllamaAgent(config=config, guardrails=guardrails, mcp_manager=mcp_manager)
        haiku_model = config.get("models", {}).get("haiku", "claude-haiku-4-5-20251001")
        sonnet_model = config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
        self._haiku = Agent(config=config, guardrails=guardrails,
                            local_agent=self._ollama, model=haiku_model, mcp_manager=mcp_manager)
        self._sonnet = Agent(config=config, guardrails=guardrails,
                             local_agent=self._ollama, model=sonnet_model, mcp_manager=mcp_manager)
        self._claude = self._sonnet   # legacy alias for claude_only / ollama_first modes
        self._history: list[dict] = []
        self._pending_compaction_notice: bool = False
        self._needs_compaction: bool = False
        self._compact_failed: bool = False
        self._anthropic_client = _anthropic.Anthropic(api_key=config.get("anthropic_api_key", ""))
        # Session state for context injection
        self._prompt_loader = prompt_loader
        self._last_cwd: str | None = None
        self._last_git_context: dict | None = None
        self._cached_system_prompt: str | None = None
        self._cached_system_prompt_cwd: str | None = None

    def reset_conversation(self) -> None:
        self._history = []
        self._needs_compaction = False
        self._compact_failed = False
        self._last_cwd = None
        self._last_git_context = None
        self._pending_compaction_notice = False

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

    def _classify(self, text: str, history: list | None = None) -> dict:
        """Ask the classifier to classify intent. Returns classification dict.
        Raises on any error — caller handles gracefully."""
        history_context = (history or [])[-4:]  # last 2 turns = 4 messages
        resp = self._http_client.post(
            f"{self._classifier_host}/v1/chat/completions",
            json={
                "model": self._classifier_model,
                "messages": [
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    *history_context,
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

    def _get_system_prompt(self, cwd: str | None) -> str | None:
        """Return cached system prompt, rebuilding if cwd changed. Returns None if no PromptLoader."""
        if self._prompt_loader is None:
            return None
        if self._cached_system_prompt is None or self._cached_system_prompt_cwd != cwd:
            base = self._prompt_loader.base_prompt().format(home=os.path.expanduser("~"))
            profile = self._prompt_loader.profile()
            refs = self._prompt_loader.refs_index(cwd)
            parts = [base]
            if profile:
                parts.append(f"\n{profile}")
            if refs:
                paths_str = ", ".join(refs)
                parts.append(f"\nAvailable refs: {paths_str} — use file_read to load when relevant to the task.\n")
            self._cached_system_prompt = "\n".join(parts)
            self._cached_system_prompt_cwd = cwd
        return self._cached_system_prompt

    def _get_local_system_prompt(self, cwd: str | None) -> str | None:
        """System prompt for local models: base + local_extra. Returns None if no PromptLoader."""
        base = self._get_system_prompt(cwd)
        if base is None:
            return None
        extra = self._prompt_loader.local_extra()
        return base + (f"\n{extra}" if extra else "")

    def _build_user_prefix(
        self, text: str, cwd: str | None, memory_context: str,
        intent_class: str | None, source: str,
    ) -> str:
        """Prepend context lines to user text. Injects git context only when changed."""
        lines: list[str] = []

        # Git context: only for non-read_only tasks in a git repo, and only when changed
        if intent_class != "read_only" and cwd:
            git_ctx = get_git_context(cwd)
            changed = (
                cwd != self._last_cwd
                or git_ctx != self._last_git_context
            )
            if changed and git_ctx:
                commits_str = " · ".join(git_ctx.get("commits", []))
                remote = git_ctx.get("remote") or "unknown"
                lines.append(
                    f"[Context: branch={git_ctx['branch']} | commits: {commits_str} | remote: {remote}]"
                )
                self._last_git_context = git_ctx
            elif changed and git_ctx is None:
                self._last_git_context = None
        else:
            git_ctx = None

        # Project memory + cwd + date: inject on first call or cwd change
        if cwd != self._last_cwd:
            if memory_context:
                lines.append(f"[Project: {memory_context}]")
            if cwd:
                lines.append(f"[cwd: {cwd}]")
            lines.append(f"[Date: {_date.today()}]")
            self._last_cwd = cwd
        elif not lines:
            if source != "scheduled":
                # Nothing changed — return bare text
                return text

        if source == "scheduled":
            lines.append(
                "\n=== SCHEDULED TASK MODE ===\n"
                "- NEVER call tools to send messages.\n"
                "- For reminders: output ONLY the reminder text.\n"
                "- For action tasks: use tools and output only the result.\n"
                "=== END ==="
            )

        return "\n".join(lines) + "\n\n" + text

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
        self._append_turn(meta.get("user_text", ""), result)
        return self._annotate(result, agent=meta.get("agent", "claude"),
                              model=meta.get("model", ""), escalated=False,
                              escalation_reason=None, intent_class=meta.get("intent_class"),
                              start=start)

    def process(self, text: str, cwd: str | None = None, memory_context: str = "",
                source: str = "", step_callback=None, command_id: str | None = None) -> dict:
        """Route a command via pre-flight classifier and return response with metadata."""
        start = time.time()
        mode = self._routing_mode

        # Deferred compaction: runs at the START of the next command so it doesn't
        # block the previous command's final SSE event ("complete").
        if self._needs_compaction and not self._compact_failed:
            self._needs_compaction = False
            self._compact()

        if self._pending_compaction_notice and step_callback is not None:
            step_callback({"type": "compacted", "message": "Context compacted."})
            self._pending_compaction_notice = False

        # haiku_first: pre-flight classifies → Haiku for non-complex, Sonnet for complex_reasoning
        if mode == "haiku_first":
            classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
            try:
                classification = self._classify(text, history=self._history)
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

            user_text = self._build_user_prefix(text, cwd, memory_context, intent_class, source)
            result = agent.run(user_text=user_text, cwd=cwd, history=self._history, source=source,
                               step_callback=step_callback, command_id=command_id,
                               system_prompt=self._get_system_prompt(cwd))
            self._append_turn(text, result)
            return self._annotate(result, agent="claude", model=model_name,
                                  escalated=False, escalation_reason=classification.get("reason"),
                                  intent_class=intent_class, start=start)

        # local_first: pre-flight classifies → OllamaAgent for non-complex, Sonnet for complex_reasoning
        if mode == "local_first":
            classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
            try:
                classification = self._classify(text, history=self._history)
            except Exception as e:
                logging.getLogger("jarvis.errors").warning(
                    f"Pre-flight classifier failed: {e} — using local executor"
                )

            intent_class = classification.get("intent_class", "read_only")
            user_text = self._build_user_prefix(text, cwd, memory_context, intent_class, source)

            if intent_class == "complex_reasoning":
                result = self._sonnet.run(
                    user_text=user_text, cwd=cwd, history=self._history, source=source,
                    step_callback=step_callback, command_id=command_id,
                    system_prompt=self._get_system_prompt(cwd),
                )
                self._append_turn(text, result)
                model_name = self._config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
                return self._annotate(result, agent="claude", model=model_name,
                                      escalated=False, escalation_reason=classification.get("reason"),
                                      intent_class=intent_class, start=start)

            # Non-complex: use local OllamaAgent; escalate to Sonnet on failure
            try:
                result = self._ollama.run(
                    user_text=user_text, cwd=cwd, history=self._history,
                    step_callback=step_callback, intent_class=intent_class, command_id=command_id,
                    system_prompt=self._get_local_system_prompt(cwd),
                )
                self._append_turn(text, result)
                executor_model = self._config.get("ollama", {}).get("executor_model") or self._ollama_model
                return self._annotate(result, agent="ollama", model=executor_model,
                                      escalated=False, escalation_reason=None,
                                      intent_class=intent_class, start=start)
            except EscalateToCloud as e:
                logging.getLogger("jarvis.errors").warning(
                    f"Local executor escalated to Sonnet: {e.reason}"
                )
                result = self._sonnet.run(
                    user_text=user_text, cwd=cwd, history=self._history,
                    ollama_available=False, source=source, step_callback=step_callback, command_id=command_id,
                    system_prompt=self._get_system_prompt(cwd),
                )
                self._append_turn(text, result)
                model_name = self._config.get("models", {}).get("sonnet", "claude-sonnet-4-6")
                return self._annotate(result, agent="claude", model=model_name,
                                      escalated=True, escalation_reason=e.reason,
                                      intent_class=intent_class, start=start)

        # claude_only: skip classifier entirely
        if mode == "claude_only":
            user_text = self._build_user_prefix(text, cwd, memory_context, None, source)
            result = self._claude.run(user_text=user_text, cwd=cwd, history=self._history, source=source,
                                      step_callback=step_callback, command_id=command_id,
                                      system_prompt=self._get_system_prompt(cwd))
            self._append_turn(text, result)
            return self._annotate(result, agent="claude", model="claude-sonnet-4-6",
                                  escalated=False, escalation_reason=None,
                                  intent_class=None, start=start)

        # Pre-flight classification (ollama_first / ollama_only)
        classification = {"can_handle_locally": True, "intent_class": "read_only", "reason": "fallback"}
        try:
            classification = self._classify(text, history=self._history)
        except Exception as e:
            logging.getLogger("jarvis.errors").warning(
                f"Pre-flight classifier failed: {e} — using ollama_first fallback"
            )

        intent_class = classification.get("intent_class", "read_only")
        can_handle_locally = classification.get("can_handle_locally", True)
        user_text = self._build_user_prefix(text, cwd, memory_context, intent_class, source)

        escalation_reason = None
        if mode == "ollama_only" or can_handle_locally:
            try:
                result = self._ollama.run(
                    user_text=user_text, cwd=cwd, history=self._history,
                    step_callback=step_callback, intent_class=intent_class, command_id=command_id,
                    system_prompt=self._get_local_system_prompt(cwd),
                )
                self._append_turn(text, result)
                return self._annotate(result, agent="ollama", model=self._ollama_model,
                                      escalated=False, escalation_reason=None,
                                      intent_class=intent_class, start=start)
            except EscalateToCloud as e:
                if mode == "ollama_only":
                    msg = "I wasn't able to complete that locally. Please try rephrasing or restart the server if this keeps happening."
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
        result = self._claude.run(
            user_text=user_text, cwd=cwd, history=self._history,
            ollama_available=(escalation_reason is None), source=source,
            step_callback=step_callback, command_id=command_id,
            system_prompt=self._get_system_prompt(cwd),
        )
        self._append_turn(text, result)
        return self._annotate(result, agent="claude", model="claude-sonnet-4-6",
                              escalated=escalation_reason is not None,
                              escalation_reason=escalation_reason or classification.get("reason"),
                              intent_class=intent_class, start=start)

    def _append_turn(self, user_text: str, result: dict) -> None:
        """Append a compressed turn to history. Skip if approval_required."""
        if result.get("approval_required"):
            return
        steps = result.get("steps") or []
        if steps:
            lines = ["Actions:"]
            for s in steps:
                tool = s.get("tool", "?")
                inp = (s.get("input_summary") or "")[:200]
                res = (s.get("result_summary") or "")[:200]
                lines.append(f"- {tool}({inp}) → {res}")
            result_text = result.get("speak") or result.get("display") or ""
            lines.append(f"\nResult: {result_text}")
            assistant_content = "\n".join(lines)
        else:
            assistant_content = result.get("display") or result.get("speak") or ""

        if not assistant_content:
            return

        self._history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_content},
        ])

        if not self._compact_failed and self._estimate_tokens() > 5000:
            self._needs_compaction = True  # compact at start of next process() call

    def _estimate_tokens(self) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        return sum(len(m["content"]) for m in self._history) // 4

    def _compact(self) -> None:
        """Summarize _history with Haiku and replace it with a compact pair.
        Best-effort: logs and continues on any failure."""
        transcript_lines = []
        for m in self._history:
            role = m["role"].upper()
            transcript_lines.append(f"{role}: {m['content']}")
        transcript = "\n\n".join(transcript_lines)

        haiku_model = self._config.get("models", {}).get("haiku", "claude-haiku-4-5-20251001")
        prompt = (
            "Summarize this conversation into a compact context block.\n"
            "Include ALL of the following that are present:\n"
            "- Active task and its current state\n"
            "- Key entities with exact values: PR numbers, branch names, file paths, "
            "error messages, usernames, URLs, check run IDs\n"
            "- Actions taken and their outcomes (which tool calls produced useful results)\n"
            "- What the user wants to happen next, if clear\n\n"
            "Max 400 words. Be specific — preserve exact IDs and numeric values.\n\n"
            f"CONVERSATION:\n{transcript}"
        )

        try:
            response = self._anthropic_client.messages.create(
                model=haiku_model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = "".join(
                b.text for b in response.content if hasattr(b, "text")
            ).strip()
            if not summary:
                raise ValueError("Empty summary from Haiku")
            self._history = [
                {"role": "user", "content": "[Prior conversation compacted]"},
                {"role": "assistant", "content": summary},
            ]
            self._pending_compaction_notice = True
            logging.getLogger("jarvis.errors").info("Session history compacted.")
        except Exception as e:
            self._compact_failed = True  # circuit breaker: don't retry until reset_conversation()
            logging.getLogger("jarvis.errors").warning(f"History compaction failed (disabled until reset): {e}")

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
