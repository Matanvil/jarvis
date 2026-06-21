import hashlib
import pytest
from unittest.mock import patch, MagicMock, call
from tools.coding_agent import CodingAgentTool


CONFIG = {
    "anthropic_api_key": "sk-test",
    "models": {"haiku": "claude-haiku-4-5-20251001"},
    "local": {"host": "http://localhost:11434"},
}


def make_tool():
    return CodingAgentTool(CONFIG)


def test_init_creates_components():
    """CodingAgentTool init creates ClaudeClient and OllamaEmbedder."""
    with patch("tools.coding_agent.ClaudeClient") as MockLLM, \
         patch("tools.coding_agent.OllamaEmbedder") as MockEmbed:
        tool = make_tool()
    MockLLM.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        api_key="sk-test",
    )
    MockEmbed.assert_called_once_with(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )


def test_init_sets_force_local_when_routing_mode_is_local():
    """CodingAgentTool sets force_local=True on HybridClient when routing_mode is 'local'."""
    config = {**CONFIG, "local_model": "qwen3-coder:30b", "local": {**CONFIG["local"], "routing_mode": "local"}}
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaClient"), \
         patch("tools.coding_agent.HybridClient") as MockHybrid, \
         patch("tools.coding_agent.OllamaEmbedder"):
        tool = CodingAgentTool(config)
    assert MockHybrid.return_value.force_local is True


def test_init_does_not_set_force_local_when_routing_mode_is_automatic():
    """CodingAgentTool leaves force_local=False when routing_mode is 'automatic'."""
    config = {**CONFIG, "local_model": "qwen3-coder:30b", "local": {**CONFIG["local"], "routing_mode": "automatic"}}
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaClient"), \
         patch("tools.coding_agent.HybridClient") as MockHybrid, \
         patch("tools.coding_agent.OllamaEmbedder"):
        tool = CodingAgentTool(config)
    assert MockHybrid.return_value.force_local is not True


def test_init_uses_hybrid_client_when_local_model_set():
    """CodingAgentTool uses HybridClient(OllamaClient, ClaudeClient) when local_model is set."""
    config = {**CONFIG, "local_model": "qwen3-coder:30b"}
    with patch("tools.coding_agent.ClaudeClient") as MockClaude, \
         patch("tools.coding_agent.OllamaClient") as MockOllama, \
         patch("tools.coding_agent.HybridClient") as MockHybrid, \
         patch("tools.coding_agent.OllamaEmbedder"):
        tool = CodingAgentTool(config)
    MockOllama.assert_called_once_with(model="qwen3-coder:30b", base_url="http://localhost:11434")
    MockHybrid.assert_called_once_with(
        ollama=MockOllama.return_value,
        claude=MockClaude.return_value,
    )
    assert tool._llm is MockHybrid.return_value


def test_ensure_indexed_calls_index_repo_on_first_use(tmp_path):
    """_ensure_indexed indexes the repo on first call for a cwd."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo") as mock_index:
        mock_store_instance = MagicMock()
        mock_store_instance.count.return_value = 0  # not yet indexed
        MockStore.return_value = mock_store_instance

        tool = make_tool()
        result = tool._ensure_indexed(cwd)

    mock_index.assert_called_once()
    assert result is mock_store_instance


def test_ensure_indexed_skips_index_if_already_indexed(tmp_path):
    """_ensure_indexed skips indexing if ChromaDB already has data."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo") as mock_index:
        mock_store_instance = MagicMock()
        mock_store_instance.count.return_value = 42  # already indexed
        MockStore.return_value = mock_store_instance

        tool = make_tool()
        tool._ensure_indexed(cwd)

    mock_index.assert_not_called()


def test_ensure_indexed_reuses_store_on_second_call(tmp_path):
    """_ensure_indexed returns the same VectorStore on subsequent calls for same cwd."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"):
        mock_store_instance = MagicMock()
        mock_store_instance.count.return_value = 0
        MockStore.return_value = mock_store_instance

        tool = make_tool()
        store1 = tool._ensure_indexed(cwd)
        store2 = tool._ensure_indexed(cwd)

    assert store1 is store2
    assert MockStore.call_count == 1  # only created once


def test_ask_returns_answer(tmp_path):
    """ask() calls AgentLoop.ask and returns the answer."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.AgentLoop") as MockLoop:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_loop = MagicMock()
        mock_loop.ask.return_value = "The router classifies intent before execution."
        MockLoop.return_value = mock_loop

        tool = make_tool()
        result = tool.ask("what does the router do?", cwd)

    assert result["answer"] == "The router classifies intent before execution."
    assert result["error"] is None
    call_kwargs = mock_loop.ask.call_args
    assert call_kwargs.args[0] == "what does the router do?"
    assert "on_event" in call_kwargs.kwargs


def test_ask_returns_error_on_exception(tmp_path):
    """ask() returns error dict when AgentLoop raises."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.AgentLoop") as MockLoop:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        MockLoop.return_value.ask.side_effect = Exception("Ollama is not running")

        tool = make_tool()
        result = tool.ask("what does the router do?", cwd)

    assert result["answer"] is None
    assert "Ollama is not running" in result["error"]


def test_ask_triggers_index_when_cwd_not_indexed(tmp_path):
    """ask() triggers _ensure_indexed which calls index_repo for unknown cwd."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo") as mock_index, \
         patch("tools.coding_agent.AgentLoop") as MockLoop:
        mock_store = MagicMock()
        mock_store.count.return_value = 0  # triggers indexing
        MockStore.return_value = mock_store
        MockLoop.return_value.ask.return_value = "answer"

        tool = make_tool()
        tool.ask("question", cwd)

    mock_index.assert_called_once()


def _make_plan_mock():
    """Return a mock Plan with two FileEdit objects."""
    edit1 = MagicMock()
    edit1.file = "auth/service.py"
    edit1.description = "Extract auth logic into AuthService class"
    edit1.old_code = "def authenticate(user):\n    pass"
    edit1.new_code = "class AuthService:\n    def authenticate(self, user):\n        pass"

    edit2 = MagicMock()
    edit2.file = "server.py"
    edit2.description = "Use AuthService in endpoint"
    edit2.old_code = "result = authenticate(user)"
    edit2.new_code = "result = AuthService().authenticate(user)"

    plan = MagicMock()
    plan.task = "extract auth into a service"
    plan.edits = [edit1, edit2]
    return plan


def test_plan_returns_edits(tmp_path):
    """plan() calls Planner.plan, saves the plan, and returns formatted edits."""
    import os
    cwd = str(tmp_path)
    repo_name = os.path.basename(cwd.rstrip("/"))
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.save_plan") as mock_save_plan, \
         patch("tools.coding_agent.Planner") as MockPlanner:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_planner = MagicMock()
        mock_planner.plan.return_value = _make_plan_mock()
        MockPlanner.return_value = mock_planner
        mock_save_plan.return_value = "/tmp/fake-plan.json"

        tool = make_tool()
        result = tool.plan("extract auth into a service", cwd)

    assert result["error"] is None
    assert len(result["edits"]) == 2
    assert result["edits"][0]["file"] == "auth/service.py"
    assert result["edits"][0]["old_code"] == "def authenticate(user):\n    pass"
    assert "plan_summary" in result
    assert result["plan_path"] == "/tmp/fake-plan.json"
    mock_planner.plan.assert_called_once_with("extract auth into a service", repo=repo_name)
    mock_save_plan.assert_called_once()


def test_plan_returns_error_on_planner_error(tmp_path):
    """plan() returns error dict when PlannerError is raised."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.Planner") as MockPlanner:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        from tools.coding_agent import PlannerError
        MockPlanner.return_value.plan.side_effect = PlannerError("submit_plan never called")

        tool = make_tool()
        result = tool.plan("task", cwd)

    assert result["edits"] is None
    assert "submit_plan never called" in result["error"]


def _make_review_result_mock():
    """Return a mock ReviewResult with two issues."""
    issue1 = MagicMock()
    issue1.category = "critical"
    issue1.description = "SQL injection risk in query builder"
    issue1.file = "db/queries.py"
    issue1.recommendation = "Use parameterized queries"

    issue2 = MagicMock()
    issue2.category = "suggestion"
    issue2.description = "Consider extracting repeated logic"
    issue2.file = "utils.py"
    issue2.recommendation = "Create a helper function"

    result = MagicMock()
    result.summary = "Found 1 critical issue and 1 suggestion."
    result.issues = [issue1, issue2]
    return result


def test_review_returns_issues(tmp_path):
    """review() runs git diff, calls Reviewer.review, returns categorized issues."""
    cwd = str(tmp_path)
    fake_diff = "diff --git a/db/queries.py b/db/queries.py\n+query = f'SELECT * FROM {table}'"

    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.Reviewer") as MockReviewer, \
         patch("tools.coding_agent.subprocess.run") as mock_run:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_proc = MagicMock()
        mock_proc.stdout = fake_diff
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        mock_reviewer = MagicMock()
        mock_reviewer.review.return_value = _make_review_result_mock()
        MockReviewer.return_value = mock_reviewer

        tool = make_tool()
        result = tool.review(cwd, context="adding SQL queries")

    assert result["error"] is None
    assert result["summary"] == "Found 1 critical issue and 1 suggestion."
    assert len(result["issues"]) == 2
    assert result["issues"][0]["category"] == "critical"
    assert result["issues"][0]["file"] == "db/queries.py"
    mock_reviewer.review.assert_called_once_with(fake_diff, "adding SQL queries")


def test_review_returns_error_when_no_diff(tmp_path):
    """review() returns error dict when git diff returns nothing."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.subprocess.run") as mock_run:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_proc = MagicMock()
        mock_proc.stdout = ""
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        tool = make_tool()
        result = tool.review(cwd)

    assert result["issues"] is None
    assert "no changes" in result["error"].lower()


def test_review_returns_error_when_git_diff_fails(tmp_path):
    """review() returns error dict when git diff returns non-zero exit code."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.subprocess.run") as mock_run:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_proc = MagicMock()
        mock_proc.stdout = ""
        mock_proc.returncode = 128  # git error (not a git repo)
        mock_proc.stderr = "fatal: not a git repository"
        mock_run.return_value = mock_proc

        tool = make_tool()
        result = tool.review(cwd)

    assert result["issues"] is None
    assert result["error"] is not None


def test_review_returns_error_on_reviewer_error(tmp_path):
    """review() returns error dict when ReviewerError is raised."""
    cwd = str(tmp_path)
    with patch("tools.coding_agent.ClaudeClient"), \
         patch("tools.coding_agent.OllamaEmbedder"), \
         patch("tools.coding_agent.VectorStore") as MockStore, \
         patch("tools.coding_agent.index_repo"), \
         patch("tools.coding_agent.Reviewer") as MockReviewer, \
         patch("tools.coding_agent.subprocess.run") as mock_run:
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        MockStore.return_value = mock_store

        mock_proc = MagicMock()
        mock_proc.stdout = "diff --git a/x.py b/x.py\n+x = 1"
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        from tools.coding_agent import ReviewerError
        MockReviewer.return_value.review.side_effect = ReviewerError("submit_review never called")

        tool = make_tool()
        result = tool.review(cwd)

    assert result["issues"] is None
    assert "submit_review never called" in result["error"]
