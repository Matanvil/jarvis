# jarvis-core/tests/test_prompt_loader.py
import hashlib
import pytest
from pathlib import Path
from prompt_loader import PromptLoader


@pytest.fixture
def tmp_dirs(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    refs = tmp_path / "refs"
    refs.mkdir()
    projects = tmp_path / "projects"
    projects.mkdir()
    return prompts, refs, projects


def test_base_prompt_loads_from_file(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    (prompts / "base.md").write_text("base content {home}")
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    assert "base content" in loader.base_prompt()


def test_base_prompt_falls_back_to_empty_string_if_missing(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    assert loader.base_prompt() == ""


def test_local_extra_loads_from_file(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    (prompts / "local.md").write_text("local rules")
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    assert loader.local_extra() == "local rules"


def test_profile_loads_when_present(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    (refs / "profile.md").write_text("my profile")
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    assert loader.profile() == "my profile"


def test_profile_returns_empty_string_when_missing(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    assert loader.profile() == ""


def test_refs_index_returns_global_refs_excluding_profile(tmp_dirs):
    prompts, refs, projects = tmp_dirs
    (refs / "profile.md").write_text("profile")
    (refs / "code.md").write_text("code guide")
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    result = loader.refs_index(cwd=None)
    assert any("code.md" in p for p in result)
    assert not any("profile.md" in p for p in result)


def test_refs_index_includes_per_project_refs(tmp_dirs, tmp_path):
    prompts, refs, projects = tmp_dirs
    cwd = str(tmp_path / "myproject")
    key = hashlib.md5(cwd.encode()).hexdigest()
    project_refs = projects / key / "refs"
    project_refs.mkdir(parents=True)
    (project_refs / "git.md").write_text("git notes")
    loader = PromptLoader(prompts_dir=prompts, refs_dir=refs, projects_dir=projects)
    result = loader.refs_index(cwd=cwd)
    assert any("git.md" in p for p in result)


def test_refs_index_returns_empty_list_when_no_dirs_exist(tmp_path):
    loader = PromptLoader(
        prompts_dir=tmp_path / "prompts",
        refs_dir=tmp_path / "refs",
        projects_dir=tmp_path / "projects",
    )
    assert loader.refs_index(cwd=None) == []
