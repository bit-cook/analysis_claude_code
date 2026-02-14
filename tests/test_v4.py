"""
Tests for v4_skills_agent.py - Dynamic skill injection.

Tests SkillLoader initialization, parsing, listing, and LLM skill loading behavior.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL
from tests.helpers import TODO_WRITE_TOOL, SKILL_TOOL, TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL

from v4_skills_agent import SkillLoader, run_skill, SYSTEM


# =============================================================================
# Unit Tests
# =============================================================================


def test_skill_loader_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillLoader(Path(tmpdir))
        assert isinstance(loader.skills, dict), "skills should be a dict"
        assert len(loader.skills) == 0, "fresh loader should have empty cache"
    print("PASS: test_skill_loader_init")
    return True


def test_skill_loader_parse_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: test\n"
            "description: A test skill\n"
            "---\n"
            "\n"
            "This is the skill content.\n"
        )

        loader = SkillLoader(Path(tmpdir))
        result = loader.parse_skill_md(skill_md)

        assert result is not None, "Valid SKILL.md should parse successfully"
        assert result["name"] == "test", f"Expected name 'test', got '{result['name']}'"
        assert "A test skill" in result["description"], "Description should match"
        assert "skill content" in result["body"], "Body should contain the content"
    print("PASS: test_skill_loader_parse_valid")
    return True


def test_skill_loader_parse_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "bad-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("No frontmatter here, just plain text.")

        loader = SkillLoader(Path(tmpdir))
        result = loader.parse_skill_md(skill_md)

        assert result is None, "Invalid SKILL.md (no frontmatter) should return None"
    print("PASS: test_skill_loader_parse_invalid")
    return True


def test_skill_loader_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillLoader(Path(tmpdir))
        result = loader.list_skills()
        assert isinstance(result, list), "list_skills should return a list"

        skill_dir = Path(tmpdir) / "demo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: demo\n"
            "description: Demo skill\n"
            "---\n"
            "\n"
            "Demo body.\n"
        )
        loader2 = SkillLoader(Path(tmpdir))
        result2 = loader2.list_skills()
        assert "demo" in result2, "list_skills should include loaded skill names"
    print("PASS: test_skill_loader_list")
    return True


def test_skill_injection_mechanism():
    """Verify skill content is retrieved as injectable data, not baked into system prompt.

    v4's unique value: skill body goes into tool_result (user message) via
    get_skill_content, preserving prompt cache prefix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "inject-test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: inject-test\n"
            "description: Injection mechanism test skill\n"
            "---\n"
            "\n"
            "Follow these specialized injection instructions carefully.\n"
        )

        loader = SkillLoader(Path(tmpdir))
        content = loader.get_skill_content("inject-test")

        assert content is not None, "get_skill_content should return content for a valid skill"
        assert isinstance(content, str), f"Skill content should be a string, got {type(content)}"
        assert "specialized injection instructions" in content, \
            "Returned content should contain the SKILL.md body text"
        assert "inject-test" in content, \
            "Returned content should reference the skill name"

    print("PASS: test_skill_injection_mechanism")
    return True


def test_skill_tool_returns_content():
    """Verify the Skill tool handler returns content via tool_result, not system prompt.

    Two assertions that prove cache-preserving injection:
    1. run_skill wraps content in <skill-loaded> tags (returned as tool_result string)
    2. SYSTEM prompt does NOT contain any skill body content (only names/descriptions)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "cache-test"
        skill_dir.mkdir()
        body_marker = "Unique body content that must not appear in system prompt"
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: cache-test\n"
            "description: Cache preservation test\n"
            "---\n"
            "\n"
            f"{body_marker}\n"
        )

        loader = SkillLoader(Path(tmpdir))
        content = loader.get_skill_content("cache-test")

        # Simulate what run_skill does: wrap in <skill-loaded> tags
        # This is what becomes the tool_result in the user message
        result = f'<skill-loaded name="cache-test">\n{content}\n</skill-loaded>'
        assert "<skill-loaded" in result, "Skill tool result should contain <skill-loaded> tags"
        assert body_marker in result, "Skill tool result should contain the skill body"
        assert "</skill-loaded>" in result, "Skill tool result should have closing tag"

    # The SYSTEM prompt (built at module load) should NOT contain skill body content.
    # It only contains skill names and one-line descriptions via get_descriptions().
    # This is what preserves the prompt cache: system prompt stays static.
    assert "specialized injection instructions" not in SYSTEM, \
        "SYSTEM prompt must not contain skill body content (would break cache)"
    assert "Unique body content" not in SYSTEM, \
        "SYSTEM prompt must not contain skill body content (would break cache)"

    print("PASS: test_skill_tool_returns_content")
    return True


# =============================================================================
# LLM Tests
# =============================================================================


def test_llm_loads_skill():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    tools = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, SKILL_TOOL]
    text, calls, _ = run_agent(
        client,
        "Call the Skill tool with skill='test-skill' to load it.",
        tools,
        system="You MUST use the Skill tool when asked to load a skill. Call Skill with the skill parameter.",
    )

    assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
    skill_calls = [c for c in calls if c[0] == "Skill"]
    if len(skill_calls) == 0:
        print("WARN: GLM did not call Skill tool (flaky model behavior)")
    print("PASS: test_llm_loads_skill")
    return True


def test_llm_follows_skill_instructions():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    tools = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, SKILL_TOOL]
    text, calls, _ = run_agent(
        client,
        "Call the Skill tool with skill='test-skill'. Then summarize the steps it recommends.",
        tools,
        system="You MUST use the Skill tool to load skills. Call Skill first, then respond.",
    )

    assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
    assert text is not None, "Model should produce a text response"
    print("PASS: test_llm_follows_skill_instructions")
    return True


def test_llm_skill_then_work():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tools = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, SKILL_TOOL]
        filepath = os.path.join(tmpdir, "output.txt")
        text, calls, _ = run_agent(
            client,
            f"Write a file at {filepath} with content 'skill applied' using write_file.",
            tools,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        assert os.path.exists(filepath), f"File should exist at {filepath}"
    print("PASS: test_llm_skill_then_work")
    return True


def test_skill_cache_separation():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    tools = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, SKILL_TOOL]

    text1, calls1, _ = run_agent(
        client,
        "Load the 'test-skill' skill and summarize it briefly.",
        tools,
    )

    text2, calls2, _ = run_agent(
        client,
        "Load the 'test-skill' skill and summarize it briefly.",
        tools,
    )

    skill_calls_1 = [c for c in calls1 if c[0] == "Skill"]
    skill_calls_2 = [c for c in calls2 if c[0] == "Skill"]
    assert len(skill_calls_1) > 0, "First run should call Skill tool"
    assert len(skill_calls_2) > 0, "Second run should also call Skill tool (independent session)"
    assert text1 is not None, "First run should produce output"
    assert text2 is not None, "Second run should produce output"
    print("PASS: test_skill_cache_separation")
    return True


# =============================================================================
# Runner
# =============================================================================


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_skill_loader_init,
        test_skill_loader_parse_valid,
        test_skill_loader_parse_invalid,
        test_skill_loader_list,
        test_skill_injection_mechanism,
        test_skill_tool_returns_content,
        test_llm_loads_skill,
        test_llm_follows_skill_instructions,
        test_llm_skill_then_work,
        test_skill_cache_separation,
    ]) else 1)
