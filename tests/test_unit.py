"""
Unit tests for learn-claude-code agents.

These tests don't require API calls - they verify code structure and logic.
"""
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Import Tests
# =============================================================================

def test_imports():
    """Test that all agent modules can be imported."""
    agents = [
        "v0_bash_agent",
        "v0_bash_agent_mini",
        "v1_basic_agent",
        "v2_todo_agent",
        "v3_subagent",
        "v4_skills_agent",
        "v5_compression_agent",
        "v6_tasks_agent",
        "v7_background_agent",
        "v8_teammate_agent",
    ]

    for agent in agents:
        spec = importlib.util.find_spec(agent)
        assert spec is not None, f"Failed to find {agent}"
        print(f"  Found: {agent}")

    print("PASS: test_imports")
    return True


# =============================================================================
# TodoManager Tests
# =============================================================================

def test_todo_manager_basic():
    """Test TodoManager basic operations."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()

    # Test valid update
    result = tm.update([
        {"content": "Task 1", "status": "pending", "activeForm": "Doing task 1"},
        {"content": "Task 2", "status": "in_progress", "activeForm": "Doing task 2"},
    ])

    assert "Task 1" in result
    assert "Task 2" in result
    assert len(tm.items) == 2

    print("PASS: test_todo_manager_basic")
    return True


def test_todo_manager_constraints():
    """Test TodoManager enforces constraints."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()

    # Test: only one in_progress allowed (should raise or return error)
    try:
        result = tm.update([
            {"content": "Task 1", "status": "in_progress", "activeForm": "Doing 1"},
            {"content": "Task 2", "status": "in_progress", "activeForm": "Doing 2"},
        ])
        # If no exception, check result contains error
        assert "Error" in result or "error" in result.lower()
    except ValueError as e:
        # Exception is expected - constraint enforced
        assert "in_progress" in str(e).lower()

    # Test: max 20 items
    tm2 = TodoManager()
    many_items = [{"content": f"Task {i}", "status": "pending", "activeForm": f"Doing {i}"} for i in range(25)]
    try:
        tm2.update(many_items)
    except ValueError:
        pass  # Exception is fine
    assert len(tm2.items) <= 20

    print("PASS: test_todo_manager_constraints")
    return True


# =============================================================================
# Reminder Tests
# =============================================================================

def test_reminder_constants():
    """Test reminder constants are defined correctly."""
    from v2_todo_agent import INITIAL_REMINDER, NAG_REMINDER

    assert "<reminder>" in INITIAL_REMINDER
    assert "</reminder>" in INITIAL_REMINDER
    assert "<reminder>" in NAG_REMINDER
    assert "</reminder>" in NAG_REMINDER
    assert "todo" in NAG_REMINDER.lower() or "Todo" in NAG_REMINDER

    print("PASS: test_reminder_constants")
    return True


def test_nag_reminder_in_agent_loop():
    """Test NAG_REMINDER injection is inside agent_loop."""
    import inspect
    from v2_todo_agent import agent_loop, NAG_REMINDER

    source = inspect.getsource(agent_loop)

    # NAG_REMINDER should be referenced in agent_loop
    assert "NAG_REMINDER" in source, "NAG_REMINDER should be in agent_loop"
    assert "rounds_without_todo" in source, "rounds_without_todo check should be in agent_loop"
    assert "results.insert" in source or "results.append" in source, "Should inject into results"

    print("PASS: test_nag_reminder_in_agent_loop")
    return True


# =============================================================================
# Configuration Tests
# =============================================================================

def test_env_config():
    """Test environment variable configuration.

    v1_basic_agent uses load_dotenv(override=True) which re-reads .env on reload.
    We verify MODEL is set from MODEL_ID (either from .env or os.environ).
    """
    import importlib
    import v1_basic_agent
    importlib.reload(v1_basic_agent)

    model_id = os.environ.get("MODEL_ID", "")
    if model_id:
        assert v1_basic_agent.MODEL == model_id, \
            f"MODEL should match MODEL_ID env var '{model_id}', got {v1_basic_agent.MODEL}"
    else:
        assert "claude" in v1_basic_agent.MODEL.lower(), \
            f"Default MODEL should contain 'claude': {v1_basic_agent.MODEL}"

    print("PASS: test_env_config")
    return True


def test_default_model():
    """Test MODEL_ID is read correctly from environment.

    When .env contains MODEL_ID, load_dotenv(override=True) will always set it.
    We verify the module reads whatever MODEL_ID is in the environment.
    """
    import importlib
    import v1_basic_agent
    importlib.reload(v1_basic_agent)

    assert v1_basic_agent.MODEL is not None, "MODEL should not be None"
    assert len(v1_basic_agent.MODEL) > 0, "MODEL should not be empty"

    print("PASS: test_default_model")
    return True


# =============================================================================
# Tool Schema Tests
# =============================================================================

def test_tool_schemas():
    """Test tool schemas are valid."""
    from v1_basic_agent import TOOLS

    required_tools = {"bash", "read_file", "write_file", "edit_file"}
    tool_names = {t["name"] for t in TOOLS}

    assert required_tools.issubset(tool_names), f"Missing tools: {required_tools - tool_names}"

    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"].get("type") == "object"

    print("PASS: test_tool_schemas")
    return True


# =============================================================================
# TodoManager Edge Case Tests
# =============================================================================

def test_todo_manager_empty_list():
    """Test TodoManager handles empty list."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()
    result = tm.update([])

    assert "No todos" in result or len(tm.items) == 0
    print("PASS: test_todo_manager_empty_list")
    return True


def test_todo_manager_status_transitions():
    """Test TodoManager status transitions."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()

    # Start with pending
    tm.update([{"content": "Task", "status": "pending", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "pending"

    # Move to in_progress
    tm.update([{"content": "Task", "status": "in_progress", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "in_progress"

    # Complete
    tm.update([{"content": "Task", "status": "completed", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "completed"

    print("PASS: test_todo_manager_status_transitions")
    return True


def test_todo_manager_missing_fields():
    """Test TodoManager rejects items with missing fields."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()

    # Missing content
    try:
        tm.update([{"status": "pending", "activeForm": "Doing"}])
        assert False, "Should reject missing content"
    except ValueError:
        pass

    # Missing activeForm
    try:
        tm.update([{"content": "Task", "status": "pending"}])
        assert False, "Should reject missing activeForm"
    except ValueError:
        pass

    print("PASS: test_todo_manager_missing_fields")
    return True


def test_todo_manager_invalid_status():
    """Test TodoManager rejects invalid status values."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()

    try:
        tm.update([{"content": "Task", "status": "invalid", "activeForm": "Doing"}])
        assert False, "Should reject invalid status"
    except ValueError as e:
        assert "status" in str(e).lower()

    print("PASS: test_todo_manager_invalid_status")
    return True


def test_todo_manager_render_format():
    """Test TodoManager render format."""
    from v2_todo_agent import TodoManager

    tm = TodoManager()
    tm.update([
        {"content": "Task A", "status": "completed", "activeForm": "A"},
        {"content": "Task B", "status": "in_progress", "activeForm": "B"},
        {"content": "Task C", "status": "pending", "activeForm": "C"},
    ])

    result = tm.render()
    assert "[x] Task A" in result
    assert "[>] Task B" in result
    assert "[ ] Task C" in result
    assert "1/3" in result  # Format may vary: "done" or "completed"

    print("PASS: test_todo_manager_render_format")
    return True


# =============================================================================
# v3 Agent Type Registry Tests
# =============================================================================

def test_v3_agent_types_structure():
    """Test v3 AGENT_TYPES structure."""
    from v3_subagent import AGENT_TYPES

    required_types = {"explore", "code", "plan"}
    assert set(AGENT_TYPES.keys()) == required_types

    for name, config in AGENT_TYPES.items():
        assert "description" in config, f"{name} missing description"
        assert "tools" in config, f"{name} missing tools"
        assert "prompt" in config, f"{name} missing prompt"

    print("PASS: test_v3_agent_types_structure")
    return True


def test_v3_get_tools_for_agent():
    """Test v3 get_tools_for_agent filters correctly."""
    from v3_subagent import get_tools_for_agent, BASE_TOOLS

    # explore: read-only
    explore_tools = get_tools_for_agent("explore")
    explore_names = {t["name"] for t in explore_tools}
    assert "bash" in explore_names
    assert "read_file" in explore_names
    assert "write_file" not in explore_names
    assert "edit_file" not in explore_names

    # code: all base tools
    code_tools = get_tools_for_agent("code")
    assert len(code_tools) == len(BASE_TOOLS)

    # plan: read-only
    plan_tools = get_tools_for_agent("plan")
    plan_names = {t["name"] for t in plan_tools}
    assert "write_file" not in plan_names

    print("PASS: test_v3_get_tools_for_agent")
    return True


def test_v3_get_agent_descriptions():
    """Test v3 get_agent_descriptions output."""
    from v3_subagent import get_agent_descriptions

    desc = get_agent_descriptions()
    assert "explore" in desc
    assert "code" in desc
    assert "plan" in desc
    assert "Read-only" in desc or "read" in desc.lower()

    print("PASS: test_v3_get_agent_descriptions")
    return True


def test_v3_task_tool_schema():
    """Test v3 Task tool schema."""
    from v3_subagent import TASK_TOOL, AGENT_TYPES

    assert TASK_TOOL["name"] == "Task"
    schema = TASK_TOOL["input_schema"]
    assert "description" in schema["properties"]
    assert "prompt" in schema["properties"]
    assert "agent_type" in schema["properties"]
    assert set(schema["properties"]["agent_type"]["enum"]) == set(AGENT_TYPES.keys())

    print("PASS: test_v3_task_tool_schema")
    return True


# =============================================================================
# v4 SkillLoader Tests
# =============================================================================

def test_v4_skill_loader_init():
    """Test v4 SkillLoader initialization."""
    from v4_skills_agent import SkillLoader
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty skills dir
        loader = SkillLoader(Path(tmpdir))
        assert len(loader.skills) == 0

    print("PASS: test_v4_skill_loader_init")
    return True


def test_v4_skill_loader_parse_valid():
    """Test v4 SkillLoader parses valid SKILL.md."""
    from v4_skills_agent import SkillLoader
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test
description: A test skill for testing
---

# Test Skill

This is the body content.
""")

        loader = SkillLoader(Path(tmpdir))
        assert "test" in loader.skills
        assert loader.skills["test"]["description"] == "A test skill for testing"
        assert "body content" in loader.skills["test"]["body"]

    print("PASS: test_v4_skill_loader_parse_valid")
    return True


def test_v4_skill_loader_parse_invalid():
    """Test v4 SkillLoader rejects invalid SKILL.md."""
    from v4_skills_agent import SkillLoader
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "bad-skill"
        skill_dir.mkdir()

        # Missing frontmatter
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# No frontmatter\n\nJust content.")

        loader = SkillLoader(Path(tmpdir))
        assert "bad-skill" not in loader.skills

    print("PASS: test_v4_skill_loader_parse_invalid")
    return True


def test_v4_skill_loader_get_content():
    """Test v4 SkillLoader get_skill_content."""
    from v4_skills_agent import SkillLoader
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "demo"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: demo
description: Demo skill
---

# Demo Instructions

Step 1: Do this
Step 2: Do that
""")

        # Add resources
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.sh").write_text("#!/bin/bash\necho hello")

        loader = SkillLoader(Path(tmpdir))

        content = loader.get_skill_content("demo")
        assert content is not None
        assert "Demo Instructions" in content
        assert "helper.sh" in content  # Resources listed

        # Non-existent skill
        assert loader.get_skill_content("nonexistent") is None

    print("PASS: test_v4_skill_loader_get_content")
    return True


def test_v4_skill_loader_list_skills():
    """Test v4 SkillLoader list_skills."""
    from v4_skills_agent import SkillLoader
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two skills
        for name in ["alpha", "beta"]:
            skill_dir = Path(tmpdir) / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {name} skill
---

Content for {name}
""")

        loader = SkillLoader(Path(tmpdir))
        skills = loader.list_skills()
        assert "alpha" in skills
        assert "beta" in skills
        assert len(skills) == 2

    print("PASS: test_v4_skill_loader_list_skills")
    return True


def test_v4_skill_tool_schema():
    """Test v4 Skill tool schema."""
    from v4_skills_agent import SKILL_TOOL

    assert SKILL_TOOL["name"] == "Skill"
    schema = SKILL_TOOL["input_schema"]
    assert "skill" in schema["properties"]
    assert "skill" in schema["required"]

    print("PASS: test_v4_skill_tool_schema")
    return True


# =============================================================================
# Path Safety Tests
# =============================================================================

def test_v3_safe_path():
    """Test v3 safe_path prevents path traversal."""
    from v3_subagent import safe_path, WORKDIR

    # Valid path
    p = safe_path("test.txt")
    assert str(p).startswith(str(WORKDIR))

    # Path traversal attempt
    try:
        safe_path("../../../etc/passwd")
        assert False, "Should reject path traversal"
    except ValueError as e:
        assert "escape" in str(e).lower()

    print("PASS: test_v3_safe_path")
    return True


# =============================================================================
# Configuration Tests (Extended)
# =============================================================================

def test_base_url_config():
    """Test ANTHROPIC_BASE_URL configuration."""
    orig = os.environ.get("ANTHROPIC_BASE_URL")

    try:
        os.environ["ANTHROPIC_BASE_URL"] = "https://custom.api.com"

        import importlib
        import v1_basic_agent
        importlib.reload(v1_basic_agent)

        # Check client was created (we can't easily verify base_url without mocking)
        assert v1_basic_agent.client is not None

        print("PASS: test_base_url_config")
        return True

    finally:
        if orig:
            os.environ["ANTHROPIC_BASE_URL"] = orig
        else:
            os.environ.pop("ANTHROPIC_BASE_URL", None)


# =============================================================================
# v5 ContextManager Tests
# =============================================================================

def test_v5_estimate_tokens():
    """Test v5 ContextManager token estimation."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    assert cm.estimate_tokens("") == 0
    assert cm.estimate_tokens("abcd") == 1
    assert cm.estimate_tokens("a" * 400) == 100

    print("PASS: test_v5_estimate_tokens")
    return True


def test_v5_microcompact_keeps_recent():
    """Test v5 microcompact keeps the most recent tool outputs."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {}} for i in range(5)]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 5000}
            for i in range(5)
        ]},
    ]

    result = cm.microcompact(messages)
    user_content = result[1]["content"]

    compacted = sum(1 for b in user_content if b.get("content") == "[Output compacted - re-read if needed]")
    preserved = sum(1 for b in user_content if len(b.get("content", "")) > 100)

    assert preserved == cm.KEEP_RECENT, f"Should keep {cm.KEEP_RECENT} recent, got {preserved}"
    assert compacted == 2, f"Should compact 2 old outputs, got {compacted}"

    print("PASS: test_v5_microcompact_keeps_recent")
    return True


def test_v5_microcompact_skips_small():
    """Test v5 microcompact doesn't compact small outputs."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "small output"}
        ]},
    ]

    result = cm.microcompact(messages)
    assert result[1]["content"][0]["content"] == "small output"

    print("PASS: test_v5_microcompact_skips_small")
    return True


def test_v5_should_compact():
    """Test v5 should_compact threshold detection."""
    from v5_compression_agent import ContextManager
    cm = ContextManager(max_context_tokens=100)

    small = [{"role": "user", "content": "hi"}]
    assert not cm.should_compact(small), "Small messages shouldn't trigger compact"

    large = [{"role": "user", "content": "x" * 1000}]
    assert cm.should_compact(large), "Large messages should trigger compact"

    print("PASS: test_v5_should_compact")
    return True


def test_v5_handle_large_output():
    """Test v5 handles oversized output correctly."""
    import tempfile
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    normal = "small output"
    assert cm.handle_large_output(normal) == normal

    large = "x" * (cm.MAX_OUTPUT_TOKENS * 4 + 100)
    result = cm.handle_large_output(large)
    assert "too large" in result.lower() or "Saved to" in result

    print("PASS: test_v5_handle_large_output")
    return True


def test_v5_save_transcript():
    """Test v5 saves transcript to disk."""
    import tempfile
    from pathlib import Path
    from v5_compression_agent import ContextManager

    with tempfile.TemporaryDirectory() as tmpdir:
        import v5_compression_agent
        orig = v5_compression_agent.TRANSCRIPT_DIR
        v5_compression_agent.TRANSCRIPT_DIR = Path(tmpdir)

        cm = ContextManager()
        cm.save_transcript([{"role": "user", "content": "test message"}])

        transcript = Path(tmpdir) / "transcript.jsonl"
        assert transcript.exists(), "Transcript file should exist"
        content = transcript.read_text()
        assert "test message" in content

        v5_compression_agent.TRANSCRIPT_DIR = orig

    print("PASS: test_v5_save_transcript")
    return True


# =============================================================================
# v6 TaskManager Tests
# =============================================================================

def test_v6_task_create():
    """Test v6 TaskManager create with auto-increment ID."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        t1 = tm.create("First task", "Description 1")
        t2 = tm.create("Second task", "Description 2")

        assert t1.id == "1"
        assert t2.id == "2"
        assert t1.subject == "First task"
        assert t1.status == "pending"

    print("PASS: test_v6_task_create")
    return True


def test_v6_task_get():
    """Test v6 TaskManager get by ID."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Test task", "Details")

        task = tm.get("1")
        assert task is not None
        assert task.subject == "Test task"

        assert tm.get("999") is None

    print("PASS: test_v6_task_get")
    return True


def test_v6_task_update_status():
    """Test v6 TaskManager status update."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Task", "Desc")

        updated = tm.update("1", status="in_progress")
        assert updated.status == "in_progress"

        updated = tm.update("1", status="completed")
        assert updated.status == "completed"

    print("PASS: test_v6_task_update_status")
    return True


def test_v6_task_dependencies():
    """Test v6 TaskManager dependency management."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Setup DB")
        tm.create("Write API")
        tm.create("Write tests")

        tm.update("2", addBlockedBy=["1"])
        tm.update("3", addBlockedBy=["1", "2"])

        t2 = tm.get("2")
        assert "1" in t2.blocked_by
        t3 = tm.get("3")
        assert "1" in t3.blocked_by
        assert "2" in t3.blocked_by

    print("PASS: test_v6_task_dependencies")
    return True


def test_v6_task_complete_clears_deps():
    """Test v6 completing a task clears it from others' blocked_by."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Task A")
        tm.create("Task B")
        tm.update("2", addBlockedBy=["1"])

        assert "1" in tm.get("2").blocked_by

        tm.update("1", status="completed")

        t2 = tm.get("2")
        assert "1" not in t2.blocked_by, "Completing task should clear dependency"

    print("PASS: test_v6_task_complete_clears_deps")
    return True


def test_v6_task_list():
    """Test v6 TaskManager list_all."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("A")
        tm.create("B")
        tm.create("C")

        tasks = tm.list_all()
        assert len(tasks) == 3
        subjects = [t.subject for t in tasks]
        assert "A" in subjects and "B" in subjects and "C" in subjects

    print("PASS: test_v6_task_list")
    return True


def test_v6_task_persistence():
    """Test v6 tasks persist as JSON files on disk."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm1 = TaskManager(Path(tmpdir))
        tm1.create("Persistent task", "Should survive reload")

        # Create new manager pointing to same dir
        tm2 = TaskManager(Path(tmpdir))
        task = tm2.get("1")
        assert task is not None
        assert task.subject == "Persistent task"

    print("PASS: test_v6_task_persistence")
    return True


def test_v6_task_delete():
    """Test v6 TaskManager delete."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("To delete")
        assert tm.delete("1") is True
        assert tm.get("1") is None
        assert tm.delete("999") is False

    print("PASS: test_v6_task_delete")
    return True


def test_v6_task_tools_in_all_tools():
    """Test v6 Task CRUD tools are in ALL_TOOLS."""
    from v6_tasks_agent import ALL_TOOLS
    tool_names = {t["name"] for t in ALL_TOOLS}
    assert "TaskCreate" in tool_names
    assert "TaskGet" in tool_names
    assert "TaskUpdate" in tool_names
    assert "TaskList" in tool_names

    print("PASS: test_v6_task_tools_in_all_tools")
    return True


# =============================================================================
# v7 BackgroundManager Tests
# =============================================================================

def test_v7_background_run():
    """Test v7 BackgroundManager runs tasks and returns task_id."""
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()

    task_id = bm.run_in_background(lambda: "result", task_type="bash")
    assert task_id.startswith("b"), f"Bash task should have 'b' prefix, got {task_id}"

    task_id2 = bm.run_in_background(lambda: "result2", task_type="agent")
    assert task_id2.startswith("a"), f"Agent task should have 'a' prefix, got {task_id2}"

    print("PASS: test_v7_background_run")
    return True


def test_v7_background_get_output_blocking():
    """Test v7 BackgroundManager blocking output retrieval."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()

    task_id = bm.run_in_background(lambda: (time.sleep(0.1), "done")[1], task_type="bash")

    result = bm.get_output(task_id, block=True, timeout=5000)
    assert result["status"] == "completed"
    assert result["output"] == "done"

    print("PASS: test_v7_background_get_output_blocking")
    return True


def test_v7_background_get_output_nonblocking():
    """Test v7 BackgroundManager non-blocking output retrieval."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()

    task_id = bm.run_in_background(lambda: (time.sleep(1), "done")[1], task_type="agent")

    result = bm.get_output(task_id, block=False)
    assert result["status"] == "running", f"Should be running, got {result['status']}"

    print("PASS: test_v7_background_get_output_nonblocking")
    return True


def test_v7_background_notifications():
    """Test v7 BackgroundManager notification queue."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()

    bm.run_in_background(lambda: "task1 done", task_type="bash")
    bm.run_in_background(lambda: "task2 done", task_type="agent")

    time.sleep(0.2)

    notifications = bm.drain_notifications()
    assert len(notifications) >= 2, f"Should have 2+ notifications, got {len(notifications)}"

    # Queue should be empty after drain
    assert len(bm.drain_notifications()) == 0

    print("PASS: test_v7_background_notifications")
    return True


def test_v7_background_stop():
    """Test v7 BackgroundManager task stopping."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()

    task_id = bm.run_in_background(lambda: (time.sleep(10), "never")[1], task_type="bash")
    result = bm.stop_task(task_id)
    assert result["status"] == "stopped"

    print("PASS: test_v7_background_stop")
    return True


def test_v7_tools_in_all_tools():
    """Test v7 TaskOutput and TaskStop are in ALL_TOOLS."""
    from v7_background_agent import ALL_TOOLS
    tool_names = {t["name"] for t in ALL_TOOLS}
    assert "TaskOutput" in tool_names
    assert "TaskStop" in tool_names

    print("PASS: test_v7_tools_in_all_tools")
    return True


# =============================================================================
# v8 TeammateManager Tests
# =============================================================================

def test_v8_create_team():
    """Test v8 TeammateManager team creation."""
    from v8_teammate_agent import TeammateManager
    tm = TeammateManager()

    result = tm.create_team("test-team")
    assert "created" in result.lower()

    result2 = tm.create_team("test-team")
    assert "already exists" in result2.lower()

    print("PASS: test_v8_create_team")
    return True


def test_v8_send_message():
    """Test v8 TeammateManager message sending via inbox."""
    import tempfile
    from pathlib import Path
    from v8_teammate_agent import TeammateManager, Teammate

    tm = TeammateManager()
    tm.create_team("msg-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="worker", team_name="msg-team", inbox_path=inbox)
    tm._teams["msg-team"]["worker"] = teammate

    tm.send_message("worker", "Hello!", msg_type="message", team_name="msg-team")

    msgs = tm.check_inbox("worker", "msg-team")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hello!"
    assert msgs[0]["type"] == "message"

    # Inbox should be cleared after check
    msgs2 = tm.check_inbox("worker", "msg-team")
    assert len(msgs2) == 0

    inbox.unlink(missing_ok=True)

    print("PASS: test_v8_send_message")
    return True


def test_v8_message_types():
    """Test v8 TeammateManager validates message types."""
    from v8_teammate_agent import TeammateManager
    tm = TeammateManager()

    result = tm.send_message("nobody", "test", msg_type="invalid_type")
    assert "invalid" in result.lower() or "error" in result.lower()

    print("PASS: test_v8_message_types")
    return True


def test_v8_delete_team():
    """Test v8 TeammateManager team deletion."""
    import tempfile
    from pathlib import Path
    from v8_teammate_agent import TeammateManager, Teammate

    tm = TeammateManager()
    tm.create_team("del-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="w1", team_name="del-team", inbox_path=inbox)
    tm._teams["del-team"]["w1"] = teammate

    result = tm.delete_team("del-team")
    assert "deleted" in result.lower()
    assert "del-team" not in tm._teams

    inbox.unlink(missing_ok=True)

    print("PASS: test_v8_delete_team")
    return True


def test_v8_team_tools_in_all_tools():
    """Test v8 Team tools are in ALL_TOOLS."""
    from v8_teammate_agent import ALL_TOOLS
    tool_names = {t["name"] for t in ALL_TOOLS}
    assert "TeamCreate" in tool_names
    assert "SendMessage" in tool_names
    assert "TeamDelete" in tool_names

    print("PASS: test_v8_team_tools_in_all_tools")
    return True


def test_v8_team_status():
    """Test v8 TeammateManager status reporting."""
    from v8_teammate_agent import TeammateManager
    tm = TeammateManager()

    assert "No teams" in tm.get_team_status()

    tm.create_team("status-team")
    status = tm.get_team_status("status-team")
    assert "status-team" in status

    print("PASS: test_v8_team_status")
    return True


# =============================================================================
# v5 Mechanism-Specific Unit Tests
# =============================================================================

def test_v5_compactable_tools():
    """Verify COMPACTABLE_TOOLS excludes write/edit tools."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()
    assert "bash" in cm.COMPACTABLE_TOOLS
    assert "read_file" in cm.COMPACTABLE_TOOLS
    assert "write_file" not in cm.COMPACTABLE_TOOLS
    assert "edit_file" not in cm.COMPACTABLE_TOOLS
    print("PASS: test_v5_compactable_tools")
    return True


def test_v5_auto_compact_source():
    """Verify auto_compact saves transcript + keeps recent messages."""
    import inspect
    from v5_compression_agent import ContextManager
    source = inspect.getsource(ContextManager.auto_compact)
    assert "save_transcript" in source, "auto_compact must archive before compressing"
    assert "messages[-5:]" in source, "auto_compact must keep recent 5 messages"
    print("PASS: test_v5_auto_compact_source")
    return True


# =============================================================================
# v6 Mechanism-Specific Unit Tests
# =============================================================================

def test_v6_dependency_bidirectional():
    """Verify addBlockedBy creates bidirectional links."""
    import tempfile
    from pathlib import Path
    from v6_tasks_agent import TaskManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Parent")
        tm.create("Child")
        tm.update("2", addBlockedBy=["1"])
        assert "1" in tm.get("2").blocked_by
        assert "2" in tm.get("1").blocks
    print("PASS: test_v6_dependency_bidirectional")
    return True


# =============================================================================
# v7 Mechanism-Specific Unit Tests
# =============================================================================

def test_v7_tool_count():
    """Verify v7 has exactly 12 tools."""
    from v7_background_agent import ALL_TOOLS
    assert len(ALL_TOOLS) == 12, f"v7 should have 12 tools, got {len(ALL_TOOLS)}"
    print("PASS: test_v7_tool_count")
    return True


def test_v7_daemon_threads():
    """Verify background tasks run in daemon threads."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()
    tid = bm.run_in_background(lambda: "x", task_type="bash")
    task = bm._tasks[tid]
    assert task.thread.daemon is True
    bm.get_output(tid, block=True, timeout=2000)
    print("PASS: test_v7_daemon_threads")
    return True


def test_v7_notification_drain_clears():
    """Verify drain_notifications clears the queue."""
    import time
    from v7_background_agent import BackgroundManager
    bm = BackgroundManager()
    bm.run_in_background(lambda: "done", task_type="bash")
    time.sleep(0.2)
    n1 = bm.drain_notifications()
    assert len(n1) >= 1
    n2 = bm.drain_notifications()
    assert len(n2) == 0, "Second drain should return empty"
    print("PASS: test_v7_notification_drain_clears")
    return True


# =============================================================================
# v8 Mechanism-Specific Unit Tests
# =============================================================================

def test_v8_tool_count():
    """Verify v8 has exactly 15 tools."""
    from v8_teammate_agent import ALL_TOOLS
    assert len(ALL_TOOLS) == 15, f"v8 should have 15 tools, got {len(ALL_TOOLS)}"
    print("PASS: test_v8_tool_count")
    return True


def test_v8_teammate_tools_subset():
    """Verify TEAMMATE_TOOLS is a proper subset of ALL_TOOLS."""
    from v8_teammate_agent import TEAMMATE_TOOLS, ALL_TOOLS
    mate_names = {t["name"] for t in TEAMMATE_TOOLS}
    all_names = {t["name"] for t in ALL_TOOLS}
    assert mate_names.issubset(all_names)
    assert len(TEAMMATE_TOOLS) < len(ALL_TOOLS)
    assert "TeamCreate" not in mate_names
    assert "TeamDelete" not in mate_names
    print("PASS: test_v8_teammate_tools_subset")
    return True


def test_v8_message_types_count():
    """Verify MESSAGE_TYPES has exactly 5 types."""
    from v8_teammate_agent import TeammateManager
    assert len(TeammateManager.MESSAGE_TYPES) == 5, \
        f"Expected 5 message types, got {len(TeammateManager.MESSAGE_TYPES)}"
    print("PASS: test_v8_message_types_count")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    tests = [
        # Basic tests
        test_imports,
        test_todo_manager_basic,
        test_todo_manager_constraints,
        test_reminder_constants,
        test_nag_reminder_in_agent_loop,
        test_env_config,
        test_default_model,
        test_tool_schemas,
        # TodoManager edge cases
        test_todo_manager_empty_list,
        test_todo_manager_status_transitions,
        test_todo_manager_missing_fields,
        test_todo_manager_invalid_status,
        test_todo_manager_render_format,
        # v3 tests
        test_v3_agent_types_structure,
        test_v3_get_tools_for_agent,
        test_v3_get_agent_descriptions,
        test_v3_task_tool_schema,
        # v4 tests
        test_v4_skill_loader_init,
        test_v4_skill_loader_parse_valid,
        test_v4_skill_loader_parse_invalid,
        test_v4_skill_loader_get_content,
        test_v4_skill_loader_list_skills,
        test_v4_skill_tool_schema,
        # Security tests
        test_v3_safe_path,
        # Config tests
        test_base_url_config,
        # v5 tests
        test_v5_estimate_tokens,
        test_v5_microcompact_keeps_recent,
        test_v5_microcompact_skips_small,
        test_v5_should_compact,
        test_v5_handle_large_output,
        test_v5_save_transcript,
        # v6 tests
        test_v6_task_create,
        test_v6_task_get,
        test_v6_task_update_status,
        test_v6_task_dependencies,
        test_v6_task_complete_clears_deps,
        test_v6_task_list,
        test_v6_task_persistence,
        test_v6_task_delete,
        test_v6_task_tools_in_all_tools,
        # v7 tests
        test_v7_background_run,
        test_v7_background_get_output_blocking,
        test_v7_background_get_output_nonblocking,
        test_v7_background_notifications,
        test_v7_background_stop,
        test_v7_tools_in_all_tools,
        # v8 tests
        test_v8_create_team,
        test_v8_send_message,
        test_v8_message_types,
        test_v8_delete_team,
        test_v8_team_tools_in_all_tools,
        test_v8_team_status,
        # v5 mechanism-specific
        test_v5_compactable_tools,
        test_v5_auto_compact_source,
        # v6 mechanism-specific
        test_v6_dependency_bidirectional,
        # v7 mechanism-specific
        test_v7_tool_count,
        test_v7_daemon_threads,
        test_v7_notification_drain_clears,
        # v8 mechanism-specific
        test_v8_tool_count,
        test_v8_teammate_tools_subset,
        test_v8_message_types_count,
    ]

    failed = []
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print('='*50)
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    print('='*50)

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All unit tests passed!")
        sys.exit(0)
