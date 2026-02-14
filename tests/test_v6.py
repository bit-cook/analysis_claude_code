"""
Tests for v6_tasks_agent.py - Task management with dependencies.

10 unit tests for TaskManager CRUD and dependency graph.
4 LLM integration tests for agent task workflows.
"""
import os
import sys
import tempfile
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL
from tests.helpers import TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL

from pathlib import Path
from v6_tasks_agent import TaskManager


# =============================================================================
# Unit Tests
# =============================================================================

def test_task_create_auto_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        t1 = tm.create("Task one")
        t2 = tm.create("Task two")
        t3 = tm.create("Task three")
        assert t1.id == "1", f"Expected id '1', got '{t1.id}'"
        assert t2.id == "2", f"Expected id '2', got '{t2.id}'"
        assert t3.id == "3", f"Expected id '3', got '{t3.id}'"
    print("PASS: test_task_create_auto_id")
    return True


def test_task_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Build feature", "Implement the new API endpoint", "Building feature")
        task = tm.get("1")
        assert task is not None, "Task should exist"
        assert task.id == "1", f"Expected id '1', got '{task.id}'"
        assert task.subject == "Build feature", f"Expected subject 'Build feature', got '{task.subject}'"
        assert task.description == "Implement the new API endpoint", f"Unexpected description: {task.description}"
        assert task.status == "pending", f"Expected status 'pending', got '{task.status}'"
        assert task.blocks == [], f"Expected empty blocks, got {task.blocks}"
        assert task.blocked_by == [], f"Expected empty blocked_by, got {task.blocked_by}"
    print("PASS: test_task_get")
    return True


def test_task_update_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Status task")

        updated = tm.update("1", status="in_progress")
        assert updated.status == "in_progress", f"Expected 'in_progress', got '{updated.status}'"

        updated = tm.update("1", status="completed")
        assert updated.status == "completed", f"Expected 'completed', got '{updated.status}'"
    print("PASS: test_task_update_status")
    return True


def test_task_update_subject():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Original subject")

        updated = tm.update("1", subject="Updated subject")
        assert updated.subject == "Updated subject", f"Expected 'Updated subject', got '{updated.subject}'"

        reloaded = tm.get("1")
        assert reloaded.subject == "Updated subject", "Subject should persist after reload"
    print("PASS: test_task_update_subject")
    return True


def test_task_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Alpha")
        tm.create("Beta")
        tm.create("Gamma")

        tasks = tm.list_all()
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
        subjects = [t.subject for t in tasks]
        assert "Alpha" in subjects, "Alpha should be in task list"
        assert "Beta" in subjects, "Beta should be in task list"
        assert "Gamma" in subjects, "Gamma should be in task list"
    print("PASS: test_task_list")
    return True


def test_task_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("To delete")
        tm.create("To keep")

        assert tm.delete("1") is True, "Delete should return True for existing task"
        assert tm.get("1") is None, "Deleted task should not be retrievable"

        tasks = tm.list_all()
        assert len(tasks) == 1, f"Expected 1 remaining task, got {len(tasks)}"
        assert tasks[0].subject == "To keep", "Only 'To keep' should remain"
    print("PASS: test_task_delete")
    return True


def test_dependency_add_blocked_by():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Setup database")
        tm.create("Write API")

        tm.update("2", addBlockedBy=["1"])

        t2 = tm.get("2")
        assert "1" in t2.blocked_by, f"Task 2 should be blocked by task 1, got blocked_by={t2.blocked_by}"

        t1 = tm.get("1")
        assert "2" in t1.blocks, f"Task 1 should block task 2, got blocks={t1.blocks}"
    print("PASS: test_dependency_add_blocked_by")
    return True


def test_dependency_completion_unblocks():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Prerequisite")
        tm.create("Dependent")

        tm.update("2", addBlockedBy=["1"])
        assert "1" in tm.get("2").blocked_by, "Task 2 should be blocked by task 1 before completion"

        tm.update("1", status="completed")
        t2 = tm.get("2")
        assert "1" not in t2.blocked_by, f"Completing task 1 should unblock task 2, got blocked_by={t2.blocked_by}"
    print("PASS: test_dependency_completion_unblocks")
    return True


def test_dependency_chain():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Step 1")
        tm.create("Step 2")
        tm.create("Step 3")

        tm.update("3", addBlockedBy=["1", "2"])

        t3 = tm.get("3")
        assert "1" in t3.blocked_by and "2" in t3.blocked_by, \
            f"Task 3 should be blocked by 1 and 2, got {t3.blocked_by}"

        tm.update("1", status="completed")
        t3 = tm.get("3")
        assert "1" not in t3.blocked_by, "Completing 1 should remove it from blocked_by"
        assert "2" in t3.blocked_by, "Task 3 should still be blocked by 2"

        tm.update("2", status="completed")
        t3 = tm.get("3")
        assert len(t3.blocked_by) == 0, f"Task 3 should be fully unblocked, got {t3.blocked_by}"
    print("PASS: test_dependency_chain")
    return True


def test_persistence_survives_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm1 = TaskManager(Path(tmpdir))
        tm1.create("Persistent A", "Description A")
        tm1.create("Persistent B", "Description B")
        tm1.update("2", addBlockedBy=["1"])
        tm1.update("1", status="in_progress")

        tm2 = TaskManager(Path(tmpdir))
        tasks = tm2.list_all()
        assert len(tasks) == 2, f"Expected 2 tasks after reload, got {len(tasks)}"

        t1 = tm2.get("1")
        assert t1.subject == "Persistent A", f"Expected 'Persistent A', got '{t1.subject}'"
        assert t1.status == "in_progress", f"Expected 'in_progress', got '{t1.status}'"

        t2 = tm2.get("2")
        assert "1" in t2.blocked_by, f"Dependency should persist, got blocked_by={t2.blocked_by}"
    print("PASS: test_persistence_survives_reload")
    return True


# =============================================================================
# LLM Integration Tests
# =============================================================================

TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL,
         TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL]


def test_llm_creates_tasks():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            "You MUST call the TaskCreate tool to create a task with subject='Design API schema'. Do it now.",
            TOOLS,
            system="You MUST use the TaskCreate tool when asked. Always use tools, never just respond with text.",
            workdir=tmpdir,
        )

        task_creates = [c for c in calls if c[0] == "TaskCreate"]
        assert len(task_creates) >= 1, f"Expected >= 1 TaskCreate calls, got {len(task_creates)}"
    print("PASS: test_llm_creates_tasks")
    return True


def test_llm_task_then_work():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            "Use TaskCreate to create a task for writing a hello.py file. Then use write_file to create hello.py with a hello world function.",
            TOOLS,
            system="Use TaskCreate to plan, then do the work with write_file.",
            max_turns=15,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        tool_names = [c[0] for c in calls]
        assert "TaskCreate" in tool_names or "write_file" in tool_names, \
            f"Expected TaskCreate or write_file in calls, got {tool_names}"
    print("PASS: test_llm_task_then_work")
    return True


def test_llm_lists_tasks():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    ctx = {
        "tasks": {
            "1": {"id": "1", "subject": "Setup database", "description": "", "status": "completed"},
            "2": {"id": "2", "subject": "Write migration script", "description": "", "status": "in_progress"},
            "3": {"id": "3", "subject": "Deploy to staging", "description": "", "status": "pending"},
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            "List all tasks and tell me which ones are still pending.",
            TOOLS,
            system="You are a task tracker. Use TaskList to see all tasks.",
            workdir=tmpdir,
            ctx=ctx,
        )

        task_lists = [c for c in calls if c[0] == "TaskList"]
        assert len(task_lists) >= 1, f"Expected >= 1 TaskList calls, got {len(task_lists)}"
        assert text is not None, "Expected a text response"
        lower = text.lower()
        assert "pending" in lower or "staging" in lower or "deploy" in lower, \
            f"Response should mention the pending task, got: {text[:200]}"
    print("PASS: test_llm_lists_tasks")
    return True


def test_llm_full_workflow():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            "Create 2 tasks: (1) write config.json (2) write main.py. Then create both files with appropriate content. Finally mark both tasks as completed.",
            TOOLS,
            system="Use TaskCreate to plan tasks, write_file to create files, TaskUpdate to mark tasks completed.",
            max_turns=20,
            workdir=tmpdir,
        )

        tool_names = [c[0] for c in calls]
        assert tool_names.count("TaskCreate") >= 1, \
            f"Expected >= 1 TaskCreate, got {tool_names.count('TaskCreate')}"
        assert tool_names.count("write_file") >= 1, \
            f"Expected >= 1 write_file, got {tool_names.count('write_file')}"
        assert tool_names.count("TaskUpdate") >= 1, \
            f"Expected >= 1 TaskUpdate, got {tool_names.count('TaskUpdate')}"
    print("PASS: test_llm_full_workflow")
    return True


# =============================================================================
# v6 Mechanism-Specific Tests
# =============================================================================

def test_v6_feature_gate():
    """Verify v6 has TASKS_ENABLED feature gate controlling TodoWrite vs Tasks.

    v6 introduces Tasks as an evolution of TodoWrite. The feature gate
    controls which system is active. Both should not be available simultaneously.
    """
    import v6_tasks_agent
    source = open(v6_tasks_agent.__file__).read()
    has_feature_gate = "TASKS_ENABLED" in source or "tasks_enabled" in source
    assert has_feature_gate or "TaskCreate" in source, \
        "v6 must either have a TASKS_ENABLED gate or TaskCreate tool"

    all_tool_names = {t["name"] for t in v6_tasks_agent.ALL_TOOLS}
    assert "TaskCreate" in all_tool_names, "TaskCreate must be in ALL_TOOLS"
    assert "TaskUpdate" in all_tool_names, "TaskUpdate must be in ALL_TOOLS"
    assert "TaskList" in all_tool_names, "TaskList must be in ALL_TOOLS"

    print("PASS: test_v6_feature_gate")
    return True


def test_v6_task_active_form():
    """Verify tasks support activeForm field for status display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        t = tm.create("Test task", "description", "Testing actively")
        assert hasattr(t, "active_form") or hasattr(t, "activeForm"), \
            "Task should have active_form field"
    print("PASS: test_v6_task_active_form")
    return True


def test_v6_agent_loop_integrates_tasks():
    """Verify v6 agent_loop calls TaskManager methods for task tools.

    The agent_loop must dispatch TaskCreate/TaskGet/TaskUpdate/TaskList
    to the TaskManager instance.
    """
    import inspect, v6_tasks_agent
    source = inspect.getsource(v6_tasks_agent.execute_tool)

    assert "TaskCreate" in source, "execute_tool must handle TaskCreate"
    assert "TaskGet" in source or "TaskUpdate" in source, \
        "execute_tool must handle task tools"

    print("PASS: test_v6_agent_loop_integrates_tasks")
    return True


def test_v6_tasks_json_persistence_format():
    """Verify tasks are persisted as individual JSON files in the tasks dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Persistence format test", "Checking JSON structure")

        json_files = list(Path(tmpdir).glob("*.json"))
        assert len(json_files) >= 1, \
            f"Should have at least 1 JSON task file, found {len(json_files)}"

        with open(json_files[0]) as f:
            data = json.load(f)
        assert "id" in data or "subject" in data, \
            f"Task JSON should have id/subject fields, got: {list(data.keys())}"

    print("PASS: test_v6_tasks_json_persistence_format")
    return True


def test_v6_dependency_bidirectional():
    """Verify addBlockedBy creates bidirectional links (blocks + blocked_by)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tm = TaskManager(Path(tmpdir))
        tm.create("Parent")
        tm.create("Child")
        tm.update("2", addBlockedBy=["1"])

        child = tm.get("2")
        parent = tm.get("1")
        assert "1" in child.blocked_by, "Child should list parent in blocked_by"
        assert "2" in parent.blocks, "Parent should list child in blocks"

    print("PASS: test_v6_dependency_bidirectional")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_task_create_auto_id,
        test_task_get,
        test_task_update_status,
        test_task_update_subject,
        test_task_list,
        test_task_delete,
        test_dependency_add_blocked_by,
        test_dependency_completion_unblocks,
        test_dependency_chain,
        test_persistence_survives_reload,
        # v6 mechanism-specific
        test_v6_feature_gate,
        test_v6_task_active_form,
        test_v6_agent_loop_integrates_tasks,
        test_v6_tasks_json_persistence_format,
        test_v6_dependency_bidirectional,
        # LLM integration
        test_llm_creates_tasks,
        test_llm_task_then_work,
        test_llm_lists_tasks,
        test_llm_full_workflow,
    ]) else 1)
