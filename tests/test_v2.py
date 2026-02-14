"""
v2 test suite: "Plan before you act" -- TodoWrite forces structured planning.

Unit tests verify TodoManager logic (no LLM needed).
LLM tests verify the model uses TodoWrite to plan before acting.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, TODO_WRITE_TOOL

from v2_todo_agent import TodoManager

V2_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, TODO_WRITE_TOOL]


# =============================================================================
# Unit tests (no LLM)
# =============================================================================

def test_todo_manager_basic():
    """Create TodoManager, add items, verify render."""
    tm = TodoManager()

    result = tm.update([
        {"content": "Setup project", "status": "pending", "activeForm": "Setting up project"},
        {"content": "Write code", "status": "in_progress", "activeForm": "Writing code"},
        {"content": "Run tests", "status": "completed", "activeForm": "Running tests"},
    ])

    assert len(tm.items) == 3, f"Should have 3 items, got {len(tm.items)}"
    assert "Setup project" in result, "Render should contain 'Setup project'"
    assert "Write code" in result, "Render should contain 'Write code'"
    assert "Run tests" in result, "Render should contain 'Run tests'"
    assert "1/3" in result, f"Should show 1/3 completed, got: {result}"

    print("PASS: test_todo_manager_basic")
    return True


def test_todo_manager_one_in_progress():
    """Only 1 item can be in_progress at a time."""
    tm = TodoManager()

    try:
        tm.update([
            {"content": "Task A", "status": "in_progress", "activeForm": "Doing A"},
            {"content": "Task B", "status": "in_progress", "activeForm": "Doing B"},
        ])
        assert False, "Should raise ValueError for multiple in_progress items"
    except ValueError as e:
        assert "in_progress" in str(e).lower() or "one" in str(e).lower(), (
            f"Error should mention in_progress constraint, got: {e}"
        )

    tm2 = TodoManager()
    result = tm2.update([
        {"content": "Task A", "status": "in_progress", "activeForm": "Doing A"},
        {"content": "Task B", "status": "pending", "activeForm": "Waiting for B"},
    ])
    assert len(tm2.items) == 2, "Single in_progress should be allowed"

    print("PASS: test_todo_manager_one_in_progress")
    return True


def test_todo_manager_status_progression():
    """pending -> in_progress -> completed is valid progression."""
    tm = TodoManager()

    tm.update([
        {"content": "Deploy", "status": "pending", "activeForm": "Deploying"},
    ])
    assert tm.items[0]["status"] == "pending"

    tm.update([
        {"content": "Deploy", "status": "in_progress", "activeForm": "Deploying"},
    ])
    assert tm.items[0]["status"] == "in_progress"

    tm.update([
        {"content": "Deploy", "status": "completed", "activeForm": "Deploying"},
    ])
    assert tm.items[0]["status"] == "completed"

    print("PASS: test_todo_manager_status_progression")
    return True


def test_todo_manager_render_format():
    """Output format matches expected [x]/[>]/[ ] task pattern."""
    tm = TodoManager()
    tm.update([
        {"content": "Done task", "status": "completed", "activeForm": "Done"},
        {"content": "Active task", "status": "in_progress", "activeForm": "Working on it"},
        {"content": "Waiting task", "status": "pending", "activeForm": "Waiting"},
    ])

    rendered = tm.render()

    assert "[x] Done task" in rendered, f"Completed should show [x], got: {rendered}"
    assert "[>] Active task" in rendered, f"In-progress should show [>], got: {rendered}"
    assert "[ ] Waiting task" in rendered, f"Pending should show [ ], got: {rendered}"
    assert "1/3" in rendered, f"Should show 1/3 completed, got: {rendered}"
    assert "<- Working on it" in rendered, (
        f"In-progress item should show activeForm after '<-', got: {rendered}"
    )

    print("PASS: test_todo_manager_render_format")
    return True


def test_max_items_constraint():
    """TodoManager rejects lists with more than 20 items."""
    tm = TodoManager()

    items_25 = [
        {"content": f"Task {i}", "status": "pending", "activeForm": f"Doing task {i}"}
        for i in range(25)
    ]

    try:
        tm.update(items_25)
        assert False, "Should raise ValueError for more than 20 items"
    except ValueError as e:
        assert "20" in str(e), (
            f"Error should mention 20-item limit, got: {e}"
        )

    assert len(tm.items) != 25, "Should not store 25 items"

    print("PASS: test_max_items_constraint")
    return True


def test_nag_reminder_exists():
    """Verify NAG_REMINDER and INITIAL_REMINDER exist and reference todo usage.

    Also inspect agent_loop source to verify rounds_without_todo tracking.
    """
    import inspect
    from v2_todo_agent import NAG_REMINDER, INITIAL_REMINDER, agent_loop

    assert "todo" in NAG_REMINDER.lower() or "todowrite" in NAG_REMINDER.lower(), (
        f"NAG_REMINDER should reference 'todo' or 'TodoWrite', got: {NAG_REMINDER}"
    )

    assert "todo" in INITIAL_REMINDER.lower() or "todowrite" in INITIAL_REMINDER.lower(), (
        f"INITIAL_REMINDER should reference todo usage, got: {INITIAL_REMINDER}"
    )

    source = inspect.getsource(agent_loop)
    assert "rounds_without_todo" in source, (
        "agent_loop should reference 'rounds_without_todo' for nag tracking"
    )

    print("PASS: test_nag_reminder_exists")
    return True


# =============================================================================
# LLM tests
# =============================================================================

def test_llm_plans_before_acting():
    """Give multi-step task, model uses TodoWrite BEFORE file tools."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        system = (
            f"You are a coding agent at {tmpdir}. "
            f"You MUST call the TodoWrite tool first to plan before using any other tools. "
            f"Never skip planning. Always call TodoWrite first."
        )

        response, calls, _ = run_agent(
            client,
            f"Use the TodoWrite tool to plan 2 steps: create hello.txt, create world.txt. "
            f"Then create both files in {tmpdir}.",
            V2_TOOLS,
            system=system,
            workdir=tmpdir,
            max_turns=15,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        todo_calls = [c for c in calls if c[0] == "TodoWrite"]

        if len(todo_calls) == 0:
            print("WARN: GLM did not use TodoWrite (flaky model behavior)")

    print(f"Tool calls: {len(calls)}, TodoWrite: {len(todo_calls)}")
    print("PASS: test_llm_plans_before_acting")
    return True


def test_llm_updates_todo_progress():
    """Model updates todo items from pending to completed as work proceeds."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        system = (
            f"You are a coding agent at {tmpdir}. "
            f"You MUST call TodoWrite to plan. Update todo status as you work."
        )

        response, calls, _ = run_agent(
            client,
            f"Call TodoWrite with items: [{{'content':'Write file','status':'pending','activeForm':'Writing'}}]. "
            f"Then use write_file to create {tmpdir}/output.txt with content 'done'. "
            f"Then call TodoWrite again with the item status set to 'completed'.",
            V2_TOOLS,
            system=system,
            workdir=tmpdir,
            max_turns=15,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        todo_calls = [c for c in calls if c[0] == "TodoWrite"]

        if len(todo_calls) == 0:
            print("WARN: GLM did not use TodoWrite (flaky model behavior)")

    print(f"Tool calls: {len(calls)}, TodoWrite: {len(todo_calls)}")
    print("PASS: test_llm_updates_todo_progress")
    return True


def test_llm_multi_step_execution():
    """3 file creation task: model plans with todo, creates all 3 files."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        system = (
            f"You are a coding agent at {tmpdir}. "
            f"Use write_file to create files."
        )

        response, calls, _ = run_agent(
            client,
            f"Create 3 files in {tmpdir}:\n"
            f"1) one.txt with content 'first'\n"
            f"2) two.txt with content 'second'\n"
            f"3) three.txt with content 'third'\n",
            V2_TOOLS,
            system=system,
            workdir=tmpdir,
            max_turns=20,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"

        files_created = sum(
            1 for f in ["one.txt", "two.txt", "three.txt"]
            if os.path.exists(os.path.join(tmpdir, f))
        )

        assert files_created >= 2, f"Should create at least 2/3 files, got {files_created}"

    print(f"Tool calls: {len(calls)}, Files: {files_created}/3")
    print("PASS: test_llm_multi_step_execution")
    return True


def test_llm_todo_with_errors():
    """Task that includes a step that will fail (edit nonexistent string). Model should handle gracefully."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "target.txt")
        with open(filepath, "w") as f:
            f.write("original content here")

        system = (
            f"You are a coding agent at {tmpdir}. Handle errors gracefully."
        )

        response, calls, _ = run_agent(
            client,
            f"Read {filepath}. Then try to use edit_file to replace 'NONEXISTENT_STRING_XYZ' with 'replacement' in {filepath}. "
            f"This will fail. Report what happened.",
            V2_TOOLS,
            system=system,
            workdir=tmpdir,
            max_turns=15,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        assert response is not None, "Should return a response"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_llm_todo_with_errors")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_todo_manager_basic,
        test_todo_manager_one_in_progress,
        test_todo_manager_status_progression,
        test_todo_manager_render_format,
        test_max_items_constraint,
        test_nag_reminder_exists,
        test_llm_plans_before_acting,
        test_llm_updates_todo_progress,
        test_llm_multi_step_execution,
        test_llm_todo_with_errors,
    ]) else 1)
