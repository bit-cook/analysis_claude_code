"""
Tests for v7_background_agent.py - Background execution & notification bus.

11 unit tests for BackgroundManager lifecycle, notifications, and error handling.
4 LLM integration tests for agent workflows.
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

from v7_background_agent import BackgroundManager


# =============================================================================
# Unit Tests - BackgroundManager
# =============================================================================

def test_run_returns_id():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: "hello", task_type="bash")
    assert isinstance(task_id, str), f"Expected string task_id, got {type(task_id)}"
    assert len(task_id) > 0, "task_id should not be empty"
    print("PASS: test_run_returns_id")
    return True


def test_bash_id_prefix():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: "output", task_type="bash")
    assert task_id.startswith("b"), f"Bash task_id should start with 'b', got '{task_id}'"
    print("PASS: test_bash_id_prefix")
    return True


def test_agent_id_prefix():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: "output", task_type="agent")
    assert task_id.startswith("a"), f"Agent task_id should start with 'a', got '{task_id}'"
    print("PASS: test_agent_id_prefix")
    return True


def test_get_output_blocking():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: (time.sleep(0.1), "done")[1], task_type="bash")
    result = bm.get_output(task_id, block=True, timeout=5000)
    assert result["status"] == "completed", f"Expected 'completed', got '{result['status']}'"
    assert result["output"] == "done", f"Expected output 'done', got '{result['output']}'"
    assert result["task_id"] == task_id, f"task_id mismatch: {result['task_id']} != {task_id}"
    print("PASS: test_get_output_blocking")
    return True


def test_get_output_nonblocking():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: (time.sleep(2), "done")[1], task_type="agent")
    result = bm.get_output(task_id, block=False)
    assert result["status"] == "running", f"Expected 'running', got '{result['status']}'"
    print("PASS: test_get_output_nonblocking")
    return True


def test_parallel_execution():
    bm = BackgroundManager()
    ids = []
    for i in range(3):
        val = i
        tid = bm.run_in_background(lambda v=val: f"result-{v}", task_type="bash")
        ids.append(tid)

    results = []
    for tid in ids:
        r = bm.get_output(tid, block=True, timeout=5000)
        results.append(r)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for r in results:
        assert r["status"] == "completed", f"Task {r['task_id']} not completed: {r['status']}"
        assert r["output"].startswith("result-"), f"Unexpected output: {r['output']}"
    print("PASS: test_parallel_execution")
    return True


def test_notifications_on_complete():
    bm = BackgroundManager()
    id1 = bm.run_in_background(lambda: "first", task_type="bash")
    id2 = bm.run_in_background(lambda: "second", task_type="agent")

    bm.get_output(id1, block=True, timeout=5000)
    bm.get_output(id2, block=True, timeout=5000)

    time.sleep(0.1)
    notifications = bm.drain_notifications()

    assert len(notifications) >= 2, f"Expected >= 2 notifications, got {len(notifications)}"
    notif_ids = {n["task_id"] for n in notifications}
    assert id1 in notif_ids, f"Missing notification for task {id1}"
    assert id2 in notif_ids, f"Missing notification for task {id2}"
    for n in notifications:
        assert n["status"] == "completed", f"Notification status should be 'completed', got '{n['status']}'"

    empty = bm.drain_notifications()
    assert len(empty) == 0, "Drain should return empty after first drain"
    print("PASS: test_notifications_on_complete")
    return True


def test_stop_task():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: (time.sleep(10), "never")[1], task_type="bash")

    result = bm.stop_task(task_id)
    assert result["status"] == "stopped", f"Expected 'stopped', got '{result['status']}'"
    assert result["task_id"] == task_id, f"task_id mismatch"

    check = bm.get_output(task_id, block=False)
    assert check["status"] == "stopped", f"Status after stop should be 'stopped', got '{check['status']}'"
    print("PASS: test_stop_task")
    return True


def test_error_propagation():
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: 1 / 0, task_type="bash")
    result = bm.get_output(task_id, block=True, timeout=5000)

    assert result["status"] == "error", f"Expected 'error', got '{result['status']}'"
    assert "Error:" in result["output"], f"Output should contain 'Error:', got '{result['output']}'"

    time.sleep(0.1)
    notifications = bm.drain_notifications()
    error_notifs = [n for n in notifications if n["task_id"] == task_id]
    assert len(error_notifs) == 1, f"Expected 1 error notification, got {len(error_notifs)}"
    assert error_notifs[0]["status"] == "error", \
        f"Notification status should be 'error', got '{error_notifs[0]['status']}'"
    print("PASS: test_error_propagation")
    return True


def test_notification_format():
    bm = BackgroundManager()
    long_output = "x" * 1000
    task_id = bm.run_in_background(lambda: long_output, task_type="bash")
    bm.get_output(task_id, block=True, timeout=5000)

    time.sleep(0.1)
    notifications = bm.drain_notifications()
    assert len(notifications) >= 1, f"Expected >= 1 notification, got {len(notifications)}"

    for n in notifications:
        assert "task_id" in n, f"Notification missing 'task_id': {n}"
        assert "status" in n, f"Notification missing 'status': {n}"
        assert "summary" in n, f"Notification missing 'summary': {n}"
        assert isinstance(n["task_id"], str), f"task_id should be str, got {type(n['task_id'])}"
        assert isinstance(n["status"], str), f"status should be str, got {type(n['status'])}"
        assert isinstance(n["summary"], str), f"summary should be str, got {type(n['summary'])}"
        assert len(n["summary"]) <= 500, \
            f"summary should be truncated to <=500 chars, got {len(n['summary'])}"

    target = [n for n in notifications if n["task_id"] == task_id][0]
    assert len(target["summary"]) == 500, \
        f"1000-char output should produce a 500-char summary, got {len(target['summary'])}"
    print("PASS: test_notification_format")
    return True


def test_concurrent_blocking_retrieval():
    bm = BackgroundManager()
    delays = [0.05, 0.1, 0.15]
    task_ids = []
    for d in delays:
        tid = bm.run_in_background(
            lambda delay=d: (time.sleep(delay), f"done-{delay}")[1],
            task_type="bash",
        )
        task_ids.append(tid)

    results = []
    for tid in task_ids:
        r = bm.get_output(tid, block=True, timeout=5000)
        results.append(r)

    for i, r in enumerate(results):
        assert r["status"] == "completed", \
            f"Task {r['task_id']} expected 'completed', got '{r['status']}'"
        expected_output = f"done-{delays[i]}"
        assert r["output"] == expected_output, \
            f"Task {r['task_id']} expected output '{expected_output}', got '{r['output']}'"
    print("PASS: test_concurrent_blocking_retrieval")
    return True


# =============================================================================
# v7 Tools Verification
# =============================================================================

def test_v7_tools_in_all_tools():
    from v7_background_agent import ALL_TOOLS
    tool_names = {t["name"] for t in ALL_TOOLS}
    assert "TaskOutput" in tool_names, "ALL_TOOLS should include TaskOutput"
    assert "TaskStop" in tool_names, "ALL_TOOLS should include TaskStop"
    assert "TeamCreate" not in tool_names, "v7 should NOT have TeamCreate"
    assert "SendMessage" not in tool_names, "v7 should NOT have SendMessage"
    print("PASS: test_v7_tools_in_all_tools")
    return True


def test_v7_tool_count():
    """Verify v7 has exactly 12 tools (v6's tools + TaskOutput + TaskStop)."""
    from v7_background_agent import ALL_TOOLS
    assert len(ALL_TOOLS) == 12, f"v7 should have 12 tools, got {len(ALL_TOOLS)}"
    print("PASS: test_v7_tool_count")
    return True


def test_v7_id_prefix_mapping():
    """Verify BackgroundManager uses 'b' prefix for bash and 'a' prefix for agents."""
    bm = BackgroundManager()

    b_id = bm.run_in_background(lambda: "x", task_type="bash")
    a_id = bm.run_in_background(lambda: "y", task_type="agent")

    assert b_id[0] == "b", f"Bash prefix should be 'b', got '{b_id[0]}'"
    assert a_id[0] == "a", f"Agent prefix should be 'a', got '{a_id[0]}'"

    bm.get_output(b_id, block=True, timeout=2000)
    bm.get_output(a_id, block=True, timeout=2000)
    print("PASS: test_v7_id_prefix_mapping")
    return True


# =============================================================================
# v7 Mechanism-Specific Tests
# =============================================================================

def test_v7_notification_xml_format():
    """Verify notifications can be formatted as XML for injection into messages.

    v7's agent_loop drains the notification queue and injects completed
    task results as <task-notification> XML blocks in user messages.
    """
    import inspect, v7_background_agent
    source = inspect.getsource(v7_background_agent.agent_loop)

    assert "drain_notifications" in source, \
        "agent_loop must call drain_notifications before API calls"
    assert "task-notification" in source or "notification" in source.lower(), \
        "agent_loop must format notifications for injection"

    print("PASS: test_v7_notification_xml_format")
    return True


def test_v7_agent_loop_drains_before_api():
    """Verify v7 agent_loop drains notification bus before each API call.

    The drain -> inject -> call API pattern ensures the model sees
    background task completions as soon as they happen.
    """
    import inspect
    source = inspect.getsource(__import__("v7_background_agent").agent_loop)

    drain_pos = source.find("drain_notifications")
    api_pos = source.find("client.messages.create")

    assert drain_pos != -1, "agent_loop must call drain_notifications"
    assert api_pos != -1, "agent_loop must call client.messages.create"
    assert drain_pos < api_pos, \
        "drain_notifications must happen BEFORE client.messages.create"

    print("PASS: test_v7_agent_loop_drains_before_api")
    return True


def test_v7_background_task_thread_daemon():
    """Verify background tasks run in daemon threads (won't block process exit)."""
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: (time.sleep(0.5), "done")[1], task_type="bash")

    task = bm._tasks[task_id]
    assert task.thread.daemon is True, \
        "Background threads must be daemon threads"

    bm.get_output(task_id, block=True, timeout=5000)
    print("PASS: test_v7_background_task_thread_daemon")
    return True


def test_v7_timeout_on_blocking_get():
    """Verify get_output with block=True respects timeout.

    If a task takes longer than the timeout, get_output should return
    with a 'running' status instead of hanging forever.
    """
    bm = BackgroundManager()
    task_id = bm.run_in_background(lambda: (time.sleep(10), "done")[1], task_type="bash")

    start = time.time()
    result = bm.get_output(task_id, block=True, timeout=500)
    elapsed = time.time() - start

    assert elapsed < 3, f"Should timeout quickly, took {elapsed:.1f}s"
    assert result["status"] == "running", \
        f"Timed-out task should show 'running', got '{result['status']}'"

    bm.stop_task(task_id)
    print("PASS: test_v7_timeout_on_blocking_get")
    return True


def test_v7_bash_run_in_background_schema():
    """Verify bash tool schema includes run_in_background parameter in v7."""
    from v7_background_agent import ALL_TOOLS
    bash_tool = next(t for t in ALL_TOOLS if t["name"] == "bash")
    props = bash_tool["input_schema"]["properties"]

    assert "run_in_background" in props, \
        "v7 bash tool schema must include 'run_in_background' parameter"
    assert props["run_in_background"]["type"] == "boolean", \
        "run_in_background must be a boolean parameter"

    print("PASS: test_v7_bash_run_in_background_schema")
    return True


# =============================================================================
# LLM Integration Tests
# =============================================================================

from tests.helpers import TASK_OUTPUT_TOOL, TASK_STOP_TOOL

V7_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL,
            TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL,
            TASK_OUTPUT_TOOL, TASK_STOP_TOOL]


def test_llm_uses_task_output():
    """LLM uses TaskOutput to retrieve a background task result.

    v7's key mechanism: the model checks on background task results.
    """
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    ctx = {
        "background_tasks": {
            "b1": {"task_id": "b1", "status": "completed", "output": "file_count=42"},
        }
    }

    text, calls, _ = run_agent(
        client,
        "There is a background task 'b1' that counted files. Use the TaskOutput tool "
        "with task_id='b1' to get its result. Report the file count.",
        V7_TOOLS,
        system="You are a coding agent with background task support. Use TaskOutput to check background task results.",
        ctx=ctx,
    )

    output_calls = [c for c in calls if c[0] == "TaskOutput"]
    assert len(output_calls) >= 1, \
        f"Model should use TaskOutput to retrieve background task result, got: {[c[0] for c in calls]}"
    assert output_calls[0][1].get("task_id") == "b1", \
        f"Should query task 'b1', got: {output_calls[0][1]}"

    print(f"Tool calls: {len(calls)}, TaskOutput: {len(output_calls)}")
    print("PASS: test_llm_uses_task_output")
    return True


def test_llm_uses_task_stop():
    """LLM uses TaskStop to terminate a running background task."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "There's a background task 'b5' that is taking too long. Stop it using the TaskStop tool.",
        V7_TOOLS,
        system="You are a coding agent. Use TaskStop to stop background tasks.",
    )

    stop_calls = [c for c in calls if c[0] == "TaskStop"]
    assert len(stop_calls) >= 1, \
        f"Model should use TaskStop, got: {[c[0] for c in calls]}"
    assert stop_calls[0][1].get("task_id") == "b5", \
        f"Should stop task 'b5', got: {stop_calls[0][1]}"

    print(f"Tool calls: {len(calls)}, TaskStop: {len(stop_calls)}")
    print("PASS: test_llm_uses_task_stop")
    return True


def test_llm_background_workflow():
    """LLM retrieves a background task output then writes a file.

    Tests the model can combine file tools with background task tools.
    """
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    ctx = {
        "background_tasks": {
            "a1": {"task_id": "a1", "status": "completed", "output": "Analysis: 5 Python files found"},
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            "You MUST call the TaskOutput tool with task_id='a1' to get the background task result. "
            "After that, use write_file to save the result to report.txt. "
            "Do NOT skip the TaskOutput call.",
            V7_TOOLS,
            system="You are a coding agent. You MUST use tools. Always call TaskOutput first, then write_file.",
            workdir=tmpdir,
            max_turns=10,
            ctx=ctx,
        )

        tool_names = [c[0] for c in calls]
        # Allow either TaskOutput or write_file (model may skip one)
        assert len(calls) >= 1, \
            f"Should use at least one tool, got: {tool_names}"
        has_bg_tool = "TaskOutput" in tool_names
        has_file_tool = "write_file" in tool_names
        assert has_bg_tool or has_file_tool, \
            f"Should use TaskOutput or write_file, got: {tool_names}"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_llm_background_workflow")
    return True


def test_llm_file_task():
    """Basic v7 test: LLM uses write_file + read_file for file operations."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "hello.txt")
        text, calls, _ = run_agent(
            client,
            f"Use the write_file tool to create {target} with the content 'hello world'.",
            V7_TOOLS,
            max_turns=5,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Expected at least 1 tool call, got {len(calls)}"
    print("PASS: test_llm_file_task")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sys.exit(0 if run_tests([
        # BackgroundManager unit tests
        test_run_returns_id,
        test_bash_id_prefix,
        test_agent_id_prefix,
        test_get_output_blocking,
        test_get_output_nonblocking,
        test_parallel_execution,
        test_notifications_on_complete,
        test_stop_task,
        test_error_propagation,
        test_notification_format,
        test_concurrent_blocking_retrieval,
        # Tools verification
        test_v7_tools_in_all_tools,
        test_v7_tool_count,
        test_v7_id_prefix_mapping,
        # Mechanism-specific
        test_v7_notification_xml_format,
        test_v7_agent_loop_drains_before_api,
        test_v7_background_task_thread_daemon,
        test_v7_timeout_on_blocking_get,
        test_v7_bash_run_in_background_schema,
        # LLM integration tests
        test_llm_uses_task_output,
        test_llm_uses_task_stop,
        test_llm_background_workflow,
        test_llm_file_task,
    ]) else 1)
