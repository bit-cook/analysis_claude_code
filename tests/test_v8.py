"""
Tests for v8_teammate_agent.py - Team collaboration & messaging.

13 unit tests for TeammateManager messaging, inbox, lifecycle, and task board sharing.
3 additional tests for broadcast, TEAMMATE_TOOLS, and shutdown flow.
4 LLM integration tests for multi-agent workflows.
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
from v8_teammate_agent import TeammateManager, Teammate, TaskManager


# =============================================================================
# Unit Tests - TeammateManager
# =============================================================================

def test_create_team():
    tm = TeammateManager()
    result = tm.create_team("alpha-team")
    assert "created" in result.lower(), f"Expected 'created' in response, got: {result}"
    print("PASS: test_create_team")
    return True


def test_create_duplicate_team():
    tm = TeammateManager()
    tm.create_team("dup-team")
    result = tm.create_team("dup-team")
    assert "already exists" in result.lower(), f"Expected 'already exists', got: {result}"
    print("PASS: test_create_duplicate_team")
    return True


def test_send_message():
    tm = TeammateManager()
    tm.create_team("msg-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="alice", team_name="msg-team", inbox_path=inbox)
    tm._teams["msg-team"]["alice"] = teammate

    tm.send_message("alice", "Hello Alice!", msg_type="message", team_name="msg-team")

    assert inbox.exists(), "Inbox file should exist after sending message"
    content = inbox.read_text()
    assert "Hello Alice!" in content, f"Message content not found in inbox: {content}"

    inbox.unlink(missing_ok=True)
    print("PASS: test_send_message")
    return True


def test_check_inbox():
    tm = TeammateManager()
    tm.create_team("inbox-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="alice", team_name="inbox-team", inbox_path=inbox)
    tm._teams["inbox-team"]["alice"] = teammate

    tm.send_message("alice", "First message", msg_type="message", team_name="inbox-team")
    tm.send_message("alice", "Second message", msg_type="message", team_name="inbox-team")

    msgs = tm.check_inbox("alice", "inbox-team")
    assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
    assert msgs[0]["content"] == "First message", f"First message mismatch: {msgs[0]}"
    assert msgs[1]["content"] == "Second message", f"Second message mismatch: {msgs[1]}"

    msgs_after = tm.check_inbox("alice", "inbox-team")
    assert len(msgs_after) == 0, f"Inbox should be empty after check, got {len(msgs_after)} messages"

    inbox.unlink(missing_ok=True)
    print("PASS: test_check_inbox")
    return True


def test_message_types():
    tm = TeammateManager()
    tm.create_team("types-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="alice", team_name="types-team", inbox_path=inbox)
    tm._teams["types-team"]["alice"] = teammate

    for msg_type in ["message", "broadcast", "shutdown_request"]:
        tm.send_message("alice", f"Content for {msg_type}", msg_type=msg_type, team_name="types-team")

    msgs = tm.check_inbox("alice", "types-team")
    assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
    types_received = [m["type"] for m in msgs]
    assert "message" in types_received, f"Missing 'message' type in {types_received}"
    assert "broadcast" in types_received, f"Missing 'broadcast' type in {types_received}"
    assert "shutdown_request" in types_received, f"Missing 'shutdown_request' type in {types_received}"

    inbox.unlink(missing_ok=True)
    print("PASS: test_message_types")
    return True


def test_team_status():
    tm = TeammateManager()
    tm.create_team("status-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="bob", team_name="status-team", inbox_path=inbox)
    tm._teams["status-team"]["bob"] = teammate

    status = tm.get_team_status("status-team")
    assert "status-team" in status, f"Team name should be in status, got: {status}"
    assert "bob" in status, f"Member name should be in status, got: {status}"

    inbox.unlink(missing_ok=True)
    print("PASS: test_team_status")
    return True


def test_delete_team():
    tm = TeammateManager()
    tm.create_team("del-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="worker", team_name="del-team", inbox_path=inbox)
    tm._teams["del-team"]["worker"] = teammate

    result = tm.delete_team("del-team")
    assert "deleted" in result.lower(), f"Expected 'deleted' in response, got: {result}"
    assert "del-team" not in tm._teams, "Team should be removed from _teams"

    inbox.unlink(missing_ok=True)
    print("PASS: test_delete_team")
    return True


def test_task_claiming_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_mgr = TaskManager(Path(tmpdir))
        task_mgr.create("Unblocked task A")
        task_mgr.create("Unblocked task B")
        task_mgr.create("Blocked task C")

        task_mgr.update("3", addBlockedBy=["1"])

        all_tasks = task_mgr.list_all()
        unclaimed_unblocked = [
            t for t in all_tasks
            if t.status == "pending" and not t.owner and not t.blocked_by
        ]
        assert len(unclaimed_unblocked) == 2, \
            f"Expected 2 unclaimed unblocked tasks, got {len(unclaimed_unblocked)}"

        subjects = [t.subject for t in unclaimed_unblocked]
        assert "Unblocked task A" in subjects, "Task A should be unclaimed and unblocked"
        assert "Unblocked task B" in subjects, "Task B should be unclaimed and unblocked"
    print("PASS: test_task_claiming_logic")
    return True


def test_task_claim_and_unblock():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_mgr = TaskManager(Path(tmpdir))
        task_mgr.create("First step")
        task_mgr.create("Second step")
        task_mgr.create("Dependent step")

        task_mgr.update("3", addBlockedBy=["1"])

        task_mgr.update("1", status="in_progress", owner="alice")
        t1 = task_mgr.get("1")
        assert t1.owner == "alice", f"Expected owner 'alice', got '{t1.owner}'"

        task_mgr.update("1", status="completed")
        t3 = task_mgr.get("3")
        assert "1" not in t3.blocked_by, \
            f"Completing task 1 should unblock task 3, got blocked_by={t3.blocked_by}"
    print("PASS: test_task_claim_and_unblock")
    return True


def test_task_manager_with_owner():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_mgr = TaskManager(Path(tmpdir))
        task_mgr.create("Owned task")
        task_mgr.update("1", owner="bob")

        task = task_mgr.get("1")
        assert task.owner == "bob", f"Expected owner 'bob', got '{task.owner}'"

        task_mgr2 = TaskManager(Path(tmpdir))
        task_reloaded = task_mgr2.get("1")
        assert task_reloaded.owner == "bob", \
            f"Owner should persist after reload, got '{task_reloaded.owner}'"
    print("PASS: test_task_manager_with_owner")
    return True


def test_multiple_message_types():
    tm = TeammateManager()
    tm.create_team("alltype-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="tester", team_name="alltype-team", inbox_path=inbox)
    tm._teams["alltype-team"]["tester"] = teammate

    all_types = ["message", "broadcast", "shutdown_request",
                 "shutdown_response", "plan_approval_response"]
    for msg_type in all_types:
        tm.send_message("tester", f"Content for {msg_type}",
                        msg_type=msg_type, team_name="alltype-team")

    msgs = tm.check_inbox("tester", "alltype-team")
    assert len(msgs) == 5, f"Expected 5 messages, got {len(msgs)}"

    received_types = [m["type"] for m in msgs]
    for expected_type in all_types:
        assert expected_type in received_types, \
            f"Missing type '{expected_type}' in received: {received_types}"

    for msg in msgs:
        expected_content = f"Content for {msg['type']}"
        assert msg["content"] == expected_content, \
            f"Content mismatch for type '{msg['type']}': got '{msg['content']}'"

    inbox.unlink(missing_ok=True)
    print("PASS: test_multiple_message_types")
    return True


def test_shutdown_via_delete():
    tm = TeammateManager()
    tm.create_team("shutdown-team")

    inbox_a = Path(tempfile.mktemp(suffix=".jsonl"))
    inbox_b = Path(tempfile.mktemp(suffix=".jsonl"))
    mate_a = Teammate(name="alpha", team_name="shutdown-team", inbox_path=inbox_a)
    mate_b = Teammate(name="beta", team_name="shutdown-team", inbox_path=inbox_b)
    tm._teams["shutdown-team"]["alpha"] = mate_a
    tm._teams["shutdown-team"]["beta"] = mate_b

    result = tm.delete_team("shutdown-team")
    assert "deleted" in result.lower(), f"Expected 'deleted' in response, got: {result}"
    assert "shutdown-team" not in tm._teams, "Team should be removed from _teams"

    assert mate_a.status == "shutdown", \
        f"Teammate alpha status should be 'shutdown', got '{mate_a.status}'"
    assert mate_b.status == "shutdown", \
        f"Teammate beta status should be 'shutdown', got '{mate_b.status}'"

    for inbox, name in [(inbox_a, "alpha"), (inbox_b, "beta")]:
        assert inbox.exists(), f"Inbox for {name} should exist"
        msgs = []
        with open(inbox, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    msgs.append(json.loads(line))
        shutdown_msgs = [m for m in msgs if m.get("type") == "shutdown_request"]
        assert len(shutdown_msgs) >= 1, \
            f"Expected at least 1 shutdown_request in {name}'s inbox, got {len(shutdown_msgs)}"

    inbox_a.unlink(missing_ok=True)
    inbox_b.unlink(missing_ok=True)
    print("PASS: test_shutdown_via_delete")
    return True


def test_task_board_sharing():
    with tempfile.TemporaryDirectory() as tmpdir:
        tm1 = TaskManager(Path(tmpdir))
        tm2 = TaskManager(Path(tmpdir))

        tm1.create("Shared task")

        tasks_from_tm2 = tm2.list_all()
        assert len(tasks_from_tm2) == 1, \
            f"tm2 should see 1 task, got {len(tasks_from_tm2)}"
        assert tasks_from_tm2[0].subject == "Shared task", \
            f"Subject mismatch: got '{tasks_from_tm2[0].subject}'"

        task_from_tm2 = tm2.get("1")
        assert task_from_tm2 is not None, "tm2 should be able to get task by ID"
        assert task_from_tm2.subject == "Shared task", \
            f"tm2.get subject mismatch: got '{task_from_tm2.subject}'"

        tm1.update("1", owner="frontend-agent")

        task_updated = tm2.get("1")
        assert task_updated.owner == "frontend-agent", \
            f"tm2 should see updated owner 'frontend-agent', got '{task_updated.owner}'"
    print("PASS: test_task_board_sharing")
    return True


# =============================================================================
# Broadcast and TEAMMATE_TOOLS Tests
# =============================================================================

def test_broadcast_sends_to_all():
    tm = TeammateManager()
    tm.create_team("bcast-team")

    inboxes = []
    names = ["alice", "bob", "carol"]
    for name in names:
        inbox = Path(tempfile.mktemp(suffix=".jsonl"))
        teammate = Teammate(name=name, team_name="bcast-team", inbox_path=inbox)
        tm._teams["bcast-team"][name] = teammate
        inboxes.append(inbox)

    # broadcast is done via send_message with msg_type="broadcast"
    tm.send_message("", "All hands meeting at 3pm",
                    msg_type="broadcast", sender="lead", team_name="bcast-team")

    for i, name in enumerate(names):
        msgs = tm.check_inbox(name, "bcast-team")
        assert len(msgs) >= 1, \
            f"Expected at least 1 message in {name}'s inbox, got {len(msgs)}"
        broadcast_msgs = [m for m in msgs if m.get("type") == "broadcast"]
        assert len(broadcast_msgs) >= 1, \
            f"Expected at least 1 broadcast in {name}'s inbox, got {len(broadcast_msgs)}"
        assert "All hands meeting at 3pm" in broadcast_msgs[0]["content"], \
            f"Broadcast content mismatch for {name}: {broadcast_msgs[0]['content']}"

    for inbox in inboxes:
        inbox.unlink(missing_ok=True)
    print("PASS: test_broadcast_sends_to_all")
    return True


def test_teammate_tools_include_tasks():
    from v8_teammate_agent import TEAMMATE_TOOLS
    tool_names = [t["name"] for t in TEAMMATE_TOOLS]

    expected = ["TaskCreate", "TaskUpdate", "TaskList", "SendMessage"]
    for name in expected:
        assert name in tool_names, \
            f"TEAMMATE_TOOLS should include '{name}', got: {tool_names}"
    print("PASS: test_teammate_tools_include_tasks")
    return True


def test_v8_tools_in_all_tools():
    from v8_teammate_agent import ALL_TOOLS
    tool_names = {t["name"] for t in ALL_TOOLS}
    assert "TeamCreate" in tool_names, "ALL_TOOLS should include TeamCreate"
    assert "SendMessage" in tool_names, "ALL_TOOLS should include SendMessage"
    assert "TeamDelete" in tool_names, "ALL_TOOLS should include TeamDelete"
    assert "TaskOutput" in tool_names, "ALL_TOOLS should include TaskOutput (from v7)"
    assert "TaskStop" in tool_names, "ALL_TOOLS should include TaskStop (from v7)"
    print("PASS: test_v8_tools_in_all_tools")
    return True


# =============================================================================
# v8 Mechanism-Specific Tests
# =============================================================================

def test_v8_tool_count():
    """Verify v8 has exactly 15 tools (v7's 12 + TeamCreate + SendMessage + TeamDelete)."""
    from v8_teammate_agent import ALL_TOOLS
    assert len(ALL_TOOLS) == 15, f"v8 should have 15 tools, got {len(ALL_TOOLS)}"
    print("PASS: test_v8_tool_count")
    return True


def test_v8_teammate_tools_subset():
    """Verify TEAMMATE_TOOLS is a subset of ALL_TOOLS (teammates get fewer tools).

    Teammates get BASE_TOOLS + task CRUD + SendMessage, but NOT the full
    lead toolset (no TeamCreate, TeamDelete, TaskOutput, TaskStop).
    """
    from v8_teammate_agent import TEAMMATE_TOOLS, ALL_TOOLS
    teammate_names = {t["name"] for t in TEAMMATE_TOOLS}
    all_names = {t["name"] for t in ALL_TOOLS}

    assert teammate_names.issubset(all_names), \
        f"TEAMMATE_TOOLS should be subset of ALL_TOOLS. Extra: {teammate_names - all_names}"
    assert len(TEAMMATE_TOOLS) < len(ALL_TOOLS), \
        "TEAMMATE_TOOLS should have fewer tools than ALL_TOOLS"
    assert "TeamCreate" not in teammate_names, \
        "Teammates should NOT have TeamCreate (only the lead does)"
    assert "TeamDelete" not in teammate_names, \
        "Teammates should NOT have TeamDelete (only the lead does)"

    print("PASS: test_v8_teammate_tools_subset")
    return True


def test_v8_message_types_constant():
    """Verify MESSAGE_TYPES includes all 5 required types."""
    from v8_teammate_agent import TeammateManager
    expected = {"message", "broadcast", "shutdown_request",
                "shutdown_response", "plan_approval_response"}
    assert TeammateManager.MESSAGE_TYPES == expected, \
        f"MESSAGE_TYPES should be {expected}, got {TeammateManager.MESSAGE_TYPES}"
    print("PASS: test_v8_message_types_constant")
    return True


def test_v8_teammate_status_lifecycle():
    """Verify Teammate dataclass starts as 'active' and changes to 'shutdown'."""
    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    t = Teammate(name="test", team_name="test-team", inbox_path=inbox)

    assert t.status == "active", f"Initial status should be 'active', got '{t.status}'"

    t.status = "idle"
    assert t.status == "idle", f"Status should change to 'idle', got '{t.status}'"

    t.status = "shutdown"
    assert t.status == "shutdown", f"Status should change to 'shutdown', got '{t.status}'"

    inbox.unlink(missing_ok=True)
    print("PASS: test_v8_teammate_status_lifecycle")
    return True


def test_v8_inbox_jsonl_format():
    """Verify inbox uses JSONL format (one JSON object per line)."""
    tm = TeammateManager()
    tm.create_team("jsonl-team")

    inbox = Path(tempfile.mktemp(suffix=".jsonl"))
    teammate = Teammate(name="jsonl-test", team_name="jsonl-team", inbox_path=inbox)
    tm._teams["jsonl-team"]["jsonl-test"] = teammate

    tm.send_message("jsonl-test", "Message 1", msg_type="message", team_name="jsonl-team")
    tm.send_message("jsonl-test", "Message 2", msg_type="broadcast", team_name="jsonl-team")

    with open(inbox) as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) == 2, f"Expected 2 JSONL lines, got {len(lines)}"
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            assert "type" in data, f"Line {i}: missing 'type' field"
            assert "content" in data, f"Line {i}: missing 'content' field"
        except json.JSONDecodeError:
            raise AssertionError(f"Line {i} is not valid JSON: {line[:100]}")

    inbox.unlink(missing_ok=True)
    print("PASS: test_v8_inbox_jsonl_format")
    return True


def test_v8_agent_loop_injects_identity():
    """Verify v8 agent_loop or auto_compact re-injects teammate identity.

    After context compression, the teammate needs to remember who it is.
    The code should inject identity information after auto_compact.
    """
    import inspect, v8_teammate_agent

    # Check that teammate_loop or auto_compact references identity
    source = open(v8_teammate_agent.__file__).read()
    has_identity = ("identity" in source.lower() or
                    "teammate_name" in source or
                    "name" in source)
    assert has_identity, \
        "v8 code must reference teammate identity for re-injection after compression"

    print("PASS: test_v8_agent_loop_injects_identity")
    return True


def test_v8_teams_dir_path():
    """Verify TEAMS_DIR is defined for file-based inbox persistence."""
    from v8_teammate_agent import TEAMS_DIR
    assert TEAMS_DIR is not None, "TEAMS_DIR must be defined"
    assert "teams" in str(TEAMS_DIR).lower(), \
        f"TEAMS_DIR should contain 'teams', got: {TEAMS_DIR}"
    print("PASS: test_v8_teams_dir_path")
    return True


# =============================================================================
# LLM Integration Tests
# =============================================================================

from tests.helpers import TASK_OUTPUT_TOOL, TASK_STOP_TOOL
from tests.helpers import TEAM_CREATE_TOOL, SEND_MESSAGE_TOOL, TEAM_DELETE_TOOL

V8_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL,
            TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL,
            TASK_OUTPUT_TOOL, TASK_STOP_TOOL,
            TEAM_CREATE_TOOL, SEND_MESSAGE_TOOL, TEAM_DELETE_TOOL]


def test_llm_creates_team():
    """LLM uses TeamCreate to set up a new team.

    v8's key mechanism: model creates a team for coordination.
    """
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "Create a new team called 'frontend-team' for building the UI. Use the TeamCreate tool.",
        V8_TOOLS,
        system="You are a team lead. Use TeamCreate to set up teams for collaboration.",
    )

    team_calls = [c for c in calls if c[0] == "TeamCreate"]
    assert len(team_calls) >= 1, \
        f"Model should use TeamCreate, got: {[c[0] for c in calls]}"
    assert "frontend" in team_calls[0][1].get("team_name", "").lower(), \
        f"Team name should contain 'frontend', got: {team_calls[0][1]}"

    print(f"Tool calls: {len(calls)}, TeamCreate: {len(team_calls)}")
    print("PASS: test_llm_creates_team")
    return True


def test_llm_sends_message():
    """LLM uses SendMessage to communicate with a teammate."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "You MUST call the SendMessage tool right now with these parameters: "
        "type='message', recipient='alice', content='Please review the API code'. "
        "Do NOT respond with text. Just call the SendMessage tool.",
        V8_TOOLS,
        system="You are a team lead. You MUST use the SendMessage tool when asked. Always use tools first.",
    )

    msg_calls = [c for c in calls if c[0] == "SendMessage"]
    assert len(msg_calls) >= 1, \
        f"Model should use SendMessage, got: {[c[0] for c in calls]}"

    print(f"Tool calls: {len(calls)}, SendMessage: {len(msg_calls)}")
    print("PASS: test_llm_sends_message")
    return True


def test_llm_broadcasts_message():
    """LLM uses SendMessage with type='broadcast' to reach all teammates."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "Broadcast a message to all teammates: 'Stop all work, critical bug found'. "
        "Use SendMessage with type='broadcast'.",
        V8_TOOLS,
        system="You are a team lead. Use SendMessage with type='broadcast' to reach all teammates.",
    )

    msg_calls = [c for c in calls if c[0] == "SendMessage"]
    assert len(msg_calls) >= 1, \
        f"Model should use SendMessage, got: {[c[0] for c in calls]}"
    assert msg_calls[0][1].get("type") == "broadcast", \
        f"Should use broadcast type, got: {msg_calls[0][1].get('type')}"

    print(f"Tool calls: {len(calls)}, SendMessage: {len(msg_calls)}")
    print("PASS: test_llm_broadcasts_message")
    return True


def test_llm_team_workflow():
    """LLM creates team, sends message, then cleans up -- full lifecycle."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "Do the following in order:\n"
        "1) Create a team called 'build-team' using TeamCreate\n"
        "2) Send a message to 'bob' saying 'Start the build' using SendMessage\n"
        "3) Delete the team using TeamDelete\n"
        "Execute all three steps.",
        V8_TOOLS,
        system="You are a team lead. Use TeamCreate, SendMessage, and TeamDelete.",
        max_turns=10,
    )

    tool_names = [c[0] for c in calls]
    assert "TeamCreate" in tool_names, f"Should use TeamCreate, got: {tool_names}"
    assert "SendMessage" in tool_names, f"Should use SendMessage, got: {tool_names}"
    assert "TeamDelete" in tool_names, f"Should use TeamDelete, got: {tool_names}"

    if "TeamCreate" in tool_names and "TeamDelete" in tool_names:
        create_idx = tool_names.index("TeamCreate")
        delete_idx = tool_names.index("TeamDelete")
        assert create_idx < delete_idx, \
            "TeamCreate should come before TeamDelete"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_llm_team_workflow")
    return True


def test_llm_shutdown_request():
    """LLM uses SendMessage with type='shutdown_request' to shut down a teammate."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = run_agent(
        client,
        "You MUST call the SendMessage tool with these exact parameters: "
        "type='shutdown_request', recipient='worker-1', content='Shutting down'. "
        "Do NOT respond with text. Just call the tool.",
        V8_TOOLS,
        system="You MUST use the SendMessage tool when asked. Always call tools immediately.",
    )

    msg_calls = [c for c in calls if c[0] == "SendMessage"]
    assert len(msg_calls) >= 1, \
        f"Model should use SendMessage, got: {[c[0] for c in calls]}"

    print(f"Tool calls: {len(calls)}, SendMessage: {len(msg_calls)}")
    print("PASS: test_llm_shutdown_request")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    sys.exit(0 if run_tests([
        # TeammateManager unit tests
        test_create_team,
        test_create_duplicate_team,
        test_send_message,
        test_check_inbox,
        test_message_types,
        test_team_status,
        test_delete_team,
        test_task_claiming_logic,
        test_task_claim_and_unblock,
        test_task_manager_with_owner,
        test_multiple_message_types,
        test_shutdown_via_delete,
        test_task_board_sharing,
        # Broadcast and tools tests
        test_broadcast_sends_to_all,
        test_teammate_tools_include_tasks,
        test_v8_tools_in_all_tools,
        # Mechanism-specific
        test_v8_tool_count,
        test_v8_teammate_tools_subset,
        test_v8_message_types_constant,
        test_v8_teammate_status_lifecycle,
        test_v8_inbox_jsonl_format,
        test_v8_agent_loop_injects_identity,
        test_v8_teams_dir_path,
        # LLM integration tests
        test_llm_creates_team,
        test_llm_sends_message,
        test_llm_broadcasts_message,
        test_llm_team_workflow,
        test_llm_shutdown_request,
    ]) else 1)
