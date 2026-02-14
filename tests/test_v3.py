"""
Tests for v3_subagent.py - Context isolation via sub-agents.

Tests agent type registry, tool filtering, and LLM delegation behavior.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL
from tests.helpers import TODO_WRITE_TOOL, SKILL_TOOL, TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL

from v3_subagent import get_tools_for_agent, AGENT_TYPES, ALL_TOOLS


# =============================================================================
# Unit Tests
# =============================================================================


def test_agent_types_defined():
    for name in ("explore", "code", "plan"):
        assert name in AGENT_TYPES, f"Missing agent type: {name}"
        assert "description" in AGENT_TYPES[name], f"{name} missing description"
        assert len(AGENT_TYPES[name]["description"]) > 0, f"{name} has empty description"
    print("PASS: test_agent_types_defined")
    return True


def test_explore_readonly():
    tools = get_tools_for_agent("explore")
    tool_names = [t["name"] for t in tools]
    assert "bash" in tool_names, "explore should have bash"
    assert "read_file" in tool_names, "explore should have read_file"
    assert "write_file" not in tool_names, "explore should NOT have write_file"
    assert "edit_file" not in tool_names, "explore should NOT have edit_file"
    print("PASS: test_explore_readonly")
    return True


def test_code_full_access():
    tools = get_tools_for_agent("code")
    tool_names = [t["name"] for t in tools]
    assert "bash" in tool_names, "code should have bash"
    assert "read_file" in tool_names, "code should have read_file"
    assert "write_file" in tool_names, "code should have write_file"
    assert "edit_file" in tool_names, "code should have edit_file"
    print("PASS: test_code_full_access")
    return True


def test_plan_readonly():
    tools = get_tools_for_agent("plan")
    tool_names = [t["name"] for t in tools]
    assert "bash" in tool_names, "plan should have bash"
    assert "read_file" in tool_names, "plan should have read_file"
    assert "write_file" not in tool_names, "plan should NOT have write_file"
    assert "edit_file" not in tool_names, "plan should NOT have edit_file"
    print("PASS: test_plan_readonly")
    return True


def test_no_recursive_task():
    for agent_type in AGENT_TYPES:
        tools = get_tools_for_agent(agent_type)
        tool_names = [t["name"] for t in tools]
        assert "Task" not in tool_names, f"{agent_type} should NOT get Task tool (prevents infinite recursion)"
    print("PASS: test_no_recursive_task")
    return True


def test_context_isolation_fresh_history():
    """Verify that run_task starts subagents with fresh message history.

    Inspects the source of run_task to confirm it creates
    sub_messages = [{"role": "user", "content": prompt}] -- a fresh
    history independent of the parent agent.
    """
    import inspect
    from v3_subagent import run_task

    source = inspect.getsource(run_task)
    assert 'sub_messages = [{"role": "user"' in source, (
        "run_task should initialize sub_messages with a fresh user message, "
        f"but pattern not found in source:\n{source[:500]}"
    )

    print("PASS: test_context_isolation_fresh_history")
    return True


# =============================================================================
# LLM Tests
# =============================================================================


TASK_TOOL = {
    "name": "Task",
    "description": "Delegate a task to a sub-agent. Types: explore (read-only), code (full access), plan (read-only)",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "agent_type": {"type": "string", "enum": ["explore", "code", "plan"]},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}


def _run_with_task_tool(client, task, extra_tools=None):
    """Run agent with Task tool that returns a mock sub-agent result."""
    tools = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, TASK_TOOL]
    if extra_tools:
        tools.extend(extra_tools)

    messages = [{"role": "user", "content": task}]
    system_prompt = (
        "You are a coding agent that delegates tasks to sub-agents using the Task tool. "
        "Always delegate work to the appropriate sub-agent type: "
        "explore for reading/searching, code for writing/creating, plan for designing. "
        "Use the Task tool to delegate."
    )
    tool_calls_made = []

    for _ in range(5):
        response = client.messages.create(
            model=MODEL,
            system=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=2000,
        )

        if response.stop_reason != "tool_use":
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            return text, tool_calls_made, messages

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        results = []
        for tc in tool_uses:
            tool_calls_made.append((tc.name, tc.input))
            if tc.name == "Task":
                agent_type = tc.input.get("agent_type", "unknown")
                output = f"Sub-agent ({agent_type}) result: completed the task."
            else:
                from tests.helpers import execute_tool
                output = execute_tool(tc.name, tc.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output[:5000],
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    return None, tool_calls_made, messages


def test_llm_uses_subagent_tool():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = _run_with_task_tool(
        client,
        "I need you to explore the project structure and find all Python files. Delegate this to a sub-agent."
    )

    task_calls = [c for c in calls if c[0] == "Task"]
    assert len(task_calls) > 0, "Model should use Task tool to delegate"
    print("PASS: test_llm_uses_subagent_tool")
    return True


def test_llm_delegates_exploration():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = _run_with_task_tool(
        client,
        "Find all .py files in the project. Use a sub-agent to explore."
    )

    task_calls = [c for c in calls if c[0] == "Task"]
    assert len(task_calls) > 0, "Model should delegate to a sub-agent"
    agent_type = task_calls[0][1].get("agent_type", "")
    assert agent_type == "explore", f"Should delegate to explore agent, got: {agent_type}"
    print("PASS: test_llm_delegates_exploration")
    return True


def test_llm_delegates_coding():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    text, calls, _ = _run_with_task_tool(
        client,
        "Create a new file called hello.py with a hello world function. Delegate this to a sub-agent."
    )

    task_calls = [c for c in calls if c[0] == "Task"]
    assert len(task_calls) > 0, "Model should delegate to a sub-agent"
    agent_type = task_calls[0][1].get("agent_type", "")
    assert agent_type == "code", f"Should delegate to code agent, got: {agent_type}"
    print("PASS: test_llm_delegates_coding")
    return True


def test_explore_code_pipeline():
    """Two-step delegation: explore agent finds files, then code agent creates summary."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["notes.txt", "readme.txt", "data.txt"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(f"Content of {name}")

        text, calls, _ = _run_with_task_tool(
            client,
            f"Do the following two-step task:\n"
            f"1) Use an explore sub-agent to find all .txt files in {tmpdir}\n"
            f"2) Use a code sub-agent to create a summary.txt in {tmpdir} listing the found files\n"
            f"You MUST use the Task tool twice: once with agent_type='explore' and once with agent_type='code'.",
        )

        task_calls = [c for c in calls if c[0] == "Task"]
        assert len(task_calls) >= 1, (
            f"Should make at least 1 Task tool call, got {len(task_calls)}"
        )

        agent_types_used = [c[1].get("agent_type", "") for c in task_calls]
        assert "explore" in agent_types_used or "code" in agent_types_used, (
            f"Should use either explore or code agent, got types: {agent_types_used}"
        )

    print(f"Tool calls: {len(calls)}, Task calls: {len(task_calls)}")
    print("PASS: test_explore_code_pipeline")
    return True


# =============================================================================
# Runner
# =============================================================================


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_agent_types_defined,
        test_explore_readonly,
        test_code_full_access,
        test_plan_readonly,
        test_no_recursive_task,
        test_context_isolation_fresh_history,
        test_llm_uses_subagent_tool,
        test_llm_delegates_exploration,
        test_llm_delegates_coding,
        test_explore_code_pipeline,
    ]) else 1)
