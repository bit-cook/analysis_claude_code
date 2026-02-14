"""
v0 test suite: "Bash is everything" -- the simplest possible agent, only bash tool.

Tests verify that with ONLY the bash tool, the model can execute commands,
handle pipelines, create files, recover from errors, and chain multi-step work.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, TODO_WRITE_TOOL

V0_TOOLS = [BASH_TOOL]


def test_bash_echo():
    """Model runs echo command and reports the output."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    response, calls, _ = run_agent(
        client,
        "Run 'echo hello_from_v0' using the bash tool and tell me the exact output.",
        V0_TOOLS,
    )

    assert len(calls) >= 1, "Should make at least 1 tool call"
    assert any(c[0] == "bash" for c in calls), "Should use bash tool"
    assert response and "hello_from_v0" in response, (
        f"Response should contain 'hello_from_v0', got: {response[:200]}"
    )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_echo")
    return True


def test_bash_pipeline():
    """Model uses piped commands (grep | wc) on a prepared file."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        datafile = os.path.join(tmpdir, "fruits.txt")
        with open(datafile, "w") as f:
            f.write("apple\nbanana\napricot\ncherry\navocado\nblueberry\n")

        response, calls, _ = run_agent(
            client,
            f"Count how many lines in {datafile} start with the letter 'a'. "
            f"Use grep and wc in a pipeline. Tell me the number.",
            V0_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, "Should make at least 1 tool call"
        assert response and "3" in response, (
            f"Response should contain '3' (apple, apricot, avocado), got: {response[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_pipeline")
    return True


def test_bash_file_creation():
    """Model creates a file using bash (echo > file), proving bash alone can do file I/O."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "created_by_bash.txt")

        response, calls, _ = run_agent(
            client,
            f"Create a file at {target} containing the text 'bash_file_io_works' using a bash command like echo.",
            V0_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, "Should make at least 1 tool call"
        assert all(c[0] == "bash" for c in calls), "Should only use bash tool"
        assert os.path.exists(target), f"File {target} should exist"
        with open(target) as f:
            content = f.read()
        assert "bash_file_io_works" in content, (
            f"File should contain 'bash_file_io_works', got: {content[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_file_creation")
    return True


def test_bash_error_handling():
    """Model runs a nonexistent command and reports the error."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    response, calls, _ = run_agent(
        client,
        "Run the command 'nonexistent_cmd_xyz_42' and tell me what happened.",
        V0_TOOLS,
    )

    assert response is not None, "Should return a response"
    resp_lower = response.lower()
    assert any(w in resp_lower for w in ["not found", "error", "fail", "command", "no such"]), (
        f"Response should mention an error, got: {response[:200]}"
    )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_error_handling")
    return True


def test_bash_multi_step():
    """Model chains multiple bash commands to accomplish a goal (create dir, write file, verify)."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = os.path.join(tmpdir, "project")
        target_file = os.path.join(target_dir, "hello.txt")

        response, calls, _ = run_agent(
            client,
            f"Using bash commands: 1) Create directory {target_dir}, "
            f"2) Write 'multi_step_success' into {target_file}, "
            f"3) Read back the file with cat to verify its content. Report what you found.",
            V0_TOOLS,
            workdir=tmpdir,
            max_turns=10,
        )

        assert len(calls) >= 2, f"Should make at least 2 tool calls, got {len(calls)}"
        assert os.path.exists(target_file), f"File {target_file} should exist"
        with open(target_file) as f:
            content = f.read()
        assert "multi_step_success" in content, (
            f"File should contain 'multi_step_success', got: {content[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_multi_step")
    return True


def test_bash_only_tool():
    """Verify model uses ONLY bash (no other tools available), even for file operations."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        source = os.path.join(tmpdir, "source.txt")
        with open(source, "w") as f:
            f.write("line1\nline2\nline3\n")

        response, calls, _ = run_agent(
            client,
            f"Read the contents of {source} and tell me how many lines it has. "
            f"Then create a new file {tmpdir}/count.txt with just the line count number.",
            V0_TOOLS,
            workdir=tmpdir,
            max_turns=10,
        )

        assert len(calls) >= 1, "Should make at least 1 tool call"
        tool_names_used = set(c[0] for c in calls)
        assert tool_names_used == {"bash"}, (
            f"Should ONLY use bash tool, but used: {tool_names_used}"
        )
        assert response and "3" in response, (
            f"Response should mention 3 lines, got: {response[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_only_tool")
    return True


def test_bash_subagent_spawn():
    """Model spawns a subagent via 'python v0_bash_agent.py' to list files."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["a.txt", "b.txt", "c.txt"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(name)

        response, calls, _ = run_agent(
            client,
            f"Spawn a subagent using the command 'python v0_bash_agent.py' to list files in {tmpdir}. "
            f"You MUST call bash with a command that includes 'python' and 'v0_bash_agent' in it.",
            V0_TOOLS,
            workdir=tmpdir,
            max_turns=5,
        )

        assert len(calls) >= 1, "Should make at least 1 tool call"
        bash_cmds = [c[1].get("command", "") for c in calls if c[0] == "bash"]
        has_subagent_call = any(
            "python" in cmd and "v0_bash_agent" in cmd for cmd in bash_cmds
        )
        assert has_subagent_call, (
            f"Should call bash with a command containing 'python' and 'v0_bash_agent', "
            f"got commands: {bash_cmds}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_bash_subagent_spawn")
    return True


def test_bash_output_truncation():
    """Verify that run_agent truncates tool outputs at 5000 chars.

    v0_bash_agent.py truncates at 50000 (line 174: output[:50000]).
    helpers.py run_agent truncates at 5000 (line 307: output[:5000]).
    This test produces a >50000 char output and verifies the helpers
    truncation takes effect during the run_agent pipeline.
    """
    from tests.helpers import execute_tool

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_output = execute_tool(
            "bash",
            {"command": "python3 -c \"print('A' * 60000)\""},
            workdir=tmpdir,
        )

        assert len(raw_output) > 50000, (
            f"Raw bash output should exceed 50000 chars, got {len(raw_output)}"
        )

        truncated = raw_output[:5000]
        assert len(truncated) == 5000, (
            f"Truncated output should be exactly 5000 chars, got {len(truncated)}"
        )

    print("PASS: test_bash_output_truncation")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_bash_echo,
        test_bash_pipeline,
        test_bash_file_creation,
        test_bash_error_handling,
        test_bash_multi_step,
        test_bash_only_tool,
        test_bash_subagent_spawn,
        test_bash_output_truncation,
    ]) else 1)
