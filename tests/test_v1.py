"""
v1 test suite: "Model as agent" with 4 structured tools replacing raw bash for file ops.

Tests verify the model correctly uses read_file, write_file, edit_file alongside bash,
prefers structured tools over raw bash for file operations, and handles multi-tool workflows.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL, TODO_WRITE_TOOL

V1_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL]


def test_read_file():
    """Model reads a prepared file and reports content."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "secret.txt")
        with open(filepath, "w") as f:
            f.write("The secret code is: ALPHA-9827")

        response, calls, _ = run_agent(
            client,
            f"Read the file {filepath} and tell me the secret code.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert any(c[0] == "read_file" for c in calls), "Should use read_file tool"
        assert response and "ALPHA-9827" in response, (
            f"Response should contain 'ALPHA-9827', got: {response[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_read_file")
    return True


def test_write_file():
    """Model creates a file with write_file (NOT bash echo)."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "greeting.txt")

        response, calls, _ = run_agent(
            client,
            f"Create a file at {filepath} with content 'Hello from v1 agent!' using the write_file tool.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert any(c[0] == "write_file" for c in calls), "Should use write_file tool"
        assert os.path.exists(filepath), f"File {filepath} should exist"
        with open(filepath) as f:
            content = f.read()
        assert "Hello" in content, f"File should contain 'Hello', got: {content[:200]}"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_write_file")
    return True


def test_edit_file():
    """Model modifies existing file content using edit_file."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.ini")
        with open(filepath, "w") as f:
            f.write("mode=development\nport=3000\n")

        response, calls, _ = run_agent(
            client,
            f"Change 'mode=development' to 'mode=production' in {filepath}. Use the edit_file tool with old_string='mode=development' and new_string='mode=production'.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        with open(filepath) as f:
            content = f.read()
        assert "mode=production" in content, (
            f"File should contain 'mode=production', got: {content[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_edit_file")
    return True


def test_read_then_edit():
    """Multi-tool: model reads a file then edits it."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "version.txt")
        with open(filepath, "w") as f:
            f.write("version=1.0.0\nauthor=test\n")

        response, calls, _ = run_agent(
            client,
            f"First use read_file to read {filepath}. Then use edit_file with old_string='version=1.0.0' and new_string='version=2.0.0' to update the version.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        with open(filepath) as f:
            content = f.read()
        assert "version=2.0.0" in content, (
            f"File should contain 'version=2.0.0', got: {content[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_read_then_edit")
    return True


def test_write_then_verify():
    """Write a file then read to verify its contents."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "verify_me.txt")

        response, calls, _ = run_agent(
            client,
            f"Use write_file to create {filepath} with content 'verification_token_XYZ'. Then use read_file to read it back.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        assert os.path.exists(filepath), f"File {filepath} should exist"
        with open(filepath) as f:
            content = f.read()
        assert "verification_token_XYZ" in content, (
            f"File should contain 'verification_token_XYZ', got: {content[:200]}"
        )

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_write_then_verify")
    return True


def test_tool_selection():
    """Model prefers read_file over cat for reading when both bash and read_file are available."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "preference.txt")
        with open(filepath, "w") as f:
            f.write("tool_selection_test_data_12345")

        response, calls, _ = run_agent(
            client,
            f"Use the read_file tool to read {filepath}. Tell me the exact content of the file.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"
        assert response is not None, "Should return a response"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_tool_selection")
    return True


def test_multi_file_workflow():
    """Create 3 files, read one, edit another -- full workflow."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        response, calls, _ = run_agent(
            client,
            f"Do the following in {tmpdir}:\n"
            f"1) Use write_file to create alpha.txt with content 'alpha_original'\n"
            f"2) Use write_file to create beta.txt with content 'beta_original'\n"
            f"3) Use write_file to create gamma.txt with content 'gamma_original'\n"
            f"Report what you did.",
            V1_TOOLS,
            workdir=tmpdir,
            max_turns=15,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"

        files_created = sum(
            1 for f in ["alpha.txt", "beta.txt", "gamma.txt"]
            if os.path.exists(os.path.join(tmpdir, f))
        )
        assert files_created >= 2, f"Should create at least 2 files, got {files_created}"

    print(f"Tool calls: {len(calls)}, Files created: {files_created}")
    print("PASS: test_multi_file_workflow")
    return True


def test_error_recovery():
    """Model handles missing file gracefully (read nonexistent file, then responds helpfully)."""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = os.path.join(tmpdir, "does_not_exist.txt")

        response, calls, _ = run_agent(
            client,
            f"Try to read the file {nonexistent}. If it doesn't exist, tell me that and suggest what to do.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert response is not None, "Should return a response"
        resp_lower = response.lower()
        assert any(w in resp_lower for w in [
            "not", "error", "exist", "found", "cannot", "no such", "missing", "doesn't"
        ]), f"Response should acknowledge the missing file, got: {response[:200]}"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_error_recovery")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_read_file,
        test_write_file,
        test_edit_file,
        test_read_then_edit,
        test_write_then_verify,
        test_tool_selection,
        test_multi_file_workflow,
        test_error_recovery,
    ]) else 1)
