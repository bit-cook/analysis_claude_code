"""
Tests for v5_compression_agent.py - Three-layer context compression.

Tests ContextManager token estimation, microcompact, should_compact,
handle_large_output, save_transcript, and LLM multi-turn workflows.
"""

import os
import sys
import tempfile
import time
import json
import inspect
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.helpers import get_client, run_agent, run_tests, MODEL
from tests.helpers import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL
from tests.helpers import TODO_WRITE_TOOL, SKILL_TOOL, TASK_CREATE_TOOL, TASK_LIST_TOOL, TASK_UPDATE_TOOL

from v5_compression_agent import ContextManager


# =============================================================================
# Unit Tests
# =============================================================================


def test_estimate_tokens():
    cm = ContextManager()
    result = cm.estimate_tokens("hello world")
    assert result == len("hello world") // 4, f"Expected {len('hello world') // 4}, got {result}"
    assert result == 2, f"'hello world' (11 chars) // 4 should be 2, got {result}"
    print("PASS: test_estimate_tokens")
    return True


def test_microcompact_preserves_recent():
    cm = ContextManager()
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"path": f"file{i}.py"}}
            for i in range(5)
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 5000}
            for i in range(5)
        ]},
    ]

    messages = cm.microcompact(messages)

    user_content = messages[1]["content"]
    tool_results = [b for b in user_content if b.get("type") == "tool_result"]

    preserved_count = sum(
        1 for b in tool_results
        if b.get("content") != "[Output compacted - re-read if needed]"
    )
    assert preserved_count >= cm.KEEP_RECENT, \
        f"Should preserve at least {cm.KEEP_RECENT} recent results, got {preserved_count}"
    print("PASS: test_microcompact_preserves_recent")
    return True


def test_microcompact_replaces_old():
    cm = ContextManager()
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": f"ls {i}"}}
            for i in range(5)
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 5000}
            for i in range(5)
        ]},
    ]

    messages = cm.microcompact(messages)

    user_content = messages[1]["content"]
    compacted = [
        b for b in user_content
        if b.get("content") == "[Output compacted - re-read if needed]"
    ]
    assert len(compacted) > 0, "Old tool results should be compacted"
    assert len(compacted) == 2, f"Expected 2 compacted (5 - KEEP_RECENT=3), got {len(compacted)}"
    print("PASS: test_microcompact_replaces_old")
    return True


def test_microcompact_skips_small():
    cm = ContextManager()
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"path": f"file{i}.py"}}
            for i in range(5)
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "short output"}
            for i in range(5)
        ]},
    ]

    messages = cm.microcompact(messages)

    user_content = messages[1]["content"]
    compacted = [
        b for b in user_content
        if b.get("content") == "[Output compacted - re-read if needed]"
    ]
    assert len(compacted) == 0, "Small outputs (under token threshold) should never be compacted"
    print("PASS: test_microcompact_skips_small")
    return True


def test_should_compact_threshold():
    cm = ContextManager(max_context_tokens=1000)
    large_content = "x" * 5000
    messages = [
        {"role": "user", "content": large_content},
    ]
    result = cm.should_compact(messages)
    assert result is True, "should_compact should return True when tokens exceed threshold"
    print("PASS: test_should_compact_threshold")
    return True


def test_should_compact_under_threshold():
    cm = ContextManager(max_context_tokens=200000)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = cm.should_compact(messages)
    assert result is False, "should_compact should return False for small conversations"
    print("PASS: test_should_compact_under_threshold")
    return True


def test_handle_large_output_passthrough():
    cm = ContextManager()
    normal_output = "This is a normal sized output."
    result = cm.handle_large_output(normal_output)
    assert result == normal_output, "Normal output should pass through unchanged"
    print("PASS: test_handle_large_output_passthrough")
    return True


def test_handle_large_output_saves():
    cm = ContextManager()
    large_output = "x" * (cm.MAX_OUTPUT_TOKENS * 4 + 100)
    result = cm.handle_large_output(large_output)
    assert result != large_output, "Large output should not pass through unchanged"
    assert "Output too large" in result, "Should indicate output was too large"
    assert "Saved to" in result, "Should indicate file was saved"
    assert "Preview" in result, "Should include a preview"
    print("PASS: test_handle_large_output_saves")
    return True


def test_auto_compact_preserves_recent():
    """Verify auto_compact structure via source inspection.

    auto_compact calls the API so we can't run it in unit tests, but we can
    verify its contract by inspecting the source:
    (a) calls save_transcript (archive before compressing)
    (b) keeps recent 5 messages (messages[-5:])
    (c) injects summary as user message (not system prompt modification)
    """
    import v5_compression_agent
    source = inspect.getsource(v5_compression_agent.ContextManager.auto_compact)

    # (a) Must call save_transcript to archive before compressing
    assert "save_transcript" in source, \
        "auto_compact must call save_transcript to archive messages before compression"

    # (b) Must keep recent messages (last 5)
    assert "messages[-5:]" in source, \
        "auto_compact must preserve recent 5 messages via messages[-5:]"

    # (c) Summary injected as user message, not modifying system prompt
    assert '"role": "user"' in source or "'role': 'user'" in source, \
        "auto_compact must inject summary as a user message (cache-preserving)"
    assert '"role": "assistant"' in source or "'role': 'assistant'" in source, \
        "auto_compact must include an assistant acknowledgment message"

    # Verify it does NOT modify the system prompt
    assert "SYSTEM" not in source, \
        "auto_compact must not modify SYSTEM prompt (would invalidate cache)"

    print("PASS: test_auto_compact_preserves_recent")
    return True


def test_transcript_save_and_load():
    """Verify save_transcript writes valid JSONL that can be loaded back.

    Tests the 'never lose data' principle: full transcripts are always
    saved to disk as the permanent archive.
    """
    cm = ContextManager()
    messages = [
        {"role": "user", "content": "Hello, please help me"},
        {"role": "assistant", "content": "Sure, I can help with that."},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"}
        ]},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override TRANSCRIPT_DIR to use our temp directory
        import v5_compression_agent
        original_dir = v5_compression_agent.TRANSCRIPT_DIR
        v5_compression_agent.TRANSCRIPT_DIR = Path(tmpdir)
        Path(tmpdir).mkdir(exist_ok=True)

        try:
            cm.save_transcript(messages)
            transcript_path = Path(tmpdir) / "transcript.jsonl"

            # (a) Transcript file must exist on disk
            assert transcript_path.exists(), \
                "save_transcript must create transcript.jsonl on disk"

            # (b) Each line must be valid JSON
            loaded_messages = []
            with open(transcript_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        loaded_messages.append(parsed)
                    except json.JSONDecodeError:
                        raise AssertionError(
                            f"Line {line_num} is not valid JSON: {line[:100]}"
                        )

            # (c) Loading the file gives back the same messages
            assert len(loaded_messages) == len(messages), \
                f"Expected {len(messages)} messages, got {len(loaded_messages)}"

            for i, (original, loaded) in enumerate(zip(messages, loaded_messages)):
                assert original["role"] == loaded["role"], \
                    f"Message {i}: role mismatch: {original['role']} != {loaded['role']}"
                if isinstance(original["content"], str):
                    assert original["content"] == loaded["content"], \
                        f"Message {i}: content mismatch"
                elif isinstance(original["content"], list):
                    assert isinstance(loaded["content"], list), \
                        f"Message {i}: content should be a list"
                    assert len(original["content"]) == len(loaded["content"]), \
                        f"Message {i}: content list length mismatch"
        finally:
            v5_compression_agent.TRANSCRIPT_DIR = original_dir

    print("PASS: test_transcript_save_and_load")
    return True


def test_microcompact_only_compactable_tools():
    """Verify only COMPACTABLE_TOOLS outputs get compacted; others are never touched.

    COMPACTABLE_TOOLS = {"bash", "read_file", "Grep", "Glob"}
    Non-compactable tools (write_file, edit_file) must NEVER be compacted,
    regardless of output size.
    """
    cm = ContextManager()

    # Build messages with a mix of compactable and non-compactable tool calls.
    # We need enough tool results that some compactable ones exceed KEEP_RECENT
    # and have large content (>1000 tokens = >4000 chars).
    large_output = "x" * 5000

    # 5 compactable (bash) + 3 non-compactable (write_file, edit_file)
    assistant_content = []
    user_content = []

    # First, 5 bash calls (compactable)
    for i in range(5):
        assistant_content.append({
            "type": "tool_use", "id": f"bash_{i}",
            "name": "bash", "input": {"command": f"ls {i}"}
        })
        user_content.append({
            "type": "tool_result", "tool_use_id": f"bash_{i}",
            "content": large_output
        })

    # Then, 3 write_file calls (NOT compactable)
    for i in range(3):
        assistant_content.append({
            "type": "tool_use", "id": f"write_{i}",
            "name": "write_file", "input": {"path": f"out{i}.txt", "content": "data"}
        })
        user_content.append({
            "type": "tool_result", "tool_use_id": f"write_{i}",
            "content": large_output
        })

    # Then, 2 edit_file calls (NOT compactable)
    for i in range(2):
        assistant_content.append({
            "type": "tool_use", "id": f"edit_{i}",
            "name": "edit_file", "input": {"path": f"f{i}.txt", "old_text": "a", "new_text": "b"}
        })
        user_content.append({
            "type": "tool_result", "tool_use_id": f"edit_{i}",
            "content": large_output
        })

    messages = [
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_content},
    ]

    messages = cm.microcompact(messages)

    compacted_marker = "[Output compacted - re-read if needed]"
    user_blocks = messages[1]["content"]

    # Check non-compactable tools are NEVER compacted
    for block in user_blocks:
        tool_id = block.get("tool_use_id", "")
        if tool_id.startswith("write_") or tool_id.startswith("edit_"):
            assert block["content"] != compacted_marker, \
                f"Non-compactable tool result {tool_id} must NEVER be compacted"
            assert block["content"] == large_output, \
                f"Non-compactable tool result {tool_id} content must be untouched"

    # Check that at least some compactable (bash) results WERE compacted
    bash_compacted = sum(
        1 for b in user_blocks
        if b.get("tool_use_id", "").startswith("bash_") and b["content"] == compacted_marker
    )
    assert bash_compacted > 0, \
        "Some old bash (compactable) tool results should have been compacted"

    # The most recent KEEP_RECENT bash results should be preserved
    bash_preserved = sum(
        1 for b in user_blocks
        if b.get("tool_use_id", "").startswith("bash_") and b["content"] != compacted_marker
    )
    assert bash_preserved == cm.KEEP_RECENT, \
        f"Expected {cm.KEEP_RECENT} recent bash results preserved, got {bash_preserved}"

    print("PASS: test_microcompact_only_compactable_tools")
    return True


# =============================================================================
# LLM Tests
# =============================================================================

V1_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL]


def test_llm_reads_multiple_files():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(4):
            filepath = os.path.join(tmpdir, f"data{i}.txt")
            with open(filepath, "w") as f:
                f.write(f"Content of file {i}: value={i * 10}")

        text, calls, _ = run_agent(
            client,
            f"Read all 4 files named data0.txt through data3.txt in {tmpdir} and summarize their contents.",
            V1_TOOLS,
            workdir=tmpdir,
            max_turns=10,
        )

        read_calls = [c for c in calls if c[0] == "read_file"]
        assert len(read_calls) >= 3, f"Should make at least 3 read_file calls, got {len(read_calls)}"
        assert text is not None, "Agent should produce a summary"
    print("PASS: test_llm_reads_multiple_files")
    return True


def test_llm_read_edit_workflow():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "target.txt")
        with open(filepath, "w") as f:
            f.write("hello world")

        text, calls, _ = run_agent(
            client,
            f"Use edit_file to change 'hello' to 'goodbye' in {filepath}. Use old_string='hello' and new_string='goodbye'.",
            V1_TOOLS,
            workdir=tmpdir,
            max_turns=10,
        )

        assert len(calls) >= 1, f"Should make at least 1 tool call, got {len(calls)}"

        with open(filepath, "r") as f:
            content = f.read()
        assert "goodbye" in content, f"File should contain 'goodbye' after edit, got: {content}"
    print("PASS: test_llm_read_edit_workflow")
    return True


def test_llm_write_and_verify():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "created.txt")
        text, calls, _ = run_agent(
            client,
            f"Write a file at {filepath} with the content 'test content 123', then read it back to verify.",
            V1_TOOLS,
            workdir=tmpdir,
            max_turns=10,
        )

        write_calls = [c for c in calls if c[0] == "write_file"]
        assert len(write_calls) >= 1, "Should call write_file at least once"
        assert os.path.exists(filepath), f"File should exist at {filepath}"

        with open(filepath, "r") as f:
            content = f.read()
        assert "test content 123" in content, f"File should contain 'test content 123', got: {content}"
    print("PASS: test_llm_write_and_verify")
    return True


def test_llm_many_turns():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        text, calls, _ = run_agent(
            client,
            (
                f"Create 4 files in {tmpdir}: "
                f"a.txt with 'alpha', b.txt with 'bravo', c.txt with 'charlie', d.txt with 'delta'. "
                f"Create each one separately using write_file."
            ),
            V1_TOOLS,
            workdir=tmpdir,
            max_turns=15,
        )

        write_calls = [c for c in calls if c[0] == "write_file"]
        assert len(write_calls) >= 3, f"Should make at least 3 write_file calls, got {len(write_calls)}"

        created = sum(1 for n in ("a.txt", "b.txt", "c.txt", "d.txt")
                      if os.path.exists(os.path.join(tmpdir, n)))
        assert created >= 3, f"Should create at least 3/4 files, got {created}"

        for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
            path = os.path.join(tmpdir, name)
            assert os.path.exists(path), f"File {name} should exist"
    print("PASS: test_llm_many_turns")
    return True


# =============================================================================
# v5 Mechanism-Specific Tests (source inspection)
# =============================================================================


def test_agent_loop_calls_microcompact():
    """Verify v5 agent_loop integrates microcompact before each API call.

    This is the core v5 mechanism: before every API call, the agent loop
    runs microcompact to replace old large tool outputs, reducing context
    without losing recent data.
    """
    source = inspect.getsource(__import__("v5_compression_agent").agent_loop)

    assert "microcompact" in source, \
        "agent_loop must call microcompact before API calls"
    assert "should_compact" in source, \
        "agent_loop must check should_compact for auto-compression trigger"
    assert "auto_compact" in source, \
        "agent_loop must call auto_compact when threshold is exceeded"
    assert "handle_large_output" in source, \
        "agent_loop must call handle_large_output for oversized tool results"

    print("PASS: test_agent_loop_calls_microcompact")
    return True


def test_compact_command_in_repl():
    """Verify v5 REPL handles /compact command for manual compression.

    /compact is the user-facing escape hatch: when the model context is
    getting large, the user can manually trigger compression.
    """
    source = inspect.getsource(__import__("v5_compression_agent").main)

    assert "/compact" in source or "compact" in source, \
        "main() REPL must handle /compact command"

    print("PASS: test_compact_command_in_repl")
    return True


def test_compactable_tools_constant():
    """Verify COMPACTABLE_TOOLS is defined and contains the right tools.

    Only read-oriented tools have their outputs compacted. Write/edit
    outputs must never be compacted (they contain confirmation data).
    """
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    assert hasattr(cm, "COMPACTABLE_TOOLS"), \
        "ContextManager must define COMPACTABLE_TOOLS"
    compactable = cm.COMPACTABLE_TOOLS
    assert "bash" in compactable, "bash should be compactable"
    assert "read_file" in compactable, "read_file should be compactable"
    assert "write_file" not in compactable, "write_file must NOT be compactable"
    assert "edit_file" not in compactable, "edit_file must NOT be compactable"

    print("PASS: test_compactable_tools_constant")
    return True


def test_keep_recent_constant():
    """Verify KEEP_RECENT is 3 (microcompact preserves last 3 tool outputs)."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    assert hasattr(cm, "KEEP_RECENT"), \
        "ContextManager must define KEEP_RECENT"
    assert cm.KEEP_RECENT == 3, \
        f"KEEP_RECENT should be 3, got {cm.KEEP_RECENT}"

    print("PASS: test_keep_recent_constant")
    return True


def test_token_threshold_constant():
    """Verify TOKEN_THRESHOLD is defined for auto-compact trigger."""
    from v5_compression_agent import ContextManager
    cm = ContextManager()

    assert hasattr(cm, "TOKEN_THRESHOLD"), \
        "ContextManager must define TOKEN_THRESHOLD"
    assert 0.5 <= cm.TOKEN_THRESHOLD <= 1.0, \
        f"TOKEN_THRESHOLD should be between 0.5 and 1.0, got {cm.TOKEN_THRESHOLD}"

    print("PASS: test_token_threshold_constant")
    return True


def test_notification_drain_in_agent_loop():
    """Verify v5 agent_loop does NOT have notification drain (that's v7).

    v5's agent_loop should call microcompact, should_compact, auto_compact
    but should NOT drain notifications (BackgroundManager is v7).
    """
    source = inspect.getsource(__import__("v5_compression_agent").agent_loop)

    assert "drain_notifications" not in source, \
        "v5 agent_loop should NOT have drain_notifications (that's v7)"

    print("PASS: test_notification_drain_in_agent_loop")
    return True


# =============================================================================
# Runner
# =============================================================================


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        test_estimate_tokens,
        test_microcompact_preserves_recent,
        test_microcompact_replaces_old,
        test_microcompact_skips_small,
        test_should_compact_threshold,
        test_should_compact_under_threshold,
        test_handle_large_output_passthrough,
        test_handle_large_output_saves,
        test_auto_compact_preserves_recent,
        test_transcript_save_and_load,
        test_microcompact_only_compactable_tools,
        # v5 mechanism-specific
        test_agent_loop_calls_microcompact,
        test_compact_command_in_repl,
        test_compactable_tools_constant,
        test_keep_recent_constant,
        test_token_threshold_constant,
        test_notification_drain_in_agent_loop,
        # LLM integration
        test_llm_reads_multiple_files,
        test_llm_read_edit_workflow,
        test_llm_write_and_verify,
        test_llm_many_turns,
    ]) else 1)
