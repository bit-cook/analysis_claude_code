"""
Shared test infrastructure for learn-claude-code test suite.

Provides: LLM client, tool definitions, tool executor, agent loop runner.
Each test_v*.py imports from here.
"""
import os
import sys
import subprocess
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from project root for local testing
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)

MODEL = os.getenv("TEST_MODEL") or os.getenv("MODEL_ID") or "glm-5"


def get_client():
    """Get Anthropic client configured for testing.

    Tries TEST_API_KEY first, falls back to ANTHROPIC_API_KEY from .env.
    Returns None if neither is set (tests will SKIP).
    Clears ANTHROPIC_AUTH_TOKEN to avoid conflicts with third-party endpoints.
    """
    from anthropic import Anthropic

    api_key = os.getenv("TEST_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("TEST_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or "https://open.bigmodel.cn/api/anthropic"
    if not api_key:
        return None

    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    return Anthropic(api_key=api_key, base_url=base_url)


# =============================================================================
# Tool Definitions (Anthropic format)
# =============================================================================

BASH_TOOL = {
    "name": "bash",
    "description": "Run a shell command and return stdout+stderr",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read contents of a file at the given path",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file (creates parent dirs, overwrites if exists)",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}

EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": "Replace first occurrence of old_string with new_string in a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["path", "old_string", "new_string"],
    },
}

TODO_WRITE_TOOL = {
    "name": "TodoWrite",
    "description": "Update the todo list to track task progress",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string"},
                    },
                    "required": ["content", "status", "activeForm"],
                },
            }
        },
        "required": ["items"],
    },
}

SKILL_TOOL = {
    "name": "Skill",
    "description": "Load a skill to gain specialized knowledge. Available: test-skill",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Name of the skill to load"},
        },
        "required": ["skill"],
    },
}

TASK_CREATE_TOOL = {
    "name": "TaskCreate",
    "description": "Create a new task to track work",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Brief task title"},
            "description": {"type": "string", "description": "Task details"},
        },
        "required": ["subject", "description"],
    },
}

TASK_LIST_TOOL = {
    "name": "TaskList",
    "description": "List all tasks and their status",
    "input_schema": {"type": "object", "properties": {}},
}

TASK_UPDATE_TOOL = {
    "name": "TaskUpdate",
    "description": "Update a task's status or details",
    "input_schema": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        },
        "required": ["taskId"],
    },
}

TASK_GET_TOOL = {
    "name": "TaskGet",
    "description": "Get full details of a task by its ID",
    "input_schema": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to retrieve"},
        },
        "required": ["taskId"],
    },
}

TASK_OUTPUT_TOOL = {
    "name": "TaskOutput",
    "description": "Retrieve output from a background task by its task_id. Use block=true to wait for completion.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The background task ID (e.g. b1, a2)"},
            "block": {"type": "boolean", "description": "Whether to wait for completion", "default": True},
        },
        "required": ["task_id"],
    },
}

TASK_STOP_TOOL = {
    "name": "TaskStop",
    "description": "Stop a running background task by its task_id",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The background task ID to stop"},
        },
        "required": ["task_id"],
    },
}

TEAM_CREATE_TOOL = {
    "name": "TeamCreate",
    "description": "Create a new team for coordinating multiple agents working together",
    "input_schema": {
        "type": "object",
        "properties": {
            "team_name": {"type": "string", "description": "Name for the new team"},
            "description": {"type": "string", "description": "Team purpose"},
        },
        "required": ["team_name"],
    },
}

SEND_MESSAGE_TOOL = {
    "name": "SendMessage",
    "description": "Send a message to a teammate or broadcast to all. Types: message, broadcast, shutdown_request",
    "input_schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["message", "broadcast", "shutdown_request"],
                     "description": "Message type"},
            "recipient": {"type": "string", "description": "Teammate name (for message/shutdown_request)"},
            "content": {"type": "string", "description": "Message content"},
        },
        "required": ["type", "content"],
    },
}

TEAM_DELETE_TOOL = {
    "name": "TeamDelete",
    "description": "Delete a team and clean up all its resources",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}


# =============================================================================
# Tool Executor
# =============================================================================

def execute_tool(name, args, workdir=None, **kwargs):
    """Execute a tool call and return output string."""
    if workdir is None:
        workdir = os.getcwd()

    if name == "bash":
        cmd = args.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=workdir,
            )
            output = result.stdout + result.stderr
            return output.strip() or "(empty)"
        except Exception as e:
            return f"Error: {e}"

    elif name == "read_file":
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    elif name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "edit_file":
        path = args.get("path", "")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        try:
            with open(path, "r") as f:
                content = f.read()
            if old not in content:
                return f"Error: '{old}' not found in file"
            content = content.replace(old, new, 1)
            with open(path, "w") as f:
                f.write(content)
            return f"Replaced in {path}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "TodoWrite":
        items = args.get("items", [])
        result = []
        for item in items:
            icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
                item.get("status", ""), "[ ]")
            result.append(f"{icon} {item.get('content', '')}")
        done = len([i for i in items if i.get("status") == "completed"])
        return "\n".join(result) + f"\n({done}/{len(items)} completed)"

    elif name == "Skill":
        skill_name = args.get("skill", "")
        return (
            f'<skill-loaded name="{skill_name}">\n'
            f'# Skill: {skill_name}\n\n'
            f'This is the loaded skill content for {skill_name}.\n\n'
            f'Step 1: Analyze the problem\n'
            f'Step 2: Apply the solution\n'
            f'</skill-loaded>\n\n'
            f'Follow the instructions in the skill above.'
        )

    elif name == "TaskCreate":
        ctx = kwargs.get("ctx", {})
        tasks = ctx.setdefault("tasks", {})
        tid = str(len(tasks) + 1)
        tasks[tid] = {
            "id": tid,
            "subject": args.get("subject", ""),
            "description": args.get("description", ""),
            "status": "pending",
        }
        return f"Created task {tid}: {args.get('subject', '')}"

    elif name == "TaskList":
        ctx = kwargs.get("ctx", {})
        tasks = ctx.get("tasks", {})
        if not tasks:
            return "No tasks."
        lines = []
        for tid, t in tasks.items():
            lines.append(f"[{t['status']}] #{tid}: {t['subject']}")
        return "\n".join(lines)

    elif name == "TaskUpdate":
        ctx = kwargs.get("ctx", {})
        tasks = ctx.get("tasks", {})
        tid = args.get("taskId", "")
        if tid not in tasks:
            return f"Error: task {tid} not found"
        if "status" in args:
            tasks[tid]["status"] = args["status"]
        return f"Updated task {tid}: status={tasks[tid]['status']}"

    elif name == "TaskGet":
        ctx = kwargs.get("ctx", {})
        tasks = ctx.get("tasks", {})
        tid = args.get("taskId", "")
        if tid not in tasks:
            return f"Error: task {tid} not found"
        t = tasks[tid]
        return json.dumps(t, indent=2)

    elif name == "TaskOutput":
        ctx = kwargs.get("ctx", {})
        bg_tasks = ctx.get("background_tasks", {})
        tid = args.get("task_id", "")
        block = args.get("block", True)
        if tid in bg_tasks:
            return json.dumps(bg_tasks[tid])
        return json.dumps({"task_id": tid, "status": "completed", "output": f"Background task {tid} output: done"})

    elif name == "TaskStop":
        tid = args.get("task_id", "")
        return json.dumps({"task_id": tid, "status": "stopped", "message": f"Task {tid} has been stopped"})

    elif name == "TeamCreate":
        team_name = args.get("team_name", "unnamed")
        ctx = kwargs.get("ctx", {})
        teams = ctx.setdefault("teams", {})
        if team_name in teams:
            return f"Error: team '{team_name}' already exists"
        teams[team_name] = {"members": [], "messages": []}
        return f"Team '{team_name}' created successfully"

    elif name == "SendMessage":
        msg_type = args.get("type", "message")
        recipient = args.get("recipient", "")
        content = args.get("content", "")
        ctx = kwargs.get("ctx", {})
        messages = ctx.setdefault("sent_messages", [])
        messages.append({"type": msg_type, "recipient": recipient, "content": content})
        if msg_type == "broadcast":
            return f"Broadcast sent to all teammates: {content[:100]}"
        elif msg_type == "shutdown_request":
            return f"Shutdown request sent to {recipient}"
        return f"Message sent to {recipient}: {content[:100]}"

    elif name == "TeamDelete":
        ctx = kwargs.get("ctx", {})
        teams = ctx.get("teams", {})
        teams.clear()
        return "Team deleted and all resources cleaned up"

    return f"Unknown tool: {name}"


# =============================================================================
# Agent Loop Runner
# =============================================================================

def run_agent(client, task, tools, system=None, max_turns=10, workdir=None, ctx=None):
    """Run agent loop using Anthropic messages API.

    Returns (final_text, tool_calls_made, messages).
    tool_calls_made is a list of (tool_name, tool_input) tuples.
    """
    messages = [{"role": "user", "content": task}]
    system_prompt = system or "You are a coding agent. Use tools to complete tasks."
    tool_calls_made = []
    if ctx is None:
        ctx = {}

    for _ in range(max_turns):
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
            output = execute_tool(tc.name, tc.input, workdir=workdir, ctx=ctx)
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output[:5000],
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})

    return None, tool_calls_made, messages


# =============================================================================
# Test Runner
# =============================================================================

def run_tests(tests):
    """Run a list of test functions with consistent formatting."""
    failed = []
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('=' * 60)
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    print('=' * 60)

    if failed:
        print(f"FAILED: {failed}")
        return False
    else:
        print("All tests passed!")
        return True
