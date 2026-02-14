#!/usr/bin/env python3
"""
v7_background_agent.py - Mini Claude Code: Background Execution & Notification Bus (~700 lines)

Core Philosophy: "Don't Wait, Orchestrate"
===========================================
v0-v6 agents are synchronous: they launch a subagent (v3) and block until
it completes. For one task that's fine. For three parallel tasks? Disaster.

    v6 (blocking):
      Agent ----[spawn A]----[wait...]----[wait...]----[result A]----
                                  ^ This time is wasted

    v7 (non-blocking):
      Agent ----[spawn A]----[spawn B]----[spawn C]----[other work]----
                   |              |            |
                   v              v            v
                [A runs]      [B runs]     [C runs]      (parallel)
                   |              |            |
                   +-- notification bus ---->  [Agent gets results]

Two new mechanisms make this work:

    BackgroundManager   Thread-based parallel execution. Bash commands
                        and subagents run in daemon threads. Each gets
                        a unique ID (b=bash, a=agent). Results are
                        collected via get_output() or drain_notifications().

    Notification Bus    When a background task completes, it pushes a
                        notification to a Queue. Before each API call,
                        the main loop drains the queue and injects
                        notifications as XML-formatted user messages.
                        The model sees them as <task-notification> blocks.

This is the INFRASTRUCTURE layer. v8 builds on it to add persistent
teammates that run as background threads with their own agent loops.

Usage:
    python v7_background_agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Queue

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# Configuration
# =============================================================================

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TASKS_DIR = WORKDIR / ".tasks"

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# BackgroundManager
# =============================================================================

@dataclass
class BackgroundTask:
    task_id: str
    task_type: str
    thread: threading.Thread = field(repr=False, default=None)
    output: str = ""
    status: str = "running"
    event: threading.Event = field(default_factory=threading.Event, repr=False)


class BackgroundManager:
    """
    Manages background execution of bash commands and subagents.

    ID prefixes indicate type:
        b = bash command
        a = local agent (subagent)

    When a background task completes, a notification is pushed to
    the notification queue. The main agent loop drains this queue
    before each API call, injecting notifications as user messages.
    """

    def __init__(self):
        self._tasks: dict[str, BackgroundTask] = {}
        self._notifications: Queue = Queue()
        self._lock = threading.Lock()

    def _gen_id(self, prefix: str) -> str:
        return f"{prefix}{uuid.uuid4().hex[:6]}"

    def run_in_background(self, func, task_type: str = "a") -> str:
        """
        Run a function in a background thread.
        Returns immediately with a task_id.
        """
        prefix = {"bash": "b", "agent": "a"}.get(task_type, "a")
        task_id = self._gen_id(prefix)

        bg_task = BackgroundTask(task_id=task_id, task_type=task_type)

        def wrapper():
            try:
                result = func()
                bg_task.output = result
                bg_task.status = "completed"
            except Exception as e:
                bg_task.output = f"Error: {e}"
                bg_task.status = "error"
            finally:
                bg_task.event.set()
                self._notifications.put({
                    "task_id": task_id,
                    "status": bg_task.status,
                    "summary": bg_task.output[:500],
                })

        thread = threading.Thread(target=wrapper, daemon=True)
        bg_task.thread = thread

        with self._lock:
            self._tasks[task_id] = bg_task

        thread.start()
        return task_id

    def get_output(self, task_id: str, block: bool = True, timeout: int = 30000) -> dict:
        """
        Get output from a background task.
        block=True waits for completion (up to timeout ms).
        """
        with self._lock:
            bg_task = self._tasks.get(task_id)

        if not bg_task:
            return {"error": f"Task {task_id} not found"}

        if block and bg_task.status == "running":
            bg_task.event.wait(timeout=timeout / 1000)

        return {
            "task_id": task_id,
            "status": bg_task.status,
            "output": bg_task.output,
        }

    def stop_task(self, task_id: str) -> dict:
        """Stop a running background task."""
        with self._lock:
            bg_task = self._tasks.get(task_id)

        if not bg_task:
            return {"error": f"Task {task_id} not found"}

        if bg_task.status == "running":
            bg_task.status = "stopped"
            bg_task.event.set()

        return {"task_id": task_id, "status": "stopped"}

    def drain_notifications(self) -> list:
        """Drain all pending notifications from the queue."""
        notifications = []
        while not self._notifications.empty():
            try:
                notifications.append(self._notifications.get_nowait())
            except Exception:
                break
        return notifications


BG = BackgroundManager()


# =============================================================================
# TaskManager (from v6)
# =============================================================================

@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str = "pending"
    active_form: str = ""
    owner: str = ""
    blocks: list = field(default_factory=list)
    blocked_by: list = field(default_factory=list)


class TaskManager:
    """
    File-based task management with dependency tracking.

    Each task is a JSON file in .tasks/ directory. Thread-level locking
    ensures safety when multiple agents (lead, subagents) access the
    same tasks concurrently.
    """

    def __init__(self, tasks_dir: Path = None):
        self.tasks_dir = tasks_dir or TASKS_DIR
        self.tasks_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._counter = self._load_counter()

    def _load_counter(self) -> int:
        existing = list(self.tasks_dir.glob("task_*.json"))
        if not existing:
            return 1
        ids = []
        for f in existing:
            try:
                ids.append(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        return max(ids) + 1 if ids else 1

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"task_{task_id}.json"

    def _save_task(self, task: Task):
        self._task_path(task.id).write_text(json.dumps(asdict(task), indent=2))

    def _load_task(self, task_id: str) -> Task:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        return Task(**json.loads(path.read_text()))

    def create(self, subject: str, description: str = "", active_form: str = "") -> Task:
        with self._lock:
            task = Task(
                id=str(self._counter),
                subject=subject,
                description=description,
                active_form=active_form or f"Working on: {subject}",
            )
            self._counter += 1
            self._save_task(task)
            return task

    def get(self, task_id: str) -> Task:
        return self._load_task(task_id)

    def update(self, task_id: str, **kwargs) -> Task:
        with self._lock:
            task = self._load_task(task_id)
            if not task:
                return None

            for key in ("status", "subject", "description", "active_form", "owner"):
                if key in kwargs:
                    setattr(task, key, kwargs[key])

            if "addBlocks" in kwargs:
                for blocked_id in kwargs["addBlocks"]:
                    if blocked_id not in task.blocks:
                        task.blocks.append(blocked_id)
                    blocked_task = self._load_task(blocked_id)
                    if blocked_task and task.id not in blocked_task.blocked_by:
                        blocked_task.blocked_by.append(task.id)
                        self._save_task(blocked_task)

            if "addBlockedBy" in kwargs:
                for blocker_id in kwargs["addBlockedBy"]:
                    if blocker_id not in task.blocked_by:
                        task.blocked_by.append(blocker_id)
                    blocker_task = self._load_task(blocker_id)
                    if blocker_task and task.id not in blocker_task.blocks:
                        blocker_task.blocks.append(task.id)
                        self._save_task(blocker_task)

            if kwargs.get("status") == "completed":
                self._clear_dependency(task.id)

            self._save_task(task)
            return task

    def _clear_dependency(self, completed_id: str):
        for path in self.tasks_dir.glob("task_*.json"):
            try:
                data = json.loads(path.read_text())
                if completed_id in data.get("blocked_by", []):
                    data["blocked_by"].remove(completed_id)
                    path.write_text(json.dumps(data, indent=2))
            except (json.JSONDecodeError, KeyError):
                pass

    def list_all(self) -> list:
        tasks = []
        for path in sorted(self.tasks_dir.glob("task_*.json")):
            try:
                tasks.append(Task(**json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError):
                pass
        return tasks

    def delete(self, task_id: str) -> bool:
        path = self._task_path(task_id)
        if path.exists():
            path.unlink()
            return True
        return False


TASK_MGR = TaskManager()


# =============================================================================
# ContextManager (from v5)
# =============================================================================

class ContextManager:
    """Three-layer context compression: microcompact, should_compact, auto_compact."""

    COMPACTABLE_TOOLS = {"bash", "read_file", "Grep", "Glob"}
    KEEP_RECENT = 3
    TOKEN_THRESHOLD = 0.93
    MAX_OUTPUT_TOKENS = 40000

    def __init__(self, max_context_tokens: int = 200000):
        self.max_context_tokens = max_context_tokens
        TRANSCRIPT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def microcompact(self, messages: list) -> list:
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for j, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_name = self._find_tool_name(messages, block.get("tool_use_id", ""))
                    if tool_name in self.COMPACTABLE_TOOLS:
                        tool_result_indices.append((i, j, block))

        to_compact = tool_result_indices[:-self.KEEP_RECENT] if len(tool_result_indices) > self.KEEP_RECENT else []
        for i, j, block in to_compact:
            content_str = block.get("content", "")
            if isinstance(content_str, str) and self.estimate_tokens(content_str) > 1000:
                block["content"] = "[Output compacted - re-read if needed]"
        return messages

    def should_compact(self, messages: list) -> bool:
        total = sum(self.estimate_tokens(json.dumps(m, default=str)) for m in messages)
        return total > self.max_context_tokens * self.TOKEN_THRESHOLD

    def auto_compact(self, messages: list) -> list:
        self.save_transcript(messages)
        conversation_text = self._messages_to_text(messages)
        summary_response = client.messages.create(
            model=MODEL,
            system="You are a conversation summarizer. Be concise but thorough.",
            messages=[{"role": "user", "content": f"Summarize this conversation chronologically. Include: goals, actions taken, decisions made, current state, and pending work.\n\n{conversation_text[:100000]}"}],
            max_tokens=2000,
        )
        summary = summary_response.content[0].text
        recent = messages[-5:] if len(messages) > 5 else messages[-2:]
        return [
            {"role": "user", "content": f"[Conversation compressed]\n\n{summary}"},
            {"role": "assistant", "content": "Understood. Continuing work with compressed context."},
            *recent
        ]

    def handle_large_output(self, output: str) -> str:
        if self.estimate_tokens(output) <= self.MAX_OUTPUT_TOKENS:
            return output
        path = TRANSCRIPT_DIR / f"output_{int(time.time())}.txt"
        path.write_text(output)
        return f"Output too large ({self.estimate_tokens(output)} tokens). Saved to: {path}\n\nPreview:\n{output[:2000]}..."

    def save_transcript(self, messages: list):
        path = TRANSCRIPT_DIR / "transcript.jsonl"
        with open(path, "a") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str) + "\n")

    def _find_tool_name(self, messages: list, tool_use_id: str) -> str:
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "id") and block.id == tool_use_id:
                        return block.name
                    if isinstance(block, dict) and block.get("id") == tool_use_id:
                        return block.get("name", "")
        return ""

    def _messages_to_text(self, messages: list) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"[{role}] {content[:500]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            lines.append(f"[tool_result] {str(block.get('content', ''))[:200]}")
                        elif block.get("type") == "text":
                            lines.append(f"[{role}] {block.get('text', '')[:500]}")
                    elif hasattr(block, "text"):
                        lines.append(f"[{role}] {block.text[:500]}")
        return "\n".join(lines)


CTX = ContextManager()


# =============================================================================
# SkillLoader (from v4)
# =============================================================================

class SkillLoader:
    """Loads and manages skills from SKILL.md files."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        content = path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None
        frontmatter, body = match.groups()
        metadata = {}
        for line in frontmatter.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")
        if "name" not in metadata or "description" not in metadata:
            return None
        return {"name": metadata["name"], "description": metadata["description"], "body": body.strip(), "path": path, "dir": path.parent}

    def load_skills(self):
        if not self.skills_dir.exists():
            return
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            skill = self.parse_skill_md(skill_md)
            if skill:
                self.skills[skill["name"]] = skill

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        return "\n".join(f"- {name}: {skill['description']}" for name, skill in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        if name not in self.skills:
            return None
        skill = self.skills[name]
        content = f"# Skill: {skill['name']}\n\n{skill['body']}"
        resources = []
        for folder, label in [("scripts", "Scripts"), ("references", "References"), ("assets", "Assets")]:
            folder_path = skill["dir"] / folder
            if folder_path.exists():
                files = list(folder_path.glob("*"))
                if files:
                    resources.append(f"{label}: {', '.join(f.name for f in files)}")
        if resources:
            content += f"\n\n**Available resources in {skill['dir']}:**\n" + "\n".join(f"- {r}" for r in resources)
        return content

    def list_skills(self) -> list:
        return list(self.skills.keys())


SKILLS = SkillLoader(SKILLS_DIR)


# =============================================================================
# Agent Type Registry (from v3)
# =============================================================================

AGENT_TYPES = {
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
}


def get_agent_descriptions() -> str:
    return "\n".join(f"- {name}: {cfg['description']}" for name, cfg in AGENT_TYPES.items())


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents available** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

You can run tasks in background with run_in_background=true on Task or bash tools.
When a background task completes, you'll receive a <task-notification> with the result.
Use TaskOutput to get full results. Use TaskStop to terminate a running task.

Rules:
- Use TaskCreate/TaskUpdate to track multi-step work
- Set run_in_background=true for independent parallel work
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Tool Definitions
# =============================================================================

BASE_TOOLS = [
    {"name": "bash", "description": "Run shell command. Set run_in_background=true for background execution.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "run_in_background": {"type": "boolean"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]

SUBAGENT_TOOL = {
    "name": "Task",
    "description": f"Spawn a subagent for a focused subtask.\n\nAgent types:\n{get_agent_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task description"},
            "prompt": {"type": "string", "description": "Detailed instructions"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
            "run_in_background": {"type": "boolean"},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

SKILL_TOOL = {
    "name": "Skill",
    "description": f"Load a skill for specialized knowledge.\n\nAvailable:\n{SKILLS.get_descriptions()}",
    "input_schema": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]},
}

TASK_CREATE_TOOL = {
    "name": "TaskCreate", "description": "Create a new task to track work.",
    "input_schema": {"type": "object", "properties": {
        "subject": {"type": "string"}, "description": {"type": "string"}, "activeForm": {"type": "string"},
    }, "required": ["subject", "description"]},
}

TASK_GET_TOOL = {
    "name": "TaskGet", "description": "Get task details by ID.",
    "input_schema": {"type": "object", "properties": {"taskId": {"type": "string"}}, "required": ["taskId"]},
}

TASK_UPDATE_TOOL = {
    "name": "TaskUpdate", "description": "Update a task: status, dependencies, owner.",
    "input_schema": {"type": "object", "properties": {
        "taskId": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]},
        "addBlockedBy": {"type": "array", "items": {"type": "string"}},
        "addBlocks": {"type": "array", "items": {"type": "string"}},
        "owner": {"type": "string"},
    }, "required": ["taskId"]},
}

TASK_LIST_TOOL = {
    "name": "TaskList", "description": "List all tasks with status and dependencies.",
    "input_schema": {"type": "object", "properties": {}},
}

TASK_OUTPUT_TOOL = {
    "name": "TaskOutput", "description": "Get output from a background task. block=true to wait for completion.",
    "input_schema": {"type": "object", "properties": {
        "task_id": {"type": "string"},
        "block": {"type": "boolean", "default": True},
        "timeout": {"type": "integer", "default": 30000},
    }, "required": ["task_id"]},
}

TASK_STOP_TOOL = {
    "name": "TaskStop", "description": "Stop a running background task.",
    "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
}

ALL_TOOLS = BASE_TOOLS + [
    SUBAGENT_TOOL, SKILL_TOOL,
    TASK_CREATE_TOOL, TASK_GET_TOOL, TASK_UPDATE_TOOL, TASK_LIST_TOOL,
    TASK_OUTPUT_TOOL, TASK_STOP_TOOL,
]


def get_tools_for_agent(agent_type: str) -> list:
    """Get tools for a one-shot subagent based on its type."""
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return BASE_TOOLS
    return [t for t in BASE_TOOLS if t["name"] in allowed]


# =============================================================================
# Tool Implementations
# =============================================================================

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str, background: bool = False) -> str:
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"

    if background:
        task_id = BG.run_in_background(lambda: _exec_bash(cmd), task_type="bash")
        return json.dumps({"task_id": task_id, "status": "running"})

    return _exec_bash(cmd)


def _exec_bash(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
        return ((r.stdout + r.stderr).strip() or "(no output)")[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        text = fp.read_text()
        if old_text not in text:
            return f"Error: Text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    content = SKILLS.get_skill_content(skill_name)
    if content is None:
        return f"Error: Unknown skill '{skill_name}'. Available: {', '.join(SKILLS.list_skills()) or 'none'}"
    return f'<skill-loaded name="{skill_name}">\n{content}\n</skill-loaded>\n\nFollow the instructions above.'


def run_subagent(description: str, prompt: str, agent_type: str,
                 background: bool = False) -> str:
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    if background:
        task_id = BG.run_in_background(
            lambda: _exec_subagent(description, prompt, agent_type),
            task_type="agent"
        )
        return json.dumps({"task_id": task_id, "status": "running"})

    return _exec_subagent(description, prompt, agent_type)


def _exec_subagent(description: str, prompt: str, agent_type: str) -> str:
    config = AGENT_TYPES[agent_type]
    sub_system = f"You are a {agent_type} subagent at {WORKDIR}.\n\n{config['prompt']}\n\nComplete the task and return a concise summary."
    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    while True:
        sub_messages = CTX.microcompact(sub_messages)
        if CTX.should_compact(sub_messages):
            sub_messages = CTX.auto_compact(sub_messages)

        response = client.messages.create(model=MODEL, system=sub_system, messages=sub_messages, tools=sub_tools, max_tokens=8000)
        if response.stop_reason != "tool_use":
            break

        results = []
        for tc in [b for b in response.content if b.type == "tool_use"]:
            output = execute_tool(tc.name, tc.input)
            output = CTX.handle_large_output(output)
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "(subagent returned no text)"


def run_task_create(subject: str, description: str = "", active_form: str = "") -> str:
    task = TASK_MGR.create(subject, description, active_form)
    return json.dumps({"id": task.id, "subject": task.subject, "status": task.status})


def run_task_get(task_id: str) -> str:
    task = TASK_MGR.get(task_id)
    if not task:
        return f"Error: Task {task_id} not found"
    return json.dumps(asdict(task), indent=2)


def run_task_update(task_id: str, **kwargs) -> str:
    if kwargs.get("status") == "deleted":
        if TASK_MGR.delete(task_id):
            return f"Task {task_id} deleted"
        return f"Error: Task {task_id} not found"
    task = TASK_MGR.update(task_id, **kwargs)
    if not task:
        return f"Error: Task {task_id} not found"
    return json.dumps({"id": task.id, "status": task.status, "blocked_by": task.blocked_by})


def run_task_list() -> str:
    tasks = TASK_MGR.list_all()
    if not tasks:
        return "No tasks."
    lines = []
    for t in tasks:
        icon = {"completed": "[x]", "in_progress": "[>]"}.get(t.status, "[ ]")
        blocked = f" (blocked by: {', '.join(t.blocked_by)})" if t.blocked_by else ""
        owner = f" @{t.owner}" if t.owner else ""
        lines.append(f"#{t.id}. {icon} {t.subject}{blocked}{owner}")
    return "\n".join(lines)


def execute_tool(name: str, args: dict) -> str:
    if name == "bash":
        return run_bash(args["command"], args.get("run_in_background", False))
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "Task":
        return run_subagent(
            args["description"], args["prompt"], args["agent_type"],
            args.get("run_in_background", False),
        )
    if name == "Skill":
        return run_skill(args["skill"])
    if name == "TaskCreate":
        return run_task_create(args["subject"], args.get("description", ""), args.get("activeForm", ""))
    if name == "TaskGet":
        return run_task_get(args["taskId"])
    if name == "TaskUpdate":
        kw = {k: v for k, v in args.items() if k != "taskId"}
        return run_task_update(args["taskId"], **kw)
    if name == "TaskList":
        return run_task_list()
    if name == "TaskOutput":
        result = BG.get_output(args["task_id"], args.get("block", True), args.get("timeout", 30000))
        return json.dumps(result)
    if name == "TaskStop":
        result = BG.stop_task(args["task_id"])
        return json.dumps(result)
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop - with background task notifications
# =============================================================================

def agent_loop(messages: list) -> list:
    while True:
        messages = CTX.microcompact(messages)
        if CTX.should_compact(messages):
            print("\n[Compressing context...]")
            messages = CTX.auto_compact(messages)
            print("[Context compressed.]\n")

        # Drain background task notifications and inject before API call
        notifications = BG.drain_notifications()
        if notifications:
            notif_text = "\n".join(
                f"<task-notification>\n"
                f"  <task-id>{n['task_id']}</task-id>\n"
                f"  <status>{n['status']}</status>\n"
                f"  Summary: {n['summary']}\n"
                f"</task-notification>"
                for n in notifications
            )
            if messages and messages[-1].get("role") == "user":
                content = messages[-1].get("content", "")
                if isinstance(content, str):
                    messages[-1]["content"] = content + "\n\n" + notif_text
                elif isinstance(content, list):
                    content.append({"type": "text", "text": notif_text})
            else:
                messages.append({"role": "user", "content": notif_text})

        response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=ALL_TOOLS, max_tokens=8000)

        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        results = []
        for tc in tool_calls:
            if tc.name == "Task":
                bg = tc.input.get("run_in_background", False)
                print(f"\n> Task{'(bg)' if bg else ''}: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            elif tc.name in ("TaskOutput", "TaskStop"):
                print(f"\n> {tc.name}: {tc.input.get('task_id', '')}")
            elif tc.name.startswith("Task"):
                print(f"\n> {tc.name}: {tc.input.get('subject', tc.input.get('taskId', ''))}")
            else:
                bg = tc.input.get("run_in_background", False) if tc.name == "bash" else False
                print(f"\n> {tc.name}{'(bg)' if bg else ''}")

            output = execute_tool(tc.name, tc.input)
            output = CTX.handle_large_output(output)

            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name != "Task" or tc.input.get("run_in_background"):
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  {preview}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================

def main():
    print(f"Mini Claude Code v7 (with Background Tasks) - {WORKDIR}")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print("Commands: /compact, /tasks, exit")
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        if user_input.strip() == "/compact":
            if history:
                print("[Manual compression...]")
                history = CTX.auto_compact(history)
                print("[Done.]\n")
            else:
                print("[Nothing to compress.]\n")
            continue

        if user_input.strip() == "/tasks":
            print(run_task_list())
            print()
            continue

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
