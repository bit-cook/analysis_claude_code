#!/usr/bin/env python3
"""
v5_compression_agent.py - Mini Claude Code: Context Compression (~650 lines)

Core Philosophy: "Forgetting is a Feature"
==========================================
v0-v4 have an implicit assumption: conversation history can grow forever.
In reality, every model has a context window limit (~200K tokens).

A complex refactoring task might need 100+ tool calls. Without compression,
the agent will hit the wall and crash.

The Three-Layer Compression Model:
----------------------------------
    Layer 1: Micro-compact (every turn, silent)
        Replace old tool outputs with placeholders, keep recent 3.
        Like short-term memory decay.

    Layer 2: Auto-compact (near limit, ~93%)
        Summarize entire conversation, keep recent 5 messages.
        Like converting detail memory to concept memory.

    Layer 3: Manual compact (/compact command)
        User-triggered with custom focus.

Key Insight: Only compress WORKING memory, never delete ARCHIVE.
Full transcripts are always saved to disk.

Cache-Preserving Design:
    Summary goes into user message, NOT system prompt.
    This preserves the prompt cache prefix.

Usage:
    python v5_compression_agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# Configuration
# =============================================================================

# When using third-party endpoints (e.g. GLM), clear ANTHROPIC_AUTH_TOKEN
# to prevent the SDK from sending a conflicting authorization header.
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# ContextManager - The core addition in v5
# =============================================================================

class ContextManager:
    """
    Three-layer context compression to keep conversations within window limits.

    Human working memory is limited too - we don't remember every line of code
    we wrote, just "what we did, why, and current state". Compression mimics
    this cognitive pattern:
    - Micro-compact = short-term memory auto-decay
    - Auto-compact  = detail memory -> concept memory
    - Disk transcript = long-term memory archive
    """

    COMPACTABLE_TOOLS = {"bash", "read_file", "Grep", "Glob"}
    KEEP_RECENT = 3
    TOKEN_THRESHOLD = 0.93
    MAX_OUTPUT_TOKENS = 40000

    def __init__(self, max_context_tokens: int = 200000):
        self.max_context_tokens = max_context_tokens
        TRANSCRIPT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def microcompact(self, messages: list) -> list:
        """
        Layer 1: Replace old large tool outputs with placeholders.

        Keeps the tool call structure intact - the model still knows WHAT
        it called, just can't see the old output. It can re-read if needed.
        """
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

        # Keep only the most recent KEEP_RECENT, compact the rest
        to_compact = tool_result_indices[:-self.KEEP_RECENT] if len(tool_result_indices) > self.KEEP_RECENT else []

        for i, j, block in to_compact:
            content_str = block.get("content", "")
            if isinstance(content_str, str) and self.estimate_tokens(content_str) > 1000:
                block["content"] = "[Output compacted - re-read if needed]"

        return messages

    def should_compact(self, messages: list) -> bool:
        """Check if context is approaching the window limit."""
        total = sum(self.estimate_tokens(json.dumps(m, default=str)) for m in messages)
        return total > self.max_context_tokens * self.TOKEN_THRESHOLD

    def auto_compact(self, messages: list) -> list:
        """
        Layer 2: Summarize entire conversation, preserving recent context.

        1. Save full transcript to disk (never lose data)
        2. Call model to generate chronological summary
        3. Replace old messages with summary, keep recent 5
        """
        self.save_transcript(messages)

        conversation_text = self._messages_to_text(messages)

        summary_response = client.messages.create(
            model=MODEL,
            system="You are a conversation summarizer. Be concise but thorough.",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation chronologically. Include: goals, actions taken, decisions made, current state, and pending work.\n\n{conversation_text[:100000]}"
            }],
            max_tokens=2000,
        )

        summary = summary_response.content[0].text

        # Inject summary as user message (preserves system prompt cache)
        recent = messages[-5:] if len(messages) > 5 else messages[-2:]
        return [
            {"role": "user", "content": f"[Conversation compressed]\n\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context from the compressed conversation. Continuing work."},
            *recent
        ]

    def handle_large_output(self, output: str) -> str:
        """
        Handle oversized tool output: save to disk, return preview.
        """
        if self.estimate_tokens(output) <= self.MAX_OUTPUT_TOKENS:
            return output

        filename = f"output_{int(time.time())}.txt"
        path = TRANSCRIPT_DIR / filename
        path.write_text(output)

        preview = output[:2000]
        return f"Output too large ({self.estimate_tokens(output)} tokens). Saved to: {path}\n\nPreview:\n{preview}..."

    def save_transcript(self, messages: list):
        """Append full transcript to disk. The permanent archive."""
        path = TRANSCRIPT_DIR / "transcript.jsonl"
        with open(path, "a") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str) + "\n")

    def _find_tool_name(self, messages: list, tool_use_id: str) -> str:
        """Find tool name from a tool_use_id in message history."""
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
        """Convert messages to plain text for summarization."""
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
                            text = str(block.get("content", ""))[:200]
                            lines.append(f"[tool_result] {text}")
                        elif block.get("type") == "text":
                            lines.append(f"[{role}] {block.get('text', '')[:500]}")
                    elif hasattr(block, "text"):
                        lines.append(f"[{role}] {block.text[:500]}")
        return "\n".join(lines)


# Global context manager
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

        return {
            "name": metadata["name"],
            "description": metadata["description"],
            "body": body.strip(),
            "path": path,
            "dir": path.parent,
        }

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
        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.skills.items()
        )

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
            content += f"\n\n**Available resources in {skill['dir']}:**\n"
            content += "\n".join(f"- {r}" for r in resources)
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
    return "\n".join(
        f"- {name}: {cfg['description']}"
        for name, cfg in AGENT_TYPES.items()
    )


# =============================================================================
# TodoManager (from v2)
# =============================================================================

class TodoManager:
    """Task list manager with constraints (max 20 items, single in_progress)."""

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated = []
        in_progress = 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active = str(item.get("activeForm", "")).strip()
            if not content or not active:
                raise ValueError(f"Item {i}: content and activeForm required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status")
            if status == "in_progress":
                in_progress += 1
            validated.append({"content": content, "status": status, "activeForm": active})

        if in_progress > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated[:20]
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for t in self.items:
            mark = "[x]" if t["status"] == "completed" else \
                   "[>]" if t["status"] == "in_progress" else "[ ]"
            lines.append(f"{mark} {t['content']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)"


TODO = TodoManager()


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents available** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

Rules:
- Use Skill tool IMMEDIATELY when a task matches a skill description
- Use Task tool for subtasks needing focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Tool Definitions
# =============================================================================

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "TodoWrite",
        "description": "Update task list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "activeForm": {"type": "string"},
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]

TASK_TOOL = {
    "name": "Task",
    "description": f"Spawn a subagent for a focused subtask.\n\nAgent types:\n{get_agent_descriptions()}",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task description (3-5 words)"},
            "prompt": {"type": "string", "description": "Detailed instructions for the subagent"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys())},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

SKILL_TOOL = {
    "name": "Skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.

Available skills:
{SKILLS.get_descriptions()}

When to use: IMMEDIATELY when user task matches a skill description.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Name of the skill to load"}
        },
        "required": ["skill"],
    },
}

ALL_TOOLS = BASE_TOOLS + [TASK_TOOL, SKILL_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
    """Filter tools based on agent type."""
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


def run_bash(cmd: str) -> str:
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"
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


def run_todo(items: list) -> str:
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill_name: str) -> str:
    content = SKILLS.get_skill_content(skill_name)
    if content is None:
        available = ", ".join(SKILLS.list_skills()) or "none"
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"
    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


def run_task(description: str, prompt: str, agent_type: str) -> str:
    """Execute a subagent task with compression support."""
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]
    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    while True:
        # v5: Subagents also compress when needed
        sub_messages = CTX.microcompact(sub_messages)
        if CTX.should_compact(sub_messages):
            sub_messages = CTX.auto_compact(sub_messages)

        response = client.messages.create(
            model=MODEL, system=sub_system,
            messages=sub_messages, tools=sub_tools, max_tokens=8000,
        )

        if response.stop_reason != "tool_use":
            break

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        results = []
        for tc in tool_calls:
            tool_count += 1
            output = execute_tool(tc.name, tc.input)
            output = CTX.handle_large_output(output)
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})
            elapsed = time.time() - start
            sys.stdout.write(f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s")
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    elapsed = time.time() - start
    sys.stdout.write(f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n")

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "(subagent returned no text)"


def execute_tool(name: str, args: dict) -> str:
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"])
    if name == "Skill":
        return run_skill(args["skill"])
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop - with compression
# =============================================================================

INITIAL_REMINDER = """<reminder>
Use TodoWrite to plan and track multi-step tasks.
Only one task should be in_progress at a time.
</reminder>"""

NAG_REMINDER = """<reminder>
You haven't updated your todo list recently.
Use TodoWrite to track progress on multi-step tasks.
</reminder>"""


def agent_loop(messages: list) -> list:
    """
    Main agent loop with three-layer compression.

    Before each API call:
    1. Micro-compact old tool outputs (silent, every turn)
    2. Check if auto-compact needed (near context limit)
    3. Handle /compact command (user-triggered)
    """
    rounds_without_todo = 0

    while True:
        # v5: Apply compression before each API call
        messages = CTX.microcompact(messages)
        if CTX.should_compact(messages):
            print("\n[Compressing context...]")
            messages = CTX.auto_compact(messages)
            print("[Context compressed. Continuing...]\n")

        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=ALL_TOOLS, max_tokens=8000,
        )

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
        has_todo = False
        for tc in tool_calls:
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            if tc.name == "TodoWrite":
                has_todo = True

            output = execute_tool(tc.name, tc.input)
            # v5: Handle large outputs
            output = CTX.handle_large_output(output)

            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name != "Task":
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  {preview}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        # Nag reminder for todo usage
        if has_todo:
            rounds_without_todo = 0
        else:
            rounds_without_todo += 1
            if rounds_without_todo >= 3 and len(tool_calls) > 0:
                results.insert(0, {"type": "text", "text": NAG_REMINDER})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================

def main():
    print(f"Mini Claude Code v5 (with Compression) - {WORKDIR}")
    print(f"Skills: {', '.join(SKILLS.list_skills()) or 'none'}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print("Commands: /compact (manual compression), exit")
    print()

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # v5: Manual compact command
        if user_input.strip() == "/compact":
            if history:
                print("[Manual compression...]")
                history = CTX.auto_compact(history)
                print("[Done. Context compressed.]\n")
            else:
                print("[Nothing to compress.]\n")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
