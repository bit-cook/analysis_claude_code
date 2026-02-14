# v5: Context Compression

**Core insight: Forgetting is a feature, not a bug.**

v0-v4 share an implicit assumption: conversation history can grow forever. In practice, it can't.

## The Problem

```
200K token context window:
  [System prompt]       ~2K tokens
  [CLAUDE.md]           ~3K tokens
  [Tool definitions]    ~8K tokens
  [Conversation]        keeps growing...
  [Tool call #50]       -> approaching 180K tokens
  [Tool call #60]       -> exceeds 200K, request fails
```

A complex refactoring task can take 100+ tool calls. Without compression, the agent hits the wall.

## Three-Layer Compression

Not one technique, but three progressive layers:

| Layer | Trigger | Action | User Awareness |
|-------|---------|--------|---------------|
| Microcompact | Every turn (auto) | Replace old tool outputs | Invisible |
| Auto-compact | Near context limit | Summarize entire conversation | User sees notice |
| Manual compact | `/compact` command | Custom compression per user | User-initiated |

## Microcompact: Silent Cleanup

After each turn, replace old large tool outputs with placeholders, keeping only recent ones:

```python
COMPACTABLE_TOOLS = {"Bash", "Read", "Grep", "Glob"}
KEEP_RECENT = 3

def microcompact(messages):
    """Replace old large tool results with placeholders."""
    tool_results = find_tool_results(messages, COMPACTABLE_TOOLS)

    for result in tool_results[:-KEEP_RECENT]:
        if estimate_tokens(result) > 1000:
            result["content"] = "[Output compacted - re-read if needed]"

    return messages
```

Key: only the **content** is cleared. The tool call structure stays intact. The model still knows what it called, just can't see old output. Re-read if needed.

## Auto-Compact: Full Summary

Triggered when context reaches ~93% of the window limit:

```python
def auto_compact(messages):
    # 1. Save full transcript to disk (never lost)
    save_transcript(messages)

    # 2. Use model to generate summary
    summary = call_api("Summarize this conversation chronologically: "
                       "goals, actions, decisions, current state...")

    # 3. Replace old messages with summary, keep recent turns
    return [
        {"role": "user", "content": f"[Conversation compressed]\n{summary}"},
        *messages[-5:]  # Keep recent conversation
    ]
```

**Key design**: the summary is injected into conversation history (user message), not into the system prompt. This keeps the system prompt's cache intact.

## Large Output Demotion

When a single tool output is too large, save to disk and return a preview:

```python
def handle_tool_output(output):
    if estimate_tokens(output) > 40000:
        path = save_to_disk(output)
        return f"Output too large. Saved to: {path}\nPreview:\n{output[:2000]}..."
    return output
```

## Compression Timeline

```
Each conversation turn:
  1. User input
  2. Microcompact: clean old tool outputs (silent)
  3. Check context size:
     - Normal: continue
     - Near limit: trigger auto-compact -> generate summary -> resume
     - At limit: block new requests
  4. Model response + tool execution
  5. Check tool output size (demote if too large)
  6. Back to step 1
```

## Subagents Compress Too

v3 subagents have their own context windows, and run compression independently:

```python
def run_subagent(prompt, agent_type):
    sub_messages = [{"role": "user", "content": prompt}]

    while True:
        if should_compact(sub_messages):
            sub_messages = auto_compact(sub_messages)

        response = call_api(sub_messages)
        if response.stop_reason != "tool_use":
            break
        # ...

    return extract_final_text(response)
```

Disk persistence from compression lays the groundwork for later mechanisms: the Tasks system (v6) and multi-agent collaboration (v8) store data on disk, unaffected by compression.

## Comparison

| Aspect | v4 and before (no compression) | v5 (three-layer) |
|--------|-------------------------------|-------------------|
| Max conversation length | Limited by context window | Theoretically unlimited |
| Long task reliability | Crashes on overflow | Graceful degradation |
| History data | All in memory | Disk persistence + in-memory summary |
| Recovery | None | Resume from summary or transcript |

## The Deeper Insight

> **Human working memory is limited too.**

We don't remember every line of code we wrote. We remember "what was done, why, and current state." Compression mirrors this cognitive pattern:

- Microcompact = short-term memory decay
- Auto-compact = shifting from detail memory to concept memory
- Disk transcript = retrievable long-term memory

The full record is always on disk. Compression only affects working memory, not the archive.

---

**Context is finite, work is infinite. Compression keeps the agent going.**

[<< v4](./v4-skills-mechanism.md) | [Back to README](../README.md) | [v6 >>](./v6-tasks-system.md)
