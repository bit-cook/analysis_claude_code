# v6: Tasks System

**Core insight: From personal sticky notes to team kanban.**

v2 introduced TodoWrite to solve "the model forgets its plan." But with compression (v5) and subagents (v3), TodoWrite's limitations are exposed.

## The Problem

```
TodoWrite limitations:
  1. Write-only with overwrite (sends entire list each time)
  2. No persistence (lost after compression)
  3. No ownership (who is working on this?)
  4. No dependencies (A must complete before B)
  5. No concurrency safety (two agents writing = data loss)
```

v5 compression clears in-memory todos. Subagents can't share tasks. The Tasks system redesigns task management from scratch.

## TodoWrite vs Tasks

| Feature | TodoWrite (v2) | Tasks (v6) |
|---------|---------------|------------|
| Operations | Overwrite | CRUD (create/read/update/delete) |
| Persistence | Memory only (lost on compact) | Disk files (survives compact) |
| Concurrency | Unsafe | File locks |
| Dependencies | None | blocks / blockedBy |
| Ownership | None | Agent name |
| Multi-agent | Not supported | Native support |

## Data Model

```python
@dataclass
class Task:
    id: str              # Auto-increment ID
    subject: str         # Imperative title: "Fix auth bug"
    description: str     # Detailed description
    status: str = "pending"  # pending | in_progress | completed
    active_form: str = ""    # Present tense: "Fixing auth bug"
    owner: str = ""          # Responsible agent
    blocks: list = []        # Tasks blocked by this one
    blocked_by: list = []    # Prerequisites blocking this task
```

Why each field matters:

| Field | Reason |
|-------|--------|
| `id` | CRUD needs a unique identifier |
| `owner` | Multi-agent: who is doing what |
| `blocks/blockedBy` | Dependency graph for task orchestration |
| `description` | Another agent can understand the task |

## Four Tools

```python
# TaskCreate: create a task
task_create("Fix auth bug", description="...", active_form="Fixing auth bug")
# -> {"id": "1", "subject": "Fix auth bug"}

# TaskGet: read details
task_get("1")
# -> {id, subject, description, status, blocks, blockedBy}

# TaskUpdate: update status, dependencies, owner
task_update("1", status="in_progress")  # auto-assigns owner
task_update("2", addBlockedBy=["1"])    # 2 depends on 1

# TaskList: list all tasks
task_list()
# -> [{id, subject, status, owner, blockedBy}, ...]
```

## Dependency Graph

```
TaskCreate: "Set up database"       -> #1
TaskCreate: "Write API endpoints"   -> #2
TaskCreate: "Write tests"           -> #3

TaskUpdate: id=2, addBlockedBy=["1"]     # API depends on database
TaskUpdate: id=3, addBlockedBy=["1","2"] # Tests depend on both
```

Rendered view:

```
#1. [>] Set up database          (in_progress)
#2. [ ] Write API endpoints      blocked by: #1
#3. [ ] Write tests              blocked by: #1, #2
```

When #1 completes, #2's blockedBy is automatically cleared and becomes executable.

## Persistence

```python
def save_task(task):
    """File-level locking for concurrency safety."""
    path = f"tasks/{task.id}.json"
    with FileLock(path + ".lock"):
        with open(path, 'w') as f:
            json.dump(task.to_dict(), f)
```

Why files instead of a database?
- One file per task = fine-grained locking
- Subagents may run in separate processes
- JSON files are human-readable, easy to debug

## Working with Compression (v5)

Tasks persist on disk, unaffected by compression:

```
Before compact: [100 turns of conversation] + [5 tasks on disk]
After compact:  [summary + recent 5 turns]  + [5 tasks on disk]  <- tasks intact
```

TodoWrite couldn't do this -- its tasks lived only in message history, gone after compression.

## Feature Gate

The two systems are mutually exclusive via feature gate:

```python
def is_tasks_enabled():
    return get_feature_flag("tasks_v2")

# TodoWrite available only when Tasks is disabled
# TaskCreate/Get/Update/List available only when Tasks is enabled
```

Enabling Tasks automatically migrates old TodoWrite data.

## The Deeper Insight

> **From personal notes to team kanban.**

TodoWrite is a sticky note -- one person uses it, then discards it. Tasks is a project board -- multi-party collaboration, state transitions, dependency tracking.

This is a **paradigm shift in collaboration**:
- TodoWrite: the model's self-discipline tool (v2 philosophy: constraints enable)
- Tasks: a multi-agent coordination protocol (v6 philosophy: collaboration enables)

When agents evolve from individual to collective, task management must evolve from "checklist" to "system."

---

**A checklist keeps one agent organized. A task system keeps a team in order.**

[<< v5](./v5-context-compression.md) | [Back to README](../README.md) | [v7 >>](./v7-background-tasks.md)
