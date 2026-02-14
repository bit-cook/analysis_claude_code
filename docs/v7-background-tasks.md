# v7: Background Tasks & Notification Bus

**Core insight: An agent that doesn't wait for results can do multiple things at once.**

v6 Tasks solved task tracking. But whether it's a subagent or a bash command, the main agent must wait during execution:

```
Main Agent: Task("explore codebase") -> waiting... -> got result -> continue
                                         ^
                                    can't do anything during this time
```

## The Solution: Background Execution + Notifications

Two changes:

1. **Background execution**: subagents and bash can run in the background while the main agent continues
2. **Notification bus**: when background tasks complete, notifications are pushed to the main agent

```
Main Agent:
  Task(background) ---\
  Bash(background) ----+--- continues other work
  Task(background) ---/
                          <- notification: "Task A completed"
                          <- notification: "Command B completed"
```

## The BackgroundTask Dataclass

Each background task is tracked as a `BackgroundTask` instance with the following fields:

```python
@dataclass
class BackgroundTask:
    task_id: str               # Unique ID with type prefix (e.g., "b3f7c2")
    task_type: str             # "bash" or "agent"
    thread: threading.Thread   # The daemon thread executing the work
    output: str                # Captured result (populated on completion)
    status: str                # "running" | "completed" | "error" | "stopped"
    event: threading.Event     # Signaled when task finishes, enables blocking waits
```

The `event` field is the synchronization primitive. `get_output(block=True)` calls `event.wait()` to sleep until the background thread signals completion, avoiding busy-polling.

## Background Task Types

Each background task type has a distinct ID prefix -- you know the type at a glance:

| Type | Prefix | Typical Use |
|------|--------|-------------|
| local_bash | `b` | Run tests, lint, build |
| local_agent | `a` | Explore code, analyze files |
| remote_agent | `r` | Remote sessions |
| in_process_teammate | `t` | Teammate collaboration (v8) |

IDs are generated as `{prefix}{uuid4_hex[:6]}`, e.g. `b3a9f1` or `a7c2d4`. The prefix gives immediate type visibility in logs and notifications.

## Thread Execution Model

Background tasks run in Python daemon threads (`daemon=True`). The execution wrapper follows this pattern:

```python
def wrapper():
    try:
        result = func()              # Execute the actual work
        bg_task.output = result
        bg_task.status = "completed"
    except Exception as e:
        bg_task.output = f"Error: {e}"
        bg_task.status = "error"      # Errors are captured, not propagated
    finally:
        bg_task.event.set()           # Always signal completion
        notifications.put({           # Always push notification
            "task_id": task_id,
            "status": bg_task.status,
            "summary": bg_task.output[:500],
        })
```

Key properties:
- **Error containment**: exceptions are caught and stored in `output`, never crashing the main agent
- **Guaranteed notification**: the `finally` block ensures a notification is always pushed, whether the task succeeds or fails
- **Daemon threads**: if the main process exits, all background threads are automatically terminated

## Triggering Background Execution

Any tool supporting `run_in_background` can run in the background:

```python
# Foreground (blocking)
Task(prompt="Analyze code")               # waits for completion

# Background (non-blocking)
Task(prompt="Analyze code", run_in_background=True)
# -> returns immediately: {"task_id": "a3f7c2", "status": "running"}
```

After background launch, the task ID returns immediately and the main agent continues working.

## Two New Tools

### TaskOutput: Read Background Task Results

```python
# Block and wait for completion
TaskOutput(task_id="a3f7c2", block=True, timeout=30000)
# -> {"status": "completed", "output": "...analysis results..."}

# Non-blocking status check
TaskOutput(task_id="a3f7c2", block=False)
# -> {"status": "running", "output": "...current output..."}
```

The `timeout` parameter (in milliseconds) prevents indefinite blocking. If the task does not complete within the timeout, the current status and partial output are returned.

### TaskStop: Terminate a Background Task

```python
TaskStop(task_id="a3f7c2")
# -> {"task_id": "a3f7c2", "status": "stopped"}
```

`stop_task` sets the status to `"stopped"` and signals the event. This is a cooperative stop -- the thread is not forcibly killed, but the status change can be checked by the running function. For bash commands already spawned via `subprocess.run`, the process will run to completion (or its own timeout).

## Notification Drain/Inject Cycle

The notification bus is implemented as a `queue.Queue`. The main agent loop performs a **drain-and-inject** cycle before every API call:

```python
# 1. Drain: pull all pending notifications from the queue
notifications = BG.drain_notifications()

# 2. Format: convert to XML blocks
notif_text = "\n".join(
    f"<task-notification>\n"
    f"  <task-id>{n['task_id']}</task-id>\n"
    f"  <status>{n['status']}</status>\n"
    f"  Summary: {n['summary']}\n"
    f"</task-notification>"
    for n in notifications
)

# 3. Inject: append to the last user message (or create a new one)
if messages[-1]["role"] == "user":
    messages[-1]["content"] += "\n\n" + notif_text
else:
    messages.append({"role": "user", "content": notif_text})
```

The model sees notifications as structured XML blocks within its conversation context. It can then decide to retrieve full output via `TaskOutput` or continue with the summary.

## Notification Format

When a background task completes, a notification is automatically injected into the main agent's next turn:

```xml
<task-notification>
  <task-id>a3f7c2</task-id>
  <status>completed</status>
  Summary: Found 3 authentication-related files in src/auth/...
</task-notification>
```

The `summary` field contains the first 500 characters of the task output -- enough for the model to decide whether to fetch the full result.

## Typical Flow

```
User: "Analyze code quality in src/ and tests/"

Main Agent:
  1. Task(background, prompt="Analyze src/")    -> task_id="a1c4e9"
  2. Task(background, prompt="Analyze tests/")  -> task_id="a7b2d3"
  3. Bash(background, command="eslint src/")     -> task_id="b5e8f1"

  (three tasks running in parallel)

  4. TaskOutput("a1c4e9", block=True)  -> wait and get result
  5. <task-notification> b5e8f1 completed  (ESLint finished while waiting)
  6. TaskOutput("a7b2d3", block=True)  -> get second result
  7. Synthesize all three results into a report
```

## Relationship with Tasks (v6)

The two systems are complementary:

| | Tasks (v6) | Background Tasks (v7) |
|-|-----------|----------------------|
| Purpose | Planning and tracking | Parallel execution |
| Granularity | High-level goals | Concrete execution |
| Lifecycle | Persistent across sessions | Single session |
| Visibility | Task board | Notification stream |

Tasks manage **what to do**. Background tasks manage **how to do it in parallel**.

## The Deeper Insight

> **From serial to parallel.**

v6 Tasks is a kanban board -- recording what needs to be done. v7 background tasks are a pipeline -- multiple lines running simultaneously.

The notification bus is the key glue layer: the main agent doesn't poll, completions push themselves. The "wait-execute-wait" serial pattern becomes the "launch-keep working-get notified" parallel pattern.

Background tasks also lay the groundwork for v8 Teammates: a Teammate is essentially a special background task (prefix `t`), reusing the same notification and output infrastructure.

---

**Serial waiting wastes time. Parallel notification unlocks efficiency.**

[<< v6](./v6-tasks-system.md) | [Back to README](../README.md) | [v8 >>](./v8-teammate-mechanism.md)
