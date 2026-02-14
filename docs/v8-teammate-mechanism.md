# v8: Teammate Mechanism

**Core insight: Subagents are dispatched workers. Teammates are colleagues sitting next to you.**

v3 subagents are "divide and conquer": the main agent dispatches a task, the subagent executes, returns a result, and is destroyed.

```
v3 Subagent:
  Main Agent -> "explore the codebase" -> Subagent
  Main Agent <- "auth is in src/auth/" <- Subagent
  (subagent destroyed, context gone)
```

For tasks like "develop frontend and backend simultaneously," subagents aren't enough: they can't communicate with each other, can't share progress, and are destroyed after execution. Teammates solve the **sustained collaboration** problem.

## Subagent vs Teammate

| Feature | Subagent (v3) | Teammate (v8) |
|---------|--------------|---------------|
| Lifecycle | One-shot | Persistent (active <-> idle) |
| Communication | Return value (one-way) | Message protocol (two-way) |
| Parallelism | Pseudo-parallel (blocks on return) | True parallel (independent threads) |
| Task management | None | Shared Tasks (v6) |
| Use case | One-off tasks | Multi-module long-term collaboration |

## Architecture

```
Team Lead (main agent)
  |-- Teammate: frontend   (daemon thread)
  |-- Teammate: backend    (daemon thread)
  +-- Shared:
        |-- .tasks/         <- everyone sees the same board
        +-- .teams/         <- JSONL inbox files per teammate
```

Each Teammate runs as a daemon thread with its own agent loop, its own context window, and runs compression (v5) independently.

## JSONL Inbox File Format

Each teammate has a dedicated inbox file at `.teams/{team_name}/{name}_inbox.jsonl`. Messages are stored one per line in JSON format:

```json
{"type": "message", "sender": "lead", "content": "Please finish the login page first", "timestamp": 1709234567.89}
{"type": "broadcast", "sender": "backend", "content": "API schema is finalized", "timestamp": 1709234590.12}
```

Reading the inbox consumes all messages and clears the file (read-and-clear pattern). This prevents duplicate processing while keeping the format simple and append-friendly for concurrent writers.

## TEAMMATE_TOOLS vs ALL_TOOLS

Teammates and the Team Lead receive different toolsets:

| Tool | Team Lead | Teammate |
|------|-----------|----------|
| bash, read_file, write_file, edit_file | Yes | Yes |
| TaskCreate, TaskUpdate, TaskList | Yes | Yes |
| TaskGet | Yes | No |
| SendMessage | Yes | Yes |
| Task (spawn subagents/teammates) | Yes | No |
| Skill | Yes | No |
| TaskOutput, TaskStop | Yes | No |
| TeamCreate, TeamDelete | Yes | No |

Teammates get `BASE_TOOLS + task CRUD + SendMessage` -- enough to do work, update the shared board, and communicate with peers, but not enough to spawn other agents or manage the team itself. This enforces the Team Lead as the orchestrator.

## Three Core Tools

```python
# TeamCreate: create a team
TeamCreate(name="my-project")

# SendMessage: send a message to a teammate
SendMessage(recipient="frontend", content="Please finish the login page first")

# TeamDelete: disband the team
TeamDelete(name="my-project")
```

Teammates are spawned via the Task tool with a `team_name` parameter:

```python
Task(prompt="Handle frontend development", team_name="my-project", name="frontend")
# -> spawns a persistent Teammate, not a one-shot subagent
```

## Teammate State Machine

The teammate lifecycle is a continuous `active -> idle -> active` cycle:

```
                    +---> active (running agent loop, using tools)
                    |         |
                    |         v
   spawn -----> active   idle (polling inbox every 2s for 60s)
                              |
                    +---------+-- new message arrives -> active
                    |         +-- unclaimed task found -> active
                    |         +-- shutdown_request -> exit
                    |         +-- 60s timeout -> continue idle loop
                    |
                    +--- (back to idle on next cycle)
```

In the **active** phase, the teammate runs a normal agent loop (API call -> tool use -> API call). When the model stops calling tools (`stop_reason != "tool_use"`), the teammate transitions to **idle**.

In the **idle** phase, the teammate polls its inbox every 2 seconds for up to 60 seconds:
1. If a **new message** arrives, inject it into context and return to active
2. If an **unclaimed task** is found on the task board (status=pending, no owner, no blockers), auto-claim it with `TaskUpdate(owner=name)` and return to active
3. If a **shutdown_request** arrives, exit the loop entirely
4. If nothing happens for 60 seconds, restart the idle polling cycle

## How Broadcast Works

Broadcasting is NOT a separate method. It uses the same `send_message()` function with `msg_type="broadcast"`:

```python
# In the code, broadcast iterates over all teammates in the team
SendMessage(recipient="anyone", content="Schema is finalized", type="broadcast", team_name="my-project")
```

Internally, the manager iterates through all teammates in the team (excluding the sender) and appends the message to each teammate's JSONL inbox file. The `recipient` field is ignored for broadcasts -- the message goes to everyone.

## How Task Claiming Works

Teammates autonomously claim work from the shared task board:

```python
# During idle phase, teammate checks for unclaimed tasks
unclaimed = [t for t in TASK_MGR.list_all()
             if t.status == "pending" and not t.owner and not t.blocked_by]
if unclaimed:
    task = unclaimed[0]
    TASK_MGR.update(task.id, status="in_progress", owner=teammate.name)
```

The claiming follows first-come-first-served order (sorted by task ID). The thread lock in `TaskManager.update()` prevents race conditions when multiple teammates try to claim the same task.

## Shutdown Protocol

The shutdown sequence is a request-response protocol:

1. **Team Lead** sends `shutdown_request` via `SendMessage(type="shutdown_request")`
2. The message is written to the teammate's JSONL inbox
3. During the teammate's next **idle poll**, it reads the `shutdown_request`
4. The teammate sets `status = "shutdown"` and exits its loop (`return`)
5. Since the thread is a daemon thread, no join is needed

`TeamDelete` sends `shutdown_request` to all teammates in the team simultaneously, then removes the team from the registry.

## Context Compression Identity Preservation

When a teammate's context is compressed (v5 auto_compact), the compressed summary does not preserve who the teammate is. The teammate loop re-injects identity:

```python
if CTX.should_compact(sub_messages):
    sub_messages = CTX.auto_compact(sub_messages)
    identity = f"\n\nRemember: You are teammate '{teammate.name}' in team '{teammate.team_name}'."
    sub_messages[0]["content"] += identity
```

This ensures the model retains its role and team context even after aggressive context compression.

## Full Collaboration Flow

```
User: "Migrate the app from REST to GraphQL"

Team Lead:
  1. TeamCreate("rest-to-graphql")
  2. TaskCreate("Analyze REST endpoints")          -> #1
  3. TaskCreate("Design GraphQL schema")           -> #2, blockedBy=#1
  4. TaskCreate("Implement resolvers")             -> #3, blockedBy=#2
  5. TaskCreate("Update frontend")                 -> #4, blockedBy=#3

  6. Task(name="analyst", team_name=..., prompt="Analyze REST endpoints")
  7. Task(name="backend", team_name=..., prompt="Handle backend tasks")
  8. Task(name="frontend", team_name=..., prompt="Handle frontend migration")

analyst:   #1 done -> idle
backend:   #2 unblocked -> #2 done -> #3 done -> idle
frontend:  #4 unblocked -> #4 done -> idle

Team Lead: all idle -> shutdown -> "Migration complete."
```

Three mechanisms working together:
- **Tasks (v6)** is the shared board -- everyone sees the same progress
- **Compression (v5)** lets each role work long -- no context limit
- **Message protocol** lets roles communicate freely -- no routing through the main agent

## Comparison

| Aspect | v3 (Subagent) | v8 (Teammate) |
|--------|--------------|---------------|
| Model | One-shot function call | Persistent worker thread |
| Communication | Return value | Message protocol |
| State | Stateless | Stateful (idle/active) |
| Task management | None | Shared Tasks |
| Parallelism | Pseudo-parallel | True parallel |

## The Deeper Insight

> **From command to collaboration.**

v3 subagents follow a command pattern: the main agent gives orders, subagents obey. v8 Teammates follow a collaboration pattern: the Team Lead sets direction, Teammates work autonomously, claim tasks on their own, communicate with each other.

```
Subagent  -> do one thing, report back  (intern)
Teammate  -> work continuously, self-assign tasks  (colleague)
Team Lead -> break down work, manage progress, review plans  (manager)
```

The ultimate form of an agent system is not a smarter model, but **a group of models that can collaborate**.

---

**One agent has limits. A team of agents has none.**

[<< v7](./v7-background-tasks.md) | [Back to README](../README.md)
