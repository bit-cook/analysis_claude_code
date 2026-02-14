# v8: Teammate 机制

**核心洞察：子代理是派出去的员工，Teammate 是坐在旁边的同事。**

v3 的子代理是"分而治之"：主代理派任务，子代理执行，返回结果，销毁。

```
v3 子代理：
  主代理 -> "去探索代码库" -> 子代理
  主代理 <- "auth 在 src/auth/" <- 子代理
  (子代理消失，上下文销毁)
```

对于"前端和后端同时开发"这样的任务，子代理不够用：不能互相通信、不能共享进度、执行完就销毁。Teammate 解决的是**持续性协作**问题。

## Subagent vs Teammate

| 特性 | Subagent (v3) | Teammate (v8) |
|------|--------------|---------------|
| 生命周期 | 一次性 | 持久性（工作 <-> 空闲） |
| 通信 | 返回值（单向） | 消息协议（双向） |
| 并行 | 伪并行（等待返回） | 真并行（独立线程） |
| 任务管理 | 无 | 共享 Tasks (v6) |
| 适用场景 | 一次性任务 | 多模块长期协作 |

## 架构

```
Team Lead（主代理）
  ├── Teammate: frontend   （守护线程）
  ├── Teammate: backend    （守护线程）
  └── 共享:
        ├── .tasks/         <- 所有人看同一个看板
        └── .teams/         <- 每个 Teammate 的 JSONL 收件箱
```

每个 Teammate 作为守护线程运行，有自己的 Agent 循环和上下文窗口，独立执行压缩（v5）。

## JSONL 收件箱文件格式

每个 Teammate 在 `.teams/{team_name}/{name}_inbox.jsonl` 有专属收件箱文件。消息以 JSON 格式每行一条存储：

```json
{"type": "message", "sender": "lead", "content": "请先完成登录页面", "timestamp": 1709234567.89}
{"type": "broadcast", "sender": "backend", "content": "API schema 已定稿", "timestamp": 1709234590.12}
```

读取收件箱会消费所有消息并清空文件（读取即清空模式）。这防止了重复处理，同时保持格式简单且对并发写入友好。

## TEAMMATE_TOOLS vs ALL_TOOLS

Teammate 和 Team Lead 获得不同的工具集：

| 工具 | Team Lead | Teammate |
|------|-----------|----------|
| bash, read_file, write_file, edit_file | 有 | 有 |
| TaskCreate, TaskUpdate, TaskList | 有 | 有 |
| TaskGet | 有 | 无 |
| SendMessage | 有 | 有 |
| Task（生成子代理/Teammate） | 有 | 无 |
| Skill | 有 | 无 |
| TaskOutput, TaskStop | 有 | 无 |
| TeamCreate, TeamDelete | 有 | 无 |

Teammate 获得 `BASE_TOOLS + 任务 CRUD + SendMessage`——足以完成工作、更新共享看板、与同伴通信，但无法生成其他代理或管理团队本身。这确保 Team Lead 作为编排者的角色。

## 三个核心工具

```python
# TeamCreate: 创建团队
TeamCreate(name="my-project")

# SendMessage: 发消息给 Teammate
SendMessage(recipient="frontend", content="请先完成登录页面")

# TeamDelete: 解散团队
TeamDelete(name="my-project")
```

Teammate 通过 Task 工具生成，指定 `team_name` 即可：

```python
Task(prompt="负责前端开发", team_name="my-project", name="frontend")
# -> 生成持久 Teammate，而非一次性子代理
```

## Teammate 状态机

Teammate 的生命周期是持续的 `active -> idle -> active` 循环：

```
                    +---> active（运行 Agent 循环，使用工具）
                    |         |
                    |         v
   spawn -----> active   idle（每 2 秒轮询收件箱，持续 60 秒）
                              |
                    +---------+-- 新消息到达 -> active
                    |         +-- 发现未认领任务 -> active
                    |         +-- shutdown_request -> 退出
                    |         +-- 60 秒超时 -> 继续 idle 循环
                    |
                    +--- (下一轮回到 idle)
```

在 **active** 阶段，Teammate 运行正常的 Agent 循环（API 调用 -> 工具使用 -> API 调用）。当模型停止调用工具（`stop_reason != "tool_use"`）时，转入 **idle**。

在 **idle** 阶段，Teammate 每 2 秒轮询一次收件箱，持续最多 60 秒：
1. 如果**新消息**到达，注入上下文并返回 active
2. 如果在任务看板上发现**未认领任务**（status=pending，无 owner，无 blocker），自动认领并返回 active
3. 如果收到 **shutdown_request**，完全退出循环
4. 如果 60 秒内无事发生，重新开始 idle 轮询循环

## 广播的工作方式

广播不是独立的方法。它使用同一个 `send_message()` 函数，`msg_type="broadcast"`：

```python
# 在代码中，broadcast 遍历团队中的所有 Teammate
SendMessage(recipient="anyone", content="Schema 已定稿", type="broadcast", team_name="my-project")
```

内部实现中，管理器遍历团队中所有 Teammate（排除发送者），将消息追加到每个 Teammate 的 JSONL 收件箱文件中。对于广播，`recipient` 字段被忽略——消息发送给所有人。

## 任务认领的工作方式

Teammate 自主从共享任务看板认领工作：

```python
# 在 idle 阶段，Teammate 检查未认领任务
unclaimed = [t for t in TASK_MGR.list_all()
             if t.status == "pending" and not t.owner and not t.blocked_by]
if unclaimed:
    task = unclaimed[0]
    TASK_MGR.update(task.id, status="in_progress", owner=teammate.name)
```

认领遵循先到先得顺序（按任务 ID 排序）。`TaskManager.update()` 中的线程锁防止多个 Teammate 同时认领同一任务时的竞态条件。

## 关闭协议

关闭序列是一个请求-响应协议：

1. **Team Lead** 通过 `SendMessage(type="shutdown_request")` 发送关闭请求
2. 消息写入 Teammate 的 JSONL 收件箱
3. 在 Teammate 的下一次 **idle 轮询**中，读取到 `shutdown_request`
4. Teammate 设置 `status = "shutdown"` 并退出循环（`return`）
5. 由于线程是守护线程，无需 join

`TeamDelete` 同时向团队中所有 Teammate 发送 `shutdown_request`，然后从注册表中删除团队。

## 上下文压缩身份保持

当 Teammate 的上下文被压缩（v5 auto_compact）后，压缩后的摘要不会保留 Teammate 的身份。Teammate 循环会重新注入身份信息：

```python
if CTX.should_compact(sub_messages):
    sub_messages = CTX.auto_compact(sub_messages)
    identity = f"\n\nRemember: You are teammate '{teammate.name}' in team '{teammate.team_name}'."
    sub_messages[0]["content"] += identity
```

这确保模型即使在激进的上下文压缩后仍能保持其角色和团队上下文。

## 完整协作流程

```
用户: "把应用从 REST 迁移到 GraphQL"

Team Lead:
  1. TeamCreate("rest-to-graphql")
  2. TaskCreate("Analyze REST endpoints")          -> #1
  3. TaskCreate("Design GraphQL schema")           -> #2, blockedBy=#1
  4. TaskCreate("Implement resolvers")             -> #3, blockedBy=#2
  5. TaskCreate("Update frontend")                 -> #4, blockedBy=#3

  6. Task(name="analyst", team_name=..., prompt="分析 REST 端点")
  7. Task(name="backend", team_name=..., prompt="处理后端任务")
  8. Task(name="frontend", team_name=..., prompt="处理前端迁移")

analyst:   #1 完成 -> idle
backend:   #2 解锁 -> #2 完成 -> #3 完成 -> idle
frontend:  #4 解锁 -> #4 完成 -> idle

Team Lead: 所有人 idle -> shutdown -> "迁移完成。"
```

三个机制联动：
- **Tasks (v6)** 是共享看板——所有人看到同一个进度
- **压缩 (v5)** 让每个角色长时间工作——不受上下文限制
- **消息协议** 让角色间随时沟通——不需要通过主代理中转

## 对比

| 方面 | v3 (Subagent) | v8 (Teammate) |
|------|--------------|---------------|
| 模型 | 一次性函数调用 | 持久性工作线程 |
| 通信 | 返回值 | 消息协议 |
| 状态 | 无状态 | 有状态（idle/active） |
| 任务管理 | 无 | 共享 Tasks |
| 并行 | 伪并行 | 真并行 |

## 更深的洞察

> **从命令到协作。**

v3 的子代理是命令模式：主代理发号施令，子代理服从执行。v8 的 Teammate 是协作模式：Team Lead 分配方向，Teammate 自主工作、自主领活、互相沟通。

```
Subagent  -> 做一件事，回来汇报（实习生）
Teammate  -> 持续工作，自主领活（同事）
Team Lead -> 拆任务，管进度，审计划（经理）
```

Agent 系统的终极形态不是一个更聪明的模型，而是**一群能协作的模型**。

---

**一个 Agent 能力有限，一群 Agent 无所不能。**

[<< v7](./v7-后台任务与通知Bus.md) | [返回 README](../README_zh.md)
