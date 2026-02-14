# v7: 后台任务与通知 Bus

**核心洞察：不等结果的 Agent，才能同时做多件事。**

v6 的 Tasks 系统解决了任务追踪问题。但无论是子代理还是 bash 命令，执行时主 Agent 必须等待：

```
主 Agent: Task("探索代码库") -> 等待... -> 收到结果 -> 继续
                                 ^
                            这段时间什么都做不了
```

## 解法：后台执行 + 通知

两个改变：

1. **后台执行**：子代理和 bash 可以在后台运行，主 Agent 继续工作
2. **通知 Bus**：后台任务完成时，通过通知告知主 Agent

```
主 Agent:
  Task(background) ──┐
  Bash(background) ──┼── 继续其他工作
  Task(background) ──┘
                          ← 通知: "任务 A 完成了"
                          ← 通知: "命令 B 完成了"
```

## BackgroundTask 数据结构

每个后台任务用 `BackgroundTask` 实例追踪，包含以下字段：

```python
@dataclass
class BackgroundTask:
    task_id: str               # 带类型前缀的唯一 ID（如 "b3f7c2"）
    task_type: str             # "bash" 或 "agent"
    thread: threading.Thread   # 执行工作的守护线程
    output: str                # 捕获的结果（完成后填充）
    status: str                # "running" | "completed" | "error" | "stopped"
    event: threading.Event     # 任务完成时触发，支持阻塞等待
```

`event` 字段是同步原语。`get_output(block=True)` 调用 `event.wait()` 来休眠直到后台线程发出完成信号，避免忙轮询。

## 后台任务类型

每种后台任务有不同的 ID 前缀，一看 ID 就知道类型：

| 类型 | 前缀 | 典型用途 |
|------|------|---------|
| local_bash | `b` | 运行测试、lint、构建 |
| local_agent | `a` | 探索代码、分析文件 |
| remote_agent | `r` | 远程会话 |
| in_process_teammate | `t` | Teammate 协作 (v8) |

ID 的生成格式是 `{前缀}{uuid4_hex[:6]}`，例如 `b3a9f1` 或 `a7c2d4`。前缀让日志和通知中的类型一目了然。

## 线程执行模型

后台任务运行在 Python 守护线程（`daemon=True`）中。执行包装器遵循以下模式：

```python
def wrapper():
    try:
        result = func()              # 执行实际工作
        bg_task.output = result
        bg_task.status = "completed"
    except Exception as e:
        bg_task.output = f"Error: {e}"
        bg_task.status = "error"      # 错误被捕获，不会传播
    finally:
        bg_task.event.set()           # 总是发出完成信号
        notifications.put({           # 总是推送通知
            "task_id": task_id,
            "status": bg_task.status,
            "summary": bg_task.output[:500],
        })
```

关键特性：
- **错误隔离**：异常被捕获并存储在 `output` 中，不会导致主 Agent 崩溃
- **保证通知**：`finally` 块确保无论任务成功还是失败，通知都会被推送
- **守护线程**：如果主进程退出，所有后台线程自动终止

## 触发后台执行

任何支持 `run_in_background` 参数的工具都可以后台运行：

```python
# 前台（阻塞）
Task(prompt="分析代码")               # 等待完成

# 后台（非阻塞）
Task(prompt="分析代码", run_in_background=True)
# -> 立即返回 {"task_id": "a3f7c2", "status": "running"}
```

后台启动后立即返回任务 ID，主 Agent 继续工作。

## 两个新工具

### TaskOutput：读取后台任务结果

```python
# 阻塞等待完成
TaskOutput(task_id="a3f7c2", block=True, timeout=30000)
# -> {"status": "completed", "output": "...分析结果..."}

# 非阻塞检查状态
TaskOutput(task_id="a3f7c2", block=False)
# -> {"status": "running", "output": "...当前输出..."}
```

`timeout` 参数（毫秒）防止无限阻塞。如果任务在超时时间内未完成，返回当前状态和部分输出。

### TaskStop：终止后台任务

```python
TaskStop(task_id="a3f7c2")
# -> {"task_id": "a3f7c2", "status": "stopped"}
```

`stop_task` 设置状态为 `"stopped"` 并触发事件。这是协作式停止——线程不会被强制杀死，但运行中的函数可以检查状态变化。对于已经通过 `subprocess.run` 启动的 bash 命令，进程会运行到完成（或自身超时）。

## 通知排空/注入循环

通知 Bus 基于 `queue.Queue` 实现。主 Agent 循环在每次 API 调用前执行**排空并注入**循环：

```python
# 1. 排空：从队列中拉取所有待处理通知
notifications = BG.drain_notifications()

# 2. 格式化：转换为 XML 块
notif_text = "\n".join(
    f"<task-notification>\n"
    f"  <task-id>{n['task_id']}</task-id>\n"
    f"  <status>{n['status']}</status>\n"
    f"  Summary: {n['summary']}\n"
    f"</task-notification>"
    for n in notifications
)

# 3. 注入：追加到最后一条用户消息（或创建新消息）
if messages[-1]["role"] == "user":
    messages[-1]["content"] += "\n\n" + notif_text
else:
    messages.append({"role": "user", "content": notif_text})
```

模型将通知视为会话上下文中的结构化 XML 块，然后决定是通过 `TaskOutput` 获取完整输出还是基于摘要继续工作。

## 通知格式

后台任务完成时，通知自动注入主 Agent 的下一轮对话：

```xml
<task-notification>
  <task-id>a3f7c2</task-id>
  <status>completed</status>
  Summary: Found 3 authentication-related files in src/auth/...
</task-notification>
```

`summary` 字段包含任务输出的前 500 个字符——足够模型判断是否需要获取完整结果。

## 典型流程

```
用户: "分析 src/ 和 tests/ 的代码质量"

主 Agent:
  1. Task(background, prompt="分析 src/")    -> task_id="a1c4e9"
  2. Task(background, prompt="分析 tests/")  -> task_id="a7b2d3"
  3. Bash(background, command="eslint src/")  -> task_id="b5e8f1"

  (三个任务并行执行)

  4. TaskOutput("a1c4e9", block=True)  -> 等待并获取结果
  5. <task-notification> b5e8f1 completed   (ESLint 在等待期间完成)
  6. TaskOutput("a7b2d3", block=True)  -> 获取第二个结果
  7. 综合三个结果给出报告
```

## 与 Tasks (v6) 的关系

两个系统互补：

| | Tasks (v6) | 后台任务 (v7) |
|-|-----------|---------------|
| 目的 | 规划和追踪 | 并行执行 |
| 粒度 | 高层目标 | 具体执行 |
| 生命周期 | 跨会话持久 | 单次会话 |
| 可见性 | 任务看板 | 通知流 |

Tasks 管理**做什么**，后台任务管理**怎么并行做**。

## 更深的洞察

> **从串行到并行。**

v6 的 Tasks 是看板——记录要做什么。v7 的后台任务是流水线——多条线同时运转。

通知 Bus 是关键胶水层：主 Agent 不需要轮询，任务完成时主动推送。"等待-执行-等待"的串行模式变成"发起-继续工作-收通知"的并行模式。

后台任务也为 v8 的 Teammate 奠定基础：Teammate 本质上是一种特殊的后台任务（`t` 前缀），复用同样的通知和输出基础设施。

---

**串行等待浪费时间，并行通知解放效率。**

[<< v6](./v6-Tasks系统.md) | [返回 README](../README_zh.md) | [v8 >>](./v8-Teammate机制.md)
