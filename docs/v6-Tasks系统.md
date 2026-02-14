# v6: Tasks 系统

**核心洞察：从个人便利贴到团队看板。**

v2 引入的 TodoWrite 解决了"模型忘记计划"的问题。但现在我们有了压缩（v5）和子代理（v3），TodoWrite 的局限暴露了。

## 问题

```
TodoWrite 的问题：
  1. 写入只有覆盖模式（每次发送完整列表）
  2. 没有持久化（压缩后 todo 丢失）
  3. 没有所有者（谁在做这个任务？）
  4. 没有依赖关系（A 必须在 B 之前完成）
  5. 没有并发安全（两个代理同时写 = 数据丢失）
```

v5 的压缩会清除内存中的 todo。子代理之间无法共享任务。Tasks 系统从根本上重新设计了任务管理。

## TodoWrite vs Tasks

| 特性 | TodoWrite (v2) | Tasks (v6) |
|------|---------------|------------|
| 操作方式 | 覆盖写入 | CRUD（创建/读取/更新/删除） |
| 持久化 | 仅内存（压缩后丢失） | 磁盘文件（压缩后存活） |
| 并发 | 不安全 | 文件锁 |
| 依赖 | 无 | blocks / blockedBy |
| 所有者 | 无 | agent name |
| 多代理 | 不支持 | 原生支持 |

## 数据模型

```python
@dataclass
class Task:
    id: str              # 自增 ID
    subject: str         # 祈使句标题: "Fix auth bug"
    description: str     # 详细描述
    status: str = "pending"  # pending | in_progress | completed
    active_form: str = ""    # 进行时态: "Fixing auth bug"
    owner: str = ""          # 负责的代理
    blocks: list = []        # 被此任务阻塞的任务
    blocked_by: list = []    # 阻塞此任务的前置任务
```

为什么每个字段都需要：

| 字段 | 原因 |
|------|------|
| `id` | CRUD 需要唯一标识 |
| `owner` | 多代理时标识谁在做 |
| `blocks/blockedBy` | 任务编排的依赖图 |
| `description` | 另一个代理也能理解任务 |

## 四个工具

```python
# TaskCreate: 创建任务
task_create("Fix auth bug", description="...", active_form="Fixing auth bug")
# -> {"id": "1", "subject": "Fix auth bug"}

# TaskGet: 读取详情
task_get("1")
# -> {id, subject, description, status, blocks, blockedBy}

# TaskUpdate: 更新状态、依赖、所有者
task_update("1", status="in_progress")  # 自动分配 owner
task_update("2", addBlockedBy=["1"])    # 2 依赖 1

# TaskList: 列出所有任务
task_list()
# -> [{id, subject, status, owner, blockedBy}, ...]
```

## 依赖图

```
TaskCreate: "Set up database"       -> #1
TaskCreate: "Write API endpoints"   -> #2
TaskCreate: "Write tests"           -> #3

TaskUpdate: id=2, addBlockedBy=["1"]     # API 依赖数据库
TaskUpdate: id=3, addBlockedBy=["1","2"] # 测试依赖两者
```

渲染效果：

```
#1. [>] Set up database          (in_progress)
#2. [ ] Write API endpoints      blocked by: #1
#3. [ ] Write tests              blocked by: #1, #2
```

当 #1 完成后，#2 的 blockedBy 自动清除，变为可执行。

## 持久化

```python
def save_task(task):
    """文件级锁保证并发安全"""
    path = f"tasks/{task.id}.json"
    with FileLock(path + ".lock"):
        with open(path, 'w') as f:
            json.dump(task.to_dict(), f)
```

为什么用文件而不是数据库？
- 每个任务一个文件 = 细粒度锁
- 子代理可能在不同进程中
- JSON 文件人类可读，方便调试

## 与压缩的协作 (v5)

Tasks 持久化在磁盘上，压缩时不会丢失：

```
压缩前:  [100 轮对话] + [5 个任务在磁盘上]
压缩后:  [摘要 + 最近 5 轮] + [5 个任务在磁盘上]  <- 任务完整保留
```

这是 TodoWrite 无法做到的——TodoWrite 的任务只在消息历史中，压缩后就没了。

## Feature Gate

两个系统通过 feature gate 互斥：

```python
def is_tasks_enabled():
    return get_feature_flag("tasks_v2")

# TodoWrite 只在 Tasks 未启用时可用
# TaskCreate/Get/Update/List 只在 Tasks 启用时可用
```

启用 Tasks 时自动迁移旧的 TodoWrite 数据。

## 更深的洞察

> **从个人笔记到团队看板。**

TodoWrite 像是便利贴——一个人用，用完就扔。Tasks 像是项目看板——多人协作，有状态流转，有依赖追踪。

这是**协作范式的转变**：
- TodoWrite: 模型的自我约束工具（v2 的哲学：约束赋能）
- Tasks: 多代理的协调协议（v6 的哲学：协作赋能）

当 Agent 从单体变成群体，任务管理必须从"清单"进化为"系统"。

---

**清单让一个 Agent 有条理，任务系统让一群 Agent 有秩序。**

[<< v5](./v5-上下文压缩.md) | [返回 README](../README_zh.md) | [v7 >>](./v7-后台任务与通知Bus.md)
