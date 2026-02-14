# Learn Claude Code - Bash 就是 Agent 的一切

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/shareAI-lab/learn-claude-code/actions/workflows/test.yml/badge.svg)](https://github.com/shareAI-lab/learn-claude-code/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> **声明**: 这是 [shareAI Lab](https://github.com/shareAI-lab) 的独立教育项目，与 Anthropic 无关，未获其认可或赞助。"Claude Code" 是 Anthropic 的商标。

**从零开始构建你自己的 AI Agent。**

[English](./README.md) | [Japanese / 日本語](./README_ja.md)

---

## 为什么有这个仓库？

这个仓库源于我们对 Claude Code 的敬佩 - **我们认为它是世界上最优秀的 AI 编程代理**。最初，我们试图通过行为观察和推测来逆向分析它的设计。然而，我们当时发布的分析内容充斥着不准确的信息、缺乏依据的猜测和技术错误。我们在此向 Claude Code 团队以及所有被这些内容误导的朋友深表歉意。

过去半年，在不断构建和迭代 Agent 系统的过程中，我们对 **"什么才是真正的 AI Agent"** 有了全新的认知。希望能把这些心得分享给大家。之前的推测性内容已全部移除，现已替换为原创教学材料。

---

> 兼容 **[Kode CLI](https://github.com/shareAI-lab/Kode)**、**Claude Code**、**Cursor**，以及任何支持 [Agent Skills Spec](https://agentskills.io/specification) 的 Agent。

<img height="400" alt="demo" src="https://github.com/user-attachments/assets/0e1e31f8-064f-4908-92ce-121e2eb8d453" />

## 你将学到什么

完成本教程后，你将理解：

- **Agent 循环** - 所有 AI 编程代理背后那个令人惊讶的简单模式
- **工具设计** - 如何让 AI 模型能够与真实世界交互
- **显式规划** - 使用约束让 AI 行为可预测
- **上下文管理** - 通过子代理隔离保持代理记忆干净
- **知识注入** - 按需加载领域专业知识，无需重新训练
- **上下文压缩** - 让 Agent 突破上下文窗口限制持续工作
- **任务系统** - 从个人便签到团队看板
- **并行执行** - 后台任务与通知驱动的工作流
- **多 Agent 协作** - 持久化队友并行工作

## 学习路径

```
从这里开始
    |
    v
[v0: Bash Agent] -----> "一个工具就够了"
    |                    16-50 行
    v
[v1: Basic Agent] ----> "完整的 Agent 模式"
    |                    4 个工具，~200 行
    v
[v2: Todo Agent] -----> "让计划显式化"
    |                    +TodoManager，~300 行
    v
[v3: Subagent] -------> "分而治之"
    |                    +Task 工具，~450 行
    v
[v4: Skills Agent] ---> "按需领域专业"
    |                    +Skill 工具，~550 行
    v
[v5: Compression] ----> "永不遗忘，永续工作"
    |                    +ContextManager，~650 行
    v
[v6: Tasks Agent] ----> "从便利贴到看板"
    |                    +TaskManager，~750 行
    v
[v7: Background] -----> "不等结果，继续干活"
    |                    +BackgroundManager，~850 行
    v
[v8: Teammate] -------> "一群 Agent 无所不能"
                         +TeammateManager，~1000 行
```

**推荐学习方式：**
1. 先阅读并运行 v0 - 理解核心循环
2. 对比 v0 和 v1 - 看工具如何演进
3. 学习 v2 的规划模式
4. 探索 v3 的复杂任务分解
5. 掌握 v4 构建可扩展的 Agent
6. 学习 v5 的上下文管理与压缩
7. 探索 v6 的持久化任务追踪
8. 理解 v7 的并行后台执行
9. 掌握 v8 的多 Agent 团队协作

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 ANTHROPIC_API_KEY

# 运行任意版本
python v0_bash_agent.py      # 极简版（从这里开始！）
python v1_basic_agent.py     # 核心 Agent 循环
python v2_todo_agent.py      # + Todo 规划
python v3_subagent.py           # + 子代理
python v4_skills_agent.py       # + Skills
python v5_compression_agent.py  # + 上下文压缩
python v6_tasks_agent.py        # + 任务系统
python v7_background_agent.py   # + 后台任务
python v8_teammate_agent.py     # + 团队协作
```

## 核心模式

每个 Agent 都只是这个循环：

```python
while True:
    response = model(messages, tools)
    if response.stop_reason != "tool_use":
        return response.text
    results = execute(response.tool_calls)
    messages.append(results)
```

就这样。模型持续调用工具直到完成。其他一切都是精化。

## 版本对比

| 版本 | 行数 | 工具 | 核心新增 | 关键洞察 |
|------|------|------|---------|---------|
| [v0](./v0_bash_agent.py) | ~50 | bash | 递归子代理 | 一个工具就够了 |
| [v1](./v1_basic_agent.py) | ~200 | bash, read, write, edit | 核心循环 | 模型即代理 |
| [v2](./v2_todo_agent.py) | ~300 | +TodoWrite | 显式规划 | 约束赋能复杂性 |
| [v3](./v3_subagent.py) | ~450 | +Task | 上下文隔离 | 干净上下文 = 更好结果 |
| [v4](./v4_skills_agent.py) | ~550 | +Skill | 知识加载 | 专业无需重训 |
| [v5](./v5_compression_agent.py) | ~650 | +ContextManager | 三层压缩 | 遗忘成就无限工作 |
| [v6](./v6_tasks_agent.py) | ~750 | +TaskCreate/Get/Update/List | 持久化任务 | 便利贴到看板 |
| [v7](./v7_background_agent.py) | ~850 | +TaskOutput/TaskStop | 后台执行 | 串行到并行 |
| [v8](./v8_teammate_agent.py) | ~1000 | +TeamCreate/SendMessage/TeamDelete | 持久化队友 | 指令到协作 |

## 子机制导航

每个版本引入一个核心类，但真正的学习在于子机制。此表帮助你定位具体概念：

| 子机制 | 版本 | 关键代码 | 学什么 |
|--------|------|---------|--------|
| **Agent 循环** | v0-v1 | `agent_loop()` | `while tool_use` 循环模式 |
| **工具分发** | v1 | `process_tool_call()` | tool_use 块如何映射到函数 |
| **显式规划** | v2 | `TodoManager` | 单 `in_progress` 约束、system reminder |
| **上下文隔离** | v3 | `run_subagent()` | 每个子代理独立消息列表 |
| **工具过滤** | v3 | `AGENT_TYPES` | Explore 代理只获得只读工具 |
| **Skill 注入** | v4 | `SkillLoader` | 内容前置到 system prompt |
| **微压缩** | v5 | `ContextManager.microcompact()` | 旧工具输出替换为占位符 |
| **自动压缩** | v5 | `ContextManager.auto_compact()` | 93% 阈值触发 API 摘要 |
| **大输出处理** | v5 | `ContextManager.handle_large_output()` | >40K token 存盘，返回预览 |
| **记录持久化** | v5 | `ContextManager.save_transcript()` | 完整历史追加到 `.jsonl` |
| **任务 CRUD** | v6 | `TaskManager` | create/get/update/list + JSON 持久化 |
| **依赖图** | v6 | `addBlocks/addBlockedBy` | 完成时自动解锁下游任务 |
| **后台执行** | v7 | `BackgroundManager.run_in_background()` | 线程执行，立即返回 task_id |
| **ID 前缀约定** | v7 | `_PREFIXES` | `b`=bash, `a`=agent（v8 增加 `t`=teammate） |
| **通知总线** | v7 | `drain_notifications()` | 每次 API 调用前清空队列 |
| **通知注入** | v7 | `<task-notification>` XML | 注入到最后一条用户消息 |
| **Teammate 生命周期** | v8 | `_teammate_loop()` | active -> 工作 -> idle -> 检查邮箱 -> active |
| **文件邮箱** | v8 | `send_message()/check_inbox()` | JSONL 格式，每个 Teammate 独立文件 |
| **消息协议** | v8 | `MESSAGE_TYPES` | 5 种：message, broadcast, shutdown_req/resp, plan_approval |
| **工具权限** | v8 | `TEAMMATE_TOOLS` | Teammate 获得 8 个工具（无 TeamCreate/Delete） |
| **任务认领** | v8 | `teammate_loop` | 空闲 Teammate 自动认领未分配任务 |
| **身份保持** | v8 | `auto_compact` + identity | 压缩后重新注入 Teammate 名称/角色 |

## 文件结构

```
learn-claude-code/
├── v0_bash_agent.py       # ~50 行: 1 个工具，递归子代理
├── v0_bash_agent_mini.py  # ~16 行: 极限压缩
├── v1_basic_agent.py      # ~200 行: 4 个工具，核心循环
├── v2_todo_agent.py       # ~300 行: + TodoManager
├── v3_subagent.py         # ~450 行: + Task 工具，代理注册表
├── v4_skills_agent.py     # ~550 行: + Skill 工具，SkillLoader
├── v5_compression_agent.py # ~650 行: + ContextManager，三层压缩
├── v6_tasks_agent.py      # ~750 行: + TaskManager，依赖图 CRUD
├── v7_background_agent.py # ~850 行: + BackgroundManager，并行执行
├── v8_teammate_agent.py   # ~1000 行: + TeammateManager，团队协作
├── skills/                # 示例 Skills（pdf, code-review, mcp-builder, agent-builder）
├── docs/                  # 技术文档（中英日三语）
├── articles/              # 公众号风格文章
└── tests/                 # 单元测试、特性测试和集成测试
```

## 深入阅读

### 技术文档 (docs/)

- [v0: Bash 就是一切](./docs/v0-Bash就是一切.md)
- [v1: 模型即代理](./docs/v1-模型即代理.md)
- [v2: 结构化规划](./docs/v2-结构化规划.md)
- [v3: 子代理机制](./docs/v3-子代理机制.md)
- [v4: Skills 机制](./docs/v4-Skills机制.md)
- [v5: 上下文压缩](./docs/v5-上下文压缩.md)
- [v6: Tasks 系统](./docs/v6-Tasks系统.md)
- [v7: 后台任务与通知 Bus](./docs/v7-后台任务与通知Bus.md)
- [v8: Teammate 机制](./docs/v8-Teammate机制.md)

### 原创文章 (articles/)

- [v0文章](./articles/v0文章.md) - Bash 就是一切
- [v1文章](./articles/v1文章.md) - 价值 3000 万美金的 400 行代码
- [v2文章](./articles/v2文章.md) - 用 Todo 实现自我约束
- [v3文章](./articles/v3文章.md) - 子代理机制
- [v4文章](./articles/v4文章.md) - Skills 机制
- [v5文章](./articles/v5文章.md) - 三层上下文压缩
- [v6文章](./articles/v6文章.md) - Tasks 系统
- [v7文章](./articles/v7文章.md) - 后台任务与通知 Bus
- [v8文章](./articles/v8文章.md) - Teammate 机制
- [上下文缓存经济学](./articles/上下文缓存经济学.md) - Agent 开发者必知的成本优化

## 使用 Skills 系统

### 内置示例 Skills

| Skill | 用途 |
|-------|------|
| [agent-builder](./skills/agent-builder/) | 元技能：如何构建 Agent |
| [code-review](./skills/code-review/) | 系统化代码审查方法论 |
| [pdf](./skills/pdf/) | PDF 操作模式 |
| [mcp-builder](./skills/mcp-builder/) | MCP 服务器开发 |

### 脚手架生成新 Agent

```bash
# 使用 agent-builder skill 创建新项目
python skills/agent-builder/scripts/init_agent.py my-agent

# 指定复杂度级别
python skills/agent-builder/scripts/init_agent.py my-agent --level 0  # 极简
python skills/agent-builder/scripts/init_agent.py my-agent --level 1  # 4 工具
```

### 生产环境安装 Skills

```bash
# Kode CLI（推荐）
kode plugins install https://github.com/shareAI-lab/shareAI-skills

# Claude Code
claude plugins install https://github.com/shareAI-lab/shareAI-skills
```

## 配置说明

```bash
# .env 文件选项
ANTHROPIC_API_KEY=sk-ant-xxx      # 必需：你的 API key
ANTHROPIC_BASE_URL=https://...    # 可选：API 代理
MODEL_ID=claude-sonnet-4-5-20250929  # 可选：模型选择
```

## 相关项目

| 仓库 | 说明 |
|------|------|
| [Kode](https://github.com/shareAI-lab/Kode) | 生产就绪的开源 Agent CLI |
| [shareAI-skills](https://github.com/shareAI-lab/shareAI-skills) | 生产 Skills 集合 |
| [Agent Skills Spec](https://agentskills.io/specification) | 官方规范 |

## 设计哲学

> **模型是 80%，代码是 20%。**

Kode 和 Claude Code 等现代 Agent 能工作，不是因为巧妙的工程，而是因为模型被训练成了 Agent。我们的工作就是给它工具，然后闪开。

## 贡献

欢迎贡献！请随时提交 issues 和 pull requests。

- 在 `skills/` 中添加新的示例 skills
- 在 `docs/` 中改进文档
- 通过 [Issues](https://github.com/shareAI-lab/learn-claude-code/issues) 报告 bug 或建议功能

## License

MIT

---

**模型即代理。这就是全部秘密。**

[@baicai003](https://x.com/baicai003) | [shareAI Lab](https://github.com/shareAI-lab)
