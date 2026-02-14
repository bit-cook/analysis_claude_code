# v4: Skills 机制

**核心洞察：Skills 是知识包，不是工具。**

## 问题背景

v3 给了我们子代理来分解任务。但还有一个更深的问题：**模型怎么知道如何处理特定领域的任务？**

- 处理 PDF？需要知道用 `pdftotext` 还是 `PyMuPDF`
- 构建 MCP 服务器？需要知道协议规范和最佳实践
- 代码审查？需要一套系统的检查清单

这些知识不是工具——是**专业技能**。Skills 通过让模型按需加载领域知识来解决这个问题。

## 核心概念

### 工具 vs 技能

| 概念 | 是什么 | 例子 |
|------|--------|------|
| **Tool** | 模型能**做**什么 | bash, read_file, write_file |
| **Skill** | 模型**知道怎么做** | PDF 处理、MCP 构建 |

工具是能力，技能是知识。

### 知识外化：从训练到编辑

传统方式修改模型行为需要训练：GPU 集群 + 数据 + ML 专业知识。Skills 改变了这一切：

```
修改模型行为 = 编辑 SKILL.md = 编辑文本文件 = 任何人都可以做
```

| 层级 | 修改方式 | 生效时间 | 成本 |
|------|----------|----------|------|
| Model Parameters | 训练/微调 | 数小时-数天 | $10K-$1M+ |
| Context Window | API 调用 | 即时 | ~$0.01/次 |
| **Skill Library** | **编辑 SKILL.md** | **下次触发** | **免费** |

### 渐进式披露

```
Layer 1: 元数据 (始终加载)     ~100 tokens/skill
         name + description

Layer 2: SKILL.md 主体 (触发时)   ~2000 tokens
         详细指南

Layer 3: 资源文件 (按需)        无限制
         scripts/, references/, assets/
```

上下文保持轻量，同时允许任意深度的知识。

### SKILL.md 标准

```
skills/
├── pdf/
│   └── SKILL.md          # 必需
├── mcp-builder/
│   ├── SKILL.md
│   └── references/       # 可选
└── code-review/
    ├── SKILL.md
    └── scripts/          # 可选
```

**格式**：YAML 前置 + Markdown 正文

```md
---
name: pdf
description: 处理 PDF 文件。用于读取、创建或合并 PDF。
---

# PDF 处理技能

## 读取 PDF
使用 pdftotext 快速提取：
pdftotext input.pdf -
```

## 实现（约 100 行新增）

### SkillLoader 类

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        """解析 YAML 前置 + Markdown 正文"""
        content = path.read_text()
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        # 返回 {name, description, body, path, dir}

    def get_descriptions(self) -> str:
        """生成系统提示词的元数据"""
        return "\n".join(f"- {name}: {skill['description']}"
                        for name, skill in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        """获取完整内容用于上下文注入"""
        return f"# Skill: {name}\n\n{skill['body']}"
```

### Skill 工具

```python
SKILL_TOOL = {
    "name": "Skill",
    "description": "加载技能获取专业知识。",
    "input_schema": {
        "properties": {"skill": {"type": "string"}},
        "required": ["skill"]
    }
}
```

### 消息注入（保持缓存）

关键洞察：Skill 内容进入 **tool_result**（user message 的一部分），而不是 system prompt：

```python
def run_skill(skill_name: str) -> str:
    content = SKILLS.get_skill_content(skill_name)
    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above."""
```

**为什么这很重要**：
- Skill 内容作为新消息**追加到末尾**
- 之前的所有内容（system prompt + 历史消息）都被缓存复用
- 只有新追加的 skill 内容需要计算，**整个前缀都命中缓存**

> **把上下文当作只追加日志，而非可编辑文档。**

## 与生产版本对比

| 机制 | Claude Code / Kode | v4 |
|------|-------------------|-----|
| 格式 | SKILL.md (YAML + MD) | 相同 |
| 触发 | 自动 + Skill 工具 | 仅 Skill 工具 |
| 注入 | newMessages (user message) | tool_result (user message) |
| 缓存机制 | 追加到末尾，前缀全部缓存 | 追加到末尾，前缀全部缓存 |

## 设计哲学

> **知识作为一等公民**

Skills 承认：**领域知识本身就是一种资源**，需要被显式管理。

1. **分离元数据与内容**：description 是索引，body 是内容
2. **按需加载**：上下文窗口是宝贵的认知资源
3. **标准化格式**：写一次，在任何兼容的 Agent 上使用
4. **注入而非返回**：Skills 改变认知，不只是提供数据

知识外化的本质是**把隐式知识变成显式文档**。这是从"训练 AI"到"教育 AI"的范式转变。

## 系列总结

| 版本 | 主题 | 新增行数 | 核心洞察 |
|------|------|----------|----------|
| v1 | Model as Agent | ~200 | 模型是 80%，代码只是循环 |
| v2 | 结构化规划 | ~100 | Todo 让计划可见 |
| v3 | 分而治之 | ~150 | 子代理隔离上下文 |
| **v4** | **领域专家** | **~100** | **Skills 注入专业知识** |

---

**工具让模型能做事，技能让模型知道怎么做。**

[<< v3](./v3-子代理机制.md) | [返回 README](../README_zh.md) | [v5 >>](./v5-上下文压缩.md)
