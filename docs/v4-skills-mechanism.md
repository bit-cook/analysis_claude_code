# v4: Skills Mechanism

**Core insight: Skills are knowledge packages, not tools.**

## The Problem

v3 gave us subagents for task decomposition. But there's a deeper question: **How does the model know HOW to handle domain-specific tasks?**

- Processing PDFs? It needs to know `pdftotext` vs `PyMuPDF`
- Building MCP servers? It needs protocol specs and best practices
- Code review? It needs a systematic checklist

This knowledge isn't a tool—it's **expertise**. Skills solve this by letting the model load domain knowledge on-demand.

## Key Concepts

### Tools vs Skills

| Concept | What it is | Example |
|---------|------------|---------|
| **Tool** | What model CAN do | bash, read_file, write_file |
| **Skill** | How model KNOWS to do | PDF processing, MCP building |

Tools are capabilities. Skills are knowledge.

### Knowledge Externalization: From Training to Editing

Traditional way to modify model behavior requires training: GPU clusters + data + ML expertise. Skills change everything:

```
Modify model behavior = Edit SKILL.md = Edit text file = Anyone can do it
```

| Layer | Modification | Effective Time | Cost |
|-------|--------------|----------------|------|
| Model Parameters | Training/Fine-tuning | Hours to Days | $10K-$1M+ |
| Context Window | API call | Instant | ~$0.01/call |
| **Skill Library** | **Edit SKILL.md** | **Next trigger** | **Free** |

This is a paradigm shift from "training AI" to "educating AI".

### Progressive Disclosure

```
Layer 1: Metadata (always loaded)     ~100 tokens/skill
         name + description

Layer 2: SKILL.md body (on trigger)   ~2000 tokens
         Detailed instructions

Layer 3: Resources (as needed)        Unlimited
         scripts/, references/, assets/
```

Context stays lean while allowing arbitrary depth of knowledge.

### SKILL.md Standard

```
skills/
├── pdf/
│   └── SKILL.md          # Required
├── mcp-builder/
│   ├── SKILL.md
│   └── references/       # Optional
└── code-review/
    ├── SKILL.md
    └── scripts/          # Optional
```

**Format**: YAML frontmatter + Markdown body

```md
---
name: pdf
description: Process PDF files. Use when reading, creating, or merging PDFs.
---

# PDF Processing Skill

## Reading PDFs
Use pdftotext for quick extraction:
pdftotext input.pdf -
```

## Implementation (~100 lines added)

### SkillLoader Class

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        """Parse YAML frontmatter + Markdown body."""
        content = path.read_text()
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        # Returns {name, description, body, path, dir}

    def get_descriptions(self) -> str:
        """Generate metadata for system prompt."""
        return "\n".join(f"- {name}: {skill['description']}"
                        for name, skill in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        """Get full content for context injection."""
        return f"# Skill: {name}\n\n{skill['body']}"
```

### Skill Tool

```python
SKILL_TOOL = {
    "name": "Skill",
    "description": "Load a skill to gain specialized knowledge.",
    "input_schema": {
        "properties": {"skill": {"type": "string"}},
        "required": ["skill"]
    }
}
```

### Message Injection (Cache-Preserving)

The key insight: Skill content goes into **tool_result** (part of user message), NOT system prompt:

```python
def run_skill(skill_name: str) -> str:
    content = SKILLS.get_skill_content(skill_name)
    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above."""
```

**Why this matters**:
- Skill content is **appended to the end** as new message
- Everything before (system prompt + all previous messages) is cached and reused
- Only the newly appended skill content needs computation — **entire prefix hits cache**

> **Treat context as append-only log, not editable document.**

## Comparison with Production

| Mechanism | Claude Code / Kode | v4 |
|-----------|-------------------|-----|
| Format | SKILL.md (YAML + MD) | Same |
| Triggering | Auto + Skill tool | Skill tool only |
| Injection | newMessages (user message) | tool_result (user message) |
| Caching | Append to end, entire prefix cached | Append to end, entire prefix cached |

## Philosophy

> **Knowledge as a first-class citizen**

Skills acknowledge that **domain knowledge is itself a resource** that needs explicit management.

1. **Separate metadata from content**: Description is index, body is content
2. **Load on demand**: Context window is precious cognitive resource
3. **Standardized format**: Write once, use in any compatible agent
4. **Inject, don't return**: Skills change cognition, not just provide data

The essence of knowledge externalization is **turning implicit knowledge into explicit documents**. Developers "teach" models new skills in natural language, Git manages and shares knowledge with version control and rollback.

## Series Summary

| Version | Theme | Lines Added | Key Insight |
|---------|-------|-------------|-------------|
| v1 | Model as Agent | ~200 | Model is 80%, code is just the loop |
| v2 | Structured Planning | ~100 | Todo makes plans visible |
| v3 | Divide and Conquer | ~150 | Subagents isolate context |
| **v4** | **Domain Expert** | **~100** | **Skills inject expertise** |

---

**Tools let models act. Skills let models know how.**

[← v3](./v3-subagent-mechanism.md) | [Back to README](../README.md) | [v5 →](./v5-context-compression.md)
