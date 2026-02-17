---
name: skill-creator
description: >
  Interactive guide for creating new skills (or updating existing skills) that extend Claude's
  capabilities with specialized knowledge, workflows, or tool integrations. Walks the user through
  use case definition, frontmatter generation, instruction writing, and validation. Use when user
  asks to create a new skill, build a skill, update an existing skill, says "skill-creator",
  "create a skill for", "build a skill that", or wants help designing skill structure and content.
  Do NOT use for general prompting questions or non-skill configuration tasks.
---

# Skill Creator

Guide for building effective, well-structured skills following Anthropic's official best practices.

## Core Concepts

A **skill** is a folder containing instructions (packaged as Markdown with YAML frontmatter) that teaches Claude how to handle specific tasks or workflows. Skills use **progressive disclosure**:

1. **First level (YAML frontmatter):** Always loaded in Claude's system prompt. Provides just enough info for Claude to know WHEN each skill should be used.
2. **Second level (SKILL.md body):** Loaded when Claude thinks the skill is relevant. Contains the full instructions and guidance.
3. **Third level (Linked files):** Additional files in `references/`, `scripts/`, `assets/` that Claude discovers and loads only as needed.

## Skill Creation Workflow

Follow these steps in order:

### Step 1: Define Use Cases

Before writing any code, identify 2-3 concrete use cases the skill should enable.

Ask yourself:
- What does a user want to accomplish?
- What multi-step workflows does this require?
- Which tools are needed (built-in or MCP)?
- What domain knowledge or best practices should be embedded?

Good use case definition:
```
Use Case: Project Sprint Planning
Trigger: User says "help me plan this sprint" or "create sprint tasks"
Steps:
1. Fetch current project status from Linear (via MCP)
2. Analyze team capacity
3. Suggest task prioritization
4. Create tasks in Linear with proper labels and estimates
Result: Fully planned sprint with tasks created
```

### Step 2: Plan Skill Contents

Analyze each use case to identify reusable resources:

- **Scripts** (`scripts/`): Code that gets rewritten repeatedly or needs deterministic reliability
- **References** (`references/`): Documentation Claude should consult while working
- **Assets** (`assets/`): Templates, fonts, icons used in output (not loaded into context)

### Step 3: Initialize the Skill

Run the init script to generate a template:

```bash
python3 scripts/init_skill.py <skill-name> --path <output-directory>
```

This creates the skill directory with a SKILL.md template and example resource directories.

### Step 4: Write the Skill

This is the critical step. Follow these rules carefully.

#### 4a: Write the Frontmatter

The YAML frontmatter is the most important part. It determines whether Claude loads your skill.

Consult `references/technical-rules.md` for all field requirements and naming conventions.

**Minimal required format:**
```yaml
---
name: your-skill-name
description: What it does. Use when user asks to [specific phrases].
---
```

**Description structure:** `[What it does] + [When to use it] + [Key capabilities]`

CRITICAL rules for the description:
- MUST include BOTH what the skill does AND when to use it (trigger conditions)
- Under 1024 characters
- No XML tags (< or >)
- Include specific tasks users might say
- Mention relevant file types if applicable

#### 4b: Write the Instructions (Body)

After the frontmatter, write Markdown instructions. Recommended structure:

```markdown
# Your Skill Name

## Instructions

### Step 1: [First Major Step]
Clear explanation of what happens.

### Step 2: [Next Step]
[...]

## Examples

### Example 1: [common scenario]
User says: "[example request]"
Actions:
1. [action]
2. [action]
Result: [expected outcome]

## Common Issues

### Error: [Common error message]
Cause: [Why it happens]
Solution: [How to fix]
```

**Best practices for instructions:**
- Be specific and actionable: `Run python scripts/validate.py --input {filename}` instead of `Validate the data before proceeding.`
- Use imperative/infinitive form
- Include error handling with specific solutions
- Reference bundled resources clearly: `Before writing queries, consult references/api-patterns.md for rate limiting guidance`
- Keep SKILL.md under 5,000 words; move detailed docs to `references/`

For detailed design patterns (sequential workflows, multi-MCP coordination, iterative refinement, context-aware tool selection, domain-specific intelligence), see `references/design-patterns.md`.

#### 4c: Apply Progressive Disclosure

Keep SKILL.md focused on core instructions. Move detailed documentation to `references/` and link to it.

Patterns:
- **High-level guide with references:** Core workflow in SKILL.md, details in reference files
- **Domain-specific organization:** One reference file per domain/variant
- **Conditional details:** Basic content inline, advanced content in separate files

IMPORTANT: All reference files should link directly from SKILL.md with clear descriptions of when to read them. Keep references one level deep (no nested references).

### Step 5: Validate and Package

Run validation:
```bash
python3 scripts/quick_validate.py <path/to/skill-folder>
```

Package for distribution:
```bash
python3 scripts/package_skill.py <path/to/skill-folder> [output-directory]
```

The packaging script validates automatically before creating the `.skill` file.

### Step 6: Test and Iterate

**Triggering tests:** Verify the skill loads at the right times.
- Should trigger on obvious tasks and paraphrased requests
- Should NOT trigger on unrelated topics

**Functional tests:** Verify correct outputs.
- Valid outputs generated
- API calls succeed (if applicable)
- Edge cases covered

**Iteration signals:**
- Undertriggering (skill doesn't load when it should): Add more detail and trigger phrases to description
- Overtriggering (skill loads for unrelated queries): Add negative triggers, be more specific
- Instructions not followed: Keep instructions concise, use bullet points, put critical instructions at the top

## Validation Checklist

Before considering a skill complete, verify:

- [ ] Folder named in kebab-case
- [ ] `SKILL.md` file exists (exact spelling, case-sensitive)
- [ ] YAML frontmatter has `---` delimiters
- [ ] `name` field: kebab-case, no spaces, no capitals
- [ ] `description` includes WHAT and WHEN
- [ ] No XML tags (< >) anywhere in frontmatter
- [ ] Instructions are clear and actionable
- [ ] Error handling included
- [ ] Examples provided
- [ ] References clearly linked
- [ ] No README.md inside skill folder (all docs go in SKILL.md or references/)
