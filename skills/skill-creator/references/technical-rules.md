# Technical Rules Reference

## File Structure

```
your-skill-name/
├── SKILL.md                # Required - main skill file
├── scripts/                # Optional - executable code (Python, Bash, etc.)
│   ├── process_data.py
│   └── validate.sh
├── references/             # Optional - documentation loaded as needed
│   ├── api-guide.md
│   └── examples/
└── assets/                 # Optional - templates, fonts, icons used in output
    └── report-template.md
```

## Critical Naming Rules

### SKILL.md Naming
- Must be exactly `SKILL.md` (case-sensitive)
- No variations accepted: SKILL.MD, skill.md, Skill.md all FAIL

### Skill Folder Naming
- Use kebab-case: `notion-project-setup`
- No spaces: `Notion Project Setup` FAILS
- No underscores: `notion_project_setup` FAILS
- No capitals: `NotionProjectSetup` FAILS

### No README.md
- Do NOT include README.md inside the skill folder
- All documentation goes in SKILL.md or references/
- When distributing via GitHub, the repo-level README is separate from the skill folder

## YAML Frontmatter Field Requirements

### name (required)
- kebab-case only
- No spaces or capitals
- Should match folder name
- Max 64 characters
- Cannot start/end with hyphen or contain consecutive hyphens
- Cannot include "claude" or "anthropic" (reserved)

### description (required)
- MUST include BOTH: what the skill does AND when to use it (trigger conditions)
- Under 1024 characters
- No XML tags (< or >)
- Include specific tasks users might say
- Mention file types if relevant

**Structure:** `[What it does] + [When to use it] + [Key capabilities]`

**Good examples:**

```yaml
# Good - specific and actionable
description: Analyzes Figma design files and generates developer handoff documentation. Use when user uploads .fig files, asks for "design specs", "component documentation", or "design-to-code handoff".

# Good - includes trigger phrases
description: Manages Linear project workflows including sprint planning, task creation, and status tracking. Use when user mentions "sprint", "Linear tasks", "project planning", or asks to "create tickets".

# Good - clear value proposition
description: End-to-end customer onboarding workflow for PayFlow. Handles account creation, payment setup, and subscription management. Use when user says "onboard new customer", "set up subscription", or "create PayFlow account".
```

**Bad examples:**

```yaml
# Too vague
description: Helps with projects.

# Missing triggers
description: Creates sophisticated multi-page documentation systems.

# Too technical, no user triggers
description: Implements the Project entity model with hierarchical relationships.
```

### license (optional)
- Use if making skill open source
- Common: MIT, Apache-2.0

### compatibility (optional)
- 1-500 characters
- Indicates environment requirements: intended product, required system packages, network access needs

### metadata (optional)
- Any custom key-value pairs
- Suggested: author, version, mcp-server
```yaml
metadata:
    author: ProjectHub
    version: 1.0.0
    mcp-server: projecthub
```

### allowed-tools (optional)
- List of tools the skill is allowed to use

## Security Restrictions

### Forbidden in frontmatter:
- XML angle brackets (< >)
- Skills with "claude" or "anthropic" in name (reserved)

**Why:** Frontmatter appears in Claude's system prompt. Malicious content could inject instructions.

## Resource Guidelines

### Scripts (`scripts/`)
- Include when the same code is rewritten repeatedly or deterministic reliability is needed
- Scripts may be executed without loading into context (token efficient)
- Test all scripts by running them before packaging

### References (`references/`)
- Include for documentation Claude should reference while working
- Keep SKILL.md lean; move detailed information here
- If files are large (>10k words), include grep search patterns in SKILL.md
- Structure longer files (>100 lines) with table of contents
- Avoid duplication between SKILL.md and references

### Assets (`assets/`)
- Files used in output, not loaded into context
- Templates, images, icons, boilerplate code, fonts
- Separate output resources from documentation

## Frontmatter Validation Checklist

1. Starts with `---` delimiter
2. Ends with `---` delimiter
3. Contains `name` field (kebab-case string)
4. Contains `description` field (string, < 1024 chars)
5. No XML tags in any field
6. No unknown fields (only: name, description, license, allowed-tools, metadata)
7. Valid YAML syntax (proper quoting, no unclosed quotes)
