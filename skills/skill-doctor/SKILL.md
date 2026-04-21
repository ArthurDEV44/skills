---
model: opus
name: skill-doctor
description: "Audit a Claude Code skill or workflow for structural quality, idempotency, and best-practice compliance. Runs 30+ binary checks grounded in Anthropic's official skill authoring docs, Agent Patterns research, and production patterns. Produces a severity-weighted report (FAIL/WARN/NOTE). Use when the user says 'audit this skill', 'check my skill', 'skill-doctor', 'review my workflow', or provides a path to a skill directory. Do NOT use for codebase audits (use meta-audit) or PRD reviews (use meta-review-prd)."
argument-hint: "<path-to-skill-directory>"
allowed-tools: Read, Grep, Glob, Bash, Agent
---

# skill-doctor — Structural Audit for Claude Code Skills

## Purpose

Single-pass structural audit of a Claude Code skill or workflow. Every check is a **binary structural test** (presence/absence, count, pattern match) — no semantic judgments. Same input produces same output on every run.

**Grounded in:**
- [Claude Code Skills docs](https://code.claude.com/docs/en/skills) — official SKILL.md spec (name, description, allowed-tools, argument-hint, disable-model-invocation)
- [Equipping agents with Agent Skills](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills) — 3-layer progressive disclosure architecture
- [Agent Patterns — loop prevention](https://www.agentpatterns.tech/en/failures/infinite-loop)
- [Anthropic Engineering — Demystifying Evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Skill-creator technical rules (`~/.claude/skills/skill-creator/references/technical-rules.md`)

## Execution Flow

```
$ARGUMENTS -> <skill-directory-path>
     |
     v
[Phase 1] INTAKE       — Locate skill, read SKILL.md, inventory files
     |
     v
[Phase 2] CHECK        — Run 30+ binary checks across 7 categories
     |
     v
[Phase 3] REPORT       — Severity-weighted report + JSON
```

## Phase 1 — INTAKE

Print: `[Phase 1/3] INTAKE`

**1a. Locate the skill:**

```
path = $ARGUMENTS
```

If path doesn't end with `/SKILL.md`, look for `SKILL.md` inside the directory. If not found, list available skills in `~/.claude/skills/` and stop.

**1b. Read and inventory:**

1. Read `SKILL.md` completely
2. Parse YAML frontmatter (between `---` delimiters)
3. Glob `references/**/*` and `scripts/**/*`
4. Count SKILL.md body lines (after frontmatter)
5. Extract all internal links (`references/*.md`, `scripts/*.py`, etc.)
6. Check for broken links (linked files that don't exist)

**1c. Read shared conventions:**

```
Read: ~/.claude/skills/_shared/three-tier-constraints.md (if referenced)
Read: ~/.claude/skills/_shared/agent-boundaries.md (if referenced)
```

---

## Phase 2 — CHECK

Print: `[Phase 2/3] CHECK — {N} checks across 7 categories`

Run ALL checks from [check-catalog.md](references/check-catalog.md). Every check is binary: `PASS | FAIL | WARN | NOTE`.

> **Idempotency enforcement:** Each check has a table row (summary) and a procedure (authoritative). When they diverge, execute the **procedure exactly as written**. Use only the closed sets, exact substrings, and regex patterns specified in each procedure. Do NOT expand pattern lists, add synonyms, interpret proximity loosely, or apply semantic judgment. If a test says "N/A (skip)" when the trigger condition is absent, skip it — do not force a verdict.

**Verdict rules:**
- `FAIL` — Structural violation with a concrete fix. Binary test (exists/missing, valid/invalid).
- `WARN` — Fixable issue with a concrete action. Binary test.
- `NOTE` — Observation. Informational only.

**Cap:** Max 3 findings per check. Beyond 3, emit top 3 + count.

**Categories:**

### Category 1 — File Structure [FAIL-capable]
Checks: SKILL.md exists, naming conventions, no README.md, directory structure.

### Category 2 — Frontmatter Quality [FAIL/WARN-capable]
Checks: required fields, name format, description WHAT+WHEN, no XML tags, length limits.

### Category 3 — Progressive Disclosure [WARN-capable]
Checks: body line count, reference files linked, no deep nesting, ToC in long references.

### Category 4 — Instruction Quality [WARN-capable]
Checks: error handling section, done-when checklist, constraints section, phase gates.

### Category 5 — Idempotency & Termination [WARN-capable]
Checks: loop caps, termination conditions, findings caps, binary vs semantic tests, self-critic prevention.

### Category 6 — Tool Scoping & Safety [WARN-capable]
Checks: allowed-tools field, read-only scoping, dangerous tool guards, disable-model-invocation.

### Category 7 — Cross-Cutting [NOTE-capable]
Checks: language consistency, _shared imports, anti-trigger clauses, output budget.

See [check-catalog.md](references/check-catalog.md) for the complete check list with binary test definitions.

---

## Phase 3 — REPORT

Print: `[Phase 3/3] REPORT`

**3a. Compile results:**

Count PASS/FAIL/WARN/NOTE across all categories.

**3b. Determine verdict:**

Verdict based on FAILs only. WARNs and NOTEs are informational.

| Condition | Verdict |
|-----------|---------|
| 0 FAIL | **HEALTHY** — Skill follows best practices |
| 0 FAIL, 1+ WARN/NOTE | **HEALTHY** — With observations |
| 1-3 FAIL | **NEEDS FIXES** — Structural issues to resolve |
| 4+ FAIL | **UNHEALTHY** — Significant structural problems |

**3c. Display report:**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILL DOCTOR REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Skill:** {name} ({path})
**Checks run:** {N}

**Verdict: {HEALTHY | NEEDS FIXES | UNHEALTHY}**

| Category | PASS | WARN | NOTE | FAIL |
|----------|------|------|------|------|
| File Structure      | {N} | {N} | {N} | {N} |
| Frontmatter         | {N} | {N} | {N} | {N} |
| Progressive Discl.  | {N} | {N} | {N} | {N} |
| Instructions        | {N} | {N} | {N} | {N} |
| Idempotency         | {N} | {N} | {N} | {N} |
| Tool Scoping        | {N} | {N} | {N} | {N} |
| Cross-Cutting       | {N} | {N} | {N} | {N} |

## Fixes Required (FAIL)

### {CHECK-ID} — {check name}
**Test:** {binary test description}
**Expected:** {what should be}
**Actual:** {what was found}
**Fix:** {concrete action}

## Observations (WARN + NOTE)

- **{CHECK-ID}** [WARN]: {description} → {concrete action}
- **{CHECK-ID}** [NOTE]: {observation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**3d. JSON output:**

```json
{
  "skill": "{name}",
  "path": "{path}",
  "verdict": "HEALTHY | NEEDS_FIXES | UNHEALTHY",
  "counts": { "pass": N, "warn": N, "note": N, "fail": N },
  "fails": [
    { "check": "FS-01", "category": "File Structure", "description": "...", "fix": "..." }
  ],
  "observations": [
    { "check": "FM-03", "type": "warn", "description": "..." }
  ]
}
```

---

## Hard Rules

### Pipeline
1. Read the ENTIRE SKILL.md before running checks — no partial reads.
2. Every check is a **binary structural test**. No semantic judgments ("vague", "unclear", "could be better").
3. Verdict based on FAIL count only. WARNs and NOTEs are informational.
4. Single-pass, then stop. No self-critic loop. No re-verification of findings.
5. Same skill input → same verdict on every run (idempotency guarantee).

### Objectivity
6. Do not invent problems. If a skill is well-structured → HEALTHY.
7. Do not suggest new features or capabilities. Scope = verify structure.
8. Do not rewrite the skill. This workflow is READ-ONLY.
9. Every FAIL cites the specific source rule it violates (Anthropic docs, technical-rules.md, or check-catalog.md).

### NEVER
- Modify any file — this workflow is READ-ONLY
- Propose new skill features or capabilities
- Use semantic judgments for FAIL or WARN verdicts
- Run more than 1 pass over the checks
- Present WARNs or NOTEs as "corrections required"
- Expand closed pattern lists with synonyms or "similar" words not in the procedure
- Use proximity checks ("within N lines") unless the procedure explicitly defines them
- Interpret table summaries independently when a procedure exists — the procedure is always authoritative
- Report a finding on documentation/meta-references that describe anti-patterns (e.g., a check-catalog listing "WARN.*vague" as an example is not itself a violation)

## Error Handling

| Scenario | Action |
|----------|--------|
| Skill path not found | List skills in `~/.claude/skills/`. Stop. |
| SKILL.md missing | FAIL on FS-01. Continue other checks with available files. |
| Frontmatter unparseable | FAIL on FM-01. Skip frontmatter checks. Continue body checks. |
| References dir missing | PASS on PD checks (references are optional for simple skills). |
| _shared files missing | NOTE. Not all skills use shared conventions. |

## References

- [Check Catalog](references/check-catalog.md) — Complete list of 30+ binary checks with test definitions, sources, and severity
