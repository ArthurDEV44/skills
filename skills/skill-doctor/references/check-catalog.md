# Check Catalog — Binary Structural Tests

Every check is a binary test: presence/absence, valid/invalid, count threshold. No semantic judgments.

> **Authority rule:** Each check has a **table row** (summary) and a **test procedure** (authoritative definition). When they diverge, the **procedure is canonical**. Execute the procedure exactly as written — do not interpret the table summary independently.

**Sources cited:**
- **[A]** = Anthropic skill authoring best practices
- **[T]** = skill-creator/references/technical-rules.md
- **[AP]** = Agent Patterns (agentpatterns.tech)
- **[AE]** = Anthropic Engineering — Demystifying Evals
- **[PW]** = Production Patterns (agentic workflows literature)
- **[CC]** = Claude Code docs (code.claude.com)

---

## Category 1 — File Structure

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| FS-01 | SKILL.md exists | `ls SKILL.md` → exists, exact case | FAIL | [T] |
| FS-02 | Folder is kebab-case | Folder name matches `^[a-z0-9]+(-[a-z0-9]+)*$` | FAIL | [T] |
| FS-03 | No README.md | `ls README.md` → not found | WARN | [T] |
| FS-04 | No broken internal links | Every `references/*.md` and `scripts/*` link in SKILL.md body resolves to a real file | FAIL | [T] |
| FS-05 | No deep nesting in references | No file in `references/` links to another file in `references/` (max 1 level from SKILL.md) | WARN | [A] |

### Test procedures

**FS-01:** `ls {skill_path}/SKILL.md 2>/dev/null` — must exist with exact case.

**FS-02:** `basename {skill_path}` — must match regex `^[a-z0-9]+(-[a-z0-9]+)*$`. No underscores, no uppercase, no spaces.

**FS-03:** `ls {skill_path}/README.md 2>/dev/null` — must NOT exist.

**FS-04:** Extract all paths matching `](references/...)` or `](scripts/...)` from SKILL.md. For each, verify file exists at `{skill_path}/{extracted_path}`.

**FS-05:** For each file in `references/`, grep for `](references/` or `](../references/`. If found → WARN (nesting detected).

---

## Category 2 — Frontmatter Quality

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| FM-01 | Frontmatter parseable | SKILL.md starts with `---` and has a closing `---` | FAIL | [T] |
| FM-02 | `name` field present | Frontmatter contains `name:` | FAIL | [T] |
| FM-03 | `name` format valid | Value matches `^[a-z0-9]+(-[a-z0-9]+)*$`, no "claude"/"anthropic" | FAIL | [T] |
| FM-04 | `name` matches folder | `name` value == `basename {skill_path}` | WARN | [T] |
| FM-05 | `description` field present | Frontmatter contains `description:` | FAIL | [A] |
| FM-06 | `description` has WHAT+WHEN | Description contains at least one verb from the closed set (WHAT) AND at least one trigger phrase from the closed set (WHEN). See procedure for exact lists. | WARN | [A] |
| FM-07 | `description` length | Length <= 1024 characters | WARN | [T] |
| FM-08 | No XML tags in frontmatter | Zero occurrences of `<` or `>` in any frontmatter value | FAIL | [T] |
| FM-09 | `description` has anti-triggers | Description contains any of: `"Do NOT"`, `"Do not use"`, `"Must NOT"`, `"MUST NOT"`, `"Not for"` (case-sensitive exact substring match) | NOTE | [A] |

### Test procedures

**FM-01:** Read first line of SKILL.md. Must be exactly `---`. Then find second occurrence of `---` within first 50 lines.

**FM-02:** Between the two `---` delimiters, grep for `^name:`.

**FM-03:** Extract value after `name:`. Strip quotes. Match against `^[a-z0-9]+(-[a-z0-9]+)*$`. Also check value does not contain "claude" or "anthropic" (case-insensitive).

**FM-04:** Compare extracted `name` value with `basename {skill_path}`.

**FM-05:** Between delimiters, grep for `^description:`.

**FM-06:** Extract description value. Check for presence of BOTH (case-insensitive grep):
- At least one verb from this **closed set**: `audit|analyz|build|check|creat|deploy|extract|fetch|generat|inspect|orchestrat|process|produc|review|run|scan|search|synthesiz|transform|validat|verif`
- At least one trigger phrase from this **closed set**: `"Use when"|"Use this when"|"Trigger when"|"Invoke with"|"Use this skill"|"when the user says"|"when the user asks"` (exact substring match, case-insensitive)

**FM-07:** `echo -n "{description}" | wc -c` — must be <= 1024.

**FM-08:** Grep all frontmatter lines for `<` or `>`. Zero matches = PASS.

**FM-09:** Grep description for the exact substrings (case-sensitive): `"Do NOT"`, `"Do not use"`, `"Must NOT"`, `"MUST NOT"`, `"Not for"`. Any match = PASS, no match = NOTE.

---

## Category 3 — Progressive Disclosure

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| PD-01 | Body length reasonable | SKILL.md body (after frontmatter) <= 500 lines | WARN | [A] |
| PD-02 | Long refs have ToC | Each file in `references/` with >100 lines contains a heading within the first 10 lines | NOTE | [A] |
| PD-03 | References linked from SKILL.md | Every file in `references/` is linked at least once from SKILL.md body | WARN | [A] |
| PD-04 | No orphan references | Count files in `references/` not linked from SKILL.md | NOTE | [A] |

### Test procedures

**PD-01:** Count lines after the closing `---` of frontmatter. If > 500 → WARN.

**PD-02:** For each `references/*.md` with > 100 lines, check if lines 1-10 contain a `#` heading. Missing = NOTE.

**PD-03:** For each file in `references/`, check if SKILL.md body contains a link to it (substring match on filename). Missing = WARN.

**PD-04:** Count of files in `references/` not linked from SKILL.md body. If > 0 → NOTE with list.

---

## Category 4 — Instruction Quality

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| IQ-01 | Error handling section | SKILL.md body contains `## Error Handling` or `## Errors` heading OR a `| Scenario | Action |` table | WARN | [A] |
| IQ-02 | Done-when checklist | SKILL.md body contains `Done When` or `Done when` heading with `- [ ]` items | NOTE | [A] |
| IQ-03 | Constraints section | SKILL.md body contains `## Constraints` or `## Hard Rules` heading, OR both `NEVER` and `ALWAYS` keywords anywhere in the body | WARN | [A] |
| IQ-04 | Phase gates (multi-phase only) | If SKILL.md has 3+ markdown headings (`##` or `###`) starting with `Phase` (excluding code blocks), each phase section contains `GATE` or `gate` keyword | WARN | [AE] |
| IQ-05 | Examples present | SKILL.md body or references/ contains `Example` or `example` heading or code block | NOTE | [A] |

### Test procedures

**IQ-01:** Grep SKILL.md body for `## Error Handling` or `## Errors` or `| Scenario | Action |` (table format). Present = PASS.

**IQ-02:** Grep for `## Done When` or `## Done when`. If found, check for `- [ ]` within 20 lines. Both present = PASS.

**IQ-03:** Grep for `## Constraints` OR `## Hard Rules`. If neither found, grep for both `ALWAYS` and `NEVER` keywords (case-sensitive, any location in body). If BOTH present anywhere = PASS. Otherwise = WARN.

**IQ-04:** Count markdown headings (`##` or `###`) whose text starts with `Phase` — exclude lines inside triple-backtick code blocks. If count >= 3, grep the text between each such heading and the next heading of same/higher level for `GATE` or `**GATE` or `gate:`. Missing in any phase section = WARN.

**IQ-05:** Grep SKILL.md + references/ for `## Example` or `### Example` or triple-backtick code blocks with a language tag. Present = PASS.

---

## Category 5 — Idempotency & Termination

These checks apply primarily to **multi-phase pipeline skills** (skills with 3+ phases or that spawn agents). For simple knowledge/reference skills, most will be N/A.

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| ID-01 | Loop cap defined | If SKILL.md contains "retry" or "max.*iteration" or "loop.*cap", a max cap number exists anywhere in body | WARN | [AP] |
| ID-02 | Findings cap defined | If SKILL.md produces findings/issues, a "max" or "cap" number is present | WARN | [AP] |
| ID-03 | No self-critic loop | SKILL.md does NOT contain a pattern where output is re-evaluated by the same phase (grep for "re-verify own" or "re-evaluate" + "findings") | NOTE | [AP] |
| ID-04 | Semantic checks produce NOTE | If SKILL.md defines WARN/FAIL verdict outputs, grep verdict-definition sections only for `WARN.*vague|WARN.*unclear|FAIL.*inconsistent|FAIL.*unrealistic`. These semantic judgments should map to NOTE, not FAIL/WARN | WARN | [AE] |
| ID-05 | Verdict on structural tests only | If SKILL.md has a verdict table (contains `Verdict` and `|`), grep the table's condition column for `WARN` or `NOTE` as part of a numeric threshold (e.g., `WARN > 0`, `3+ WARN`). Found = WARN | WARN | [AE] |
| ID-06 | Agent output budget | If skill spawns agents (contains `Agent(`), a token/word budget is stated | NOTE | [CC] |

### Test procedures

**ID-01:** Grep body for `retry` or `max.*iteration` or `iteration.*cap` or `iteration.*limit` (case-insensitive). If none found → N/A (skip). If found, grep the entire body for `max \d|cap \d|limit \d|\d+ iteration|\d+ retr` (a number adjacent to a cap keyword, anywhere in the body). Number present = PASS. Absent = WARN.

**ID-02:** Grep for `findings|issues|checks`. If the skill produces structured output (contains `FAIL|WARN|severity`), grep for `max|cap|limit` + number. Present = PASS.

**ID-03:** Grep for patterns like `re-verify.*findings`, `re-evaluate.*own`, `check.*own.*output`. Absent = PASS (no self-critic). Present = NOTE.

**ID-04:** First, identify verdict-definition sections: locate the section(s) where the skill defines what triggers WARN or FAIL outputs (typically a verdict table, severity table, or check-definition section). Grep ONLY those sections for `WARN.*vague|WARN.*unclear|FAIL.*inconsistent|FAIL.*unrealistic|FAIL.*tension`. Exclude check-catalog documentation or meta-references that describe anti-patterns. If found in the verdict-definition section → WARN. Not found = PASS.

**ID-05:** Grep for lines containing both `|` and `Verdict` (table rows). If no such table exists → N/A (skip). If found, grep only those table rows for patterns where WARN or NOTE appear adjacent to a comparison operator or number: `WARN.*>|WARN.*>=|\d+.*WARN|NOTE.*>|NOTE.*>=|\d+.*NOTE|WARN.*\+|NOTE.*\+`. If any match → WARN (verdict depends on non-structural counts). No match = PASS.

**ID-06:** Grep for `Agent(`. If found, grep for `budget|tokens|words|1,500|1500`. Present = PASS.

---

## Category 6 — Tool Scoping & Safety

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| TS-01 | `allowed-tools` for read-only skills | If SKILL.md contains "READ-ONLY" or "read-only" or "does not modify", frontmatter should have `allowed-tools` that excludes Write/Edit | WARN | [CC] |
| TS-02 | `disable-model-invocation` for heavy pipelines | If SKILL.md has 5+ phases AND spawns 3+ agents, frontmatter should have `disable-model-invocation: true` | NOTE | [A] |
| TS-03 | Dangerous operations guarded | If SKILL.md contains `delete|drop|reset|force|--hard|rm -rf` (case-insensitive), a confirmation/escalation step is present within 5 lines | WARN | [PW] |
| TS-04 | `argument-hint` for parameterized skills | If SKILL.md body contains `$ARGUMENTS`, frontmatter has `argument-hint` | WARN | [T] |

### Test procedures

**TS-01:** Grep body for `READ-ONLY|read-only|does not modify|ne modifie pas`. If found, check frontmatter for `allowed-tools:`. If allowed-tools is missing or contains `Write` or `Edit` → WARN.

**TS-02:** Count `Phase` headings (N). Grep for `Agent(` occurrences (M). If N >= 5 AND M >= 3, check frontmatter for `disable-model-invocation`. Missing = NOTE.

**TS-03:** Grep body for dangerous operation keywords. For each match, check if the surrounding 5 lines contain "confirm", "ask", "escalate", "user approval", "ASK FIRST". Missing = WARN.

**TS-04:** Grep body for `$ARGUMENTS`. If found, check frontmatter for `argument-hint:`. Missing = WARN.

---

## Category 7 — Cross-Cutting

| ID | Check | Test | Severity | Source |
|----|-------|------|----------|--------|
| CC-01 | _shared imports consistent | If SKILL.md references `_shared/`, verify the referenced files exist in `~/.claude/skills/_shared/` | WARN | Internal |
| CC-02 | Output format defined | If SKILL.md body contains a `## Output` or `### Output Format` heading, the section must contain a triple-backtick code block | NOTE | [A] |
| CC-03 | `model` field for agent orchestrators | If skill spawns 3+ agents, frontmatter has `model:` field. Note: `model` is a user convention, not part of the official Anthropic SKILL.md spec | NOTE | Internal |

### Test procedures

**CC-01:** Extract all paths matching `_shared/` from SKILL.md. For each, check file exists at `~/.claude/skills/_shared/{filename}`. Missing = WARN.

**CC-02:** Grep body for `## Output` or `### Output Format` heading. If no such heading exists → N/A (skip). If heading found, check if the section (text between this heading and the next heading of same or higher level) contains a triple-backtick code block. Code block present = PASS. Missing = NOTE.

**CC-03:** Count `Agent(` or `subagent_type` occurrences. If >= 3, check frontmatter for `model:`. Missing = NOTE.

---

## Check Count Summary

| Category | Checks | Max FAIL | Max WARN | Max NOTE |
|----------|--------|----------|----------|----------|
| File Structure | 5 | 3 | 2 | 0 |
| Frontmatter | 9 | 4 | 3 | 2 |
| Progressive Disclosure | 4 | 0 | 2 | 2 |
| Instructions | 5 | 0 | 3 | 2 |
| Idempotency | 6 | 0 | 4 | 2 |
| Tool Scoping | 4 | 0 | 3 | 1 |
| Cross-Cutting | 3 | 0 | 1 | 2 |
| **Total** | **36** | **7** | **18** | **11** |

Maximum possible FAILs: 7 (all in File Structure + Frontmatter — the foundational categories).
