---
model: haiku
name: security-review
context: fork
description: "Comprehensive security audit of code changes. Analyzes changed files for OWASP Top 10 vulnerabilities, injection flaws, authentication issues, secrets exposure, and insecure patterns. Produces a structured report with severity ratings and actionable remediations. Use when the user says 'security review', 'security audit', 'check for vulnerabilities', 'OWASP check', '/security-review', or asks to review code for security issues. Do NOT trigger for general code quality reviews, refactoring, or non-security concerns."
argument-hint: "[file-or-folder?]"
allowed-tools: Read, Grep, Glob, Bash(git diff *), Bash(git log *)
---

# security-review — Security Audit Pipeline

## Overview

Systematic security audit that analyzes code changes for vulnerabilities. Works on any language/framework. Produces a structured report with severity levels (CRITICAL, HIGH, MEDIUM, LOW, INFO) and specific remediation actions.

## Execution Flow

```
Changed Files Detection
        |
        v
+---------------+
|  Step 1:      |
|  SCOPE        |  <- Detect changes, identify language/framework
|  (instant)    |
+-------+-------+
        |
        v
+-------+-------+
|  Step 2:      |
|  AUDIT        |  <- Systematic check against security checklist
|  (analysis)   |
+-------+-------+
        |
        v
+-------+-------+
|  Step 3:      |
|  REPORT       |  <- Structured findings with remediations
|  (output)     |
+-------+-------+
```

## Runtime Output Format

Before each step, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Step N/3] STEP_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step-by-Step Execution

### Step 1 — Scope Detection

Print: `[Step 1/3] SCOPE DETECTION`

**1a. Parse arguments and identify changed files:**

- If `$ARGUMENTS` contains a file or folder path → audit those specific files
- If `$ARGUMENTS` is empty → detect changed files from git:

```bash
git diff --name-only HEAD  # unstaged changes
git diff --name-only --cached  # staged changes
git diff --name-only main...HEAD  # all branch changes (fallback: master)
```

If no git changes found and no arguments provided, ask the user which files to audit.

**1b. Classify the scope:**

| Signal | Detection |
|--------|-----------|
| Language | File extensions (.rs, .ts, .py, .go, .java, etc.) |
| Framework | Import statements, manifest files |
| Risk tier | Auth/billing/crypto = HIGH, business logic = MEDIUM, UI/docs = LOW |

**1c. Read all changed files** using the Read tool. For large diffs (>20 files), prioritize:
1. Files touching auth, session, crypto, database, user input
2. New files (more likely to have new vulnerabilities)
3. Files with the most lines changed

### Step 2 — Security Audit

Print: `[Step 2/3] AUDIT`

For each changed file, systematically check against the security checklist. See [references/security-checklist.md](references/security-checklist.md) for the full checklist organized by category.

**Audit categories (always check all):**

1. **Injection** — SQL, XSS, command injection, LDAP, path traversal, template injection
2. **Authentication & Authorization** — broken access control, missing auth checks, privilege escalation
3. **Cryptography** — weak algorithms, hardcoded keys, insecure random, missing encryption
4. **Secrets** — hardcoded credentials, API keys, tokens, connection strings
5. **Data Handling** — insecure deserialization, sensitive data exposure, missing sanitization
6. **Configuration** — debug mode in production, CORS misconfiguration, permissive CSP, verbose errors
7. **Dependencies** — known vulnerable versions, unmaintained packages
8. **AI-Generated Code Patterns** — eval(), innerHTML from untrusted input, Math.random() for security, MD5/SHA1 for passwords

**For each finding, record:**
- Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
- File and line number
- Vulnerability type (CWE ID if applicable)
- Description of the issue
- Specific remediation with code example

### Step 3 — Report

Print: `[Step 3/3] REPORT`

Output the structured security report.

## Output Format

```markdown
## Security Audit Report

**Scope:** {N} files analyzed | Language: {lang} | Framework: {framework}
**Risk Summary:** {N} CRITICAL | {N} HIGH | {N} MEDIUM | {N} LOW | {N} INFO

### CRITICAL

#### [C-1] {Vulnerability Title}
- **File:** `path/to/file.ext:42`
- **Type:** {CWE-XXX: Vulnerability Name}
- **Description:** {What is wrong and why it's dangerous}
- **Remediation:**
  ```{lang}
  // Before (vulnerable)
  {vulnerable_code}

  // After (fixed)
  {fixed_code}
  ```

### HIGH

#### [H-1] {Title}
...

### MEDIUM

#### [M-1] {Title}
...

### LOW / INFO

#### [L-1] {Title}
...

### Summary

- **Total findings:** {N}
- **Must fix before merge:** {list CRITICAL + HIGH IDs}
- **Recommended fixes:** {list MEDIUM IDs}
- **No action required (informational):** {list LOW + INFO IDs}
```

## Severity Definitions

| Severity | Criteria | Action |
|----------|----------|--------|
| CRITICAL | Exploitable remotely, no auth needed, data breach/RCE risk | Block merge, fix immediately |
| HIGH | Exploitable with some prerequisites, auth bypass, significant data exposure | Block merge, fix before release |
| MEDIUM | Limited exploitability, defense-in-depth violation, information disclosure | Fix recommended |
| LOW | Best practice violation, minor information leak, hardening opportunity | Fix when convenient |
| INFO | Observation, code smell, potential future risk | No action required |

## Hard Rules

1. Read ALL changed files before auditing — never audit from memory or assumptions.
2. Check ALL 8 categories for every file — do not skip categories based on file type.
3. Every finding must include file:line, severity, and a specific remediation with code.
4. CRITICAL and HIGH findings must include a "Before/After" code example.
5. Never downplay severity — if it's exploitable remotely without auth, it's CRITICAL.
6. If no vulnerabilities found, explicitly state "No security issues found" — do not invent findings.
7. Do NOT modify any files — this is a read-only audit. Remediation is code examples only.
8. Print `[Step N/3]` progress headers before each step — NEVER skip progress indicators.

## DO NOT

- Skip the scope detection — you need to know what changed before auditing.
- Audit files that haven't changed (unless the user explicitly asks for a full audit).
- Report style issues or non-security code quality concerns — this is a security-only audit.
- Mark known-safe patterns as vulnerabilities (e.g., prepared statements are not SQL injection).
- Provide vague remediations like "validate input" — always show specific fixed code.

## Constraints (Three-Tier)

### ALWAYS
- Read ALL changed files before auditing
- Check ALL 8 audit categories for every file
- Include file:line, severity, and code remediation for every finding

### ASK FIRST
- Nothing — this is a read-only audit skill

### NEVER
- Modify any file — this is a read-only audit (remediation is code examples only)
- Downplay severity — if it's exploitable remotely without auth, it's CRITICAL
- Invent findings when no vulnerabilities exist — state "No security issues found"
- Report style issues or non-security concerns

## Done When

- [ ] Changed files identified and read (Step 1)
- [ ] All 8 audit categories checked for every file (Step 2)
- [ ] Structured report produced with severity ratings (Step 3)
- [ ] Every CRITICAL/HIGH finding includes before/after code remediation
- [ ] No files modified — this is a read-only audit

## References

- [Security Checklist](references/security-checklist.md) — detailed vulnerability patterns per category with language-specific indicators and remediation templates
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — ALWAYS/ASK FIRST/NEVER model
