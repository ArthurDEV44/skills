# Phase Protocols — Agent Prompts and Coordination

## Agent Assignments

| Phase | Role | subagent_type | Condition |
|-------|------|---------------|-----------|
| 2b | Web research | `agent-websearch` | STANDARD/COMPLEX only |
| 2d | Codebase exploration | `agent-explore` | Always (codebase detected) |
| 2d | Documentation lookup | `agent-docs` | New/unfamiliar libraries only |
| 6+7 | Combined review+security | `agent-explore` | SIMPLE only |
| 6 | Code review | `agent-explore` | STANDARD/COMPLEX |
| 7 | Security audit | `agent-explore` | STANDARD/COMPLEX |

See [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) for the full CAN/CANNOT table and call budgets.

---

## Phase 2b — Web Research Prompt Template

```
Research best practices for implementing the following user story.

## User Story
Title: {story_title}
Description: {story_description}
Acceptance Criteria:
{acceptance_criteria_list}

## Technical Context
- Language/Framework: {detected_from_manifest}
- Project type: {detected_from_structure}

## Research Goal
Find best practices, common pitfalls, security considerations, and recommended libraries for implementing this type of feature in {language/framework}. Prioritize official documentation and engineering blogs from major companies. For each key finding, also look for known limitations or counter-evidence.

## Output Budget
Maximum 1,000 tokens. Prioritize findings by relevance to the acceptance criteria. Cut low-value details.

## Output Requirements
### Key Findings
[Numbered list of 3-5 findings, most important first]

### Libraries & Versions
[List with version numbers if found]

### Implementation Patterns
[Specific patterns recommended for this feature type]

### Security Considerations
[Risks specific to this feature]

### Sources
[All URLs as markdown links]
```

---

## Phase 2d — Codebase Exploration Prompt Template

```
Explore the codebase to understand how to implement the following user story.

## User Story
Title: {story_title}
Description: {story_description}

## Research Context (from web research)
{compressed_phase_2b_output — max 500 words}

## Exploration Tasks
1. Find existing code related to this feature area
2. Identify project conventions: file organization, error handling, testing patterns, state management
3. Find the integration point for this feature
4. Check for existing types, interfaces, or models to reuse
5. Identify relevant shared utilities
6. Check for configuration or environment variables needed

## Output Requirements
Return findings with file:line references for every claim:

### Project Architecture
[Organization relevant to this feature]

### Existing Patterns to Follow
[Conventions to match, with file:line examples]

### Integration Points
[Where new code connects to existing code, with file:line]

### Relevant Types and Interfaces
[Existing types to use or extend, with file:line]

### Dependencies and Config
[Relevant dependencies, env vars, configuration]

## Output Budget
Maximum 1,000 tokens. Include file:line references for every claim. Cut narrative.
```

---

## Phase 2d — Documentation Lookup Prompt Template

```
Look up official documentation for libraries needed to implement this feature.

## User Story
{story_title}: {story_description}

## Libraries to Look Up
{library_list_with_versions}

## What to Find
1. Exact API signatures for functions we'll need
2. Code examples for this specific use case
3. Version-specific behavior or migration notes
4. Configuration or setup requirements

## ctx7 CLI Protocol
Two-step process via Bash:
1. bunx ctx7@latest library {library_name} "{query}"  — resolve library ID
2. bunx ctx7@latest docs {library_id} "{query}"       — fetch documentation

## Output Budget
Maximum 800 tokens. Return exact API signatures and code examples only. No general overviews.

## Important
- Use ctx7 CLI two-step protocol: library first, then docs
- Maximum 3 ctx7 calls total
- Do NOT modify any files — this is read-only research
- Focus on precise API details, not general overviews
```

---

## Phases 6+7 Combined — Review + Security (SIMPLE stories only)

```
Review recent changes for correctness and security issues.

## Context
Story: {story_title}
Acceptance Criteria: {acceptance_criteria_list}
Changed files: {changed_files_list}

Read each changed file.

## Review Focus
1. **Correctness:** Does the code implement the acceptance criteria? Logic errors, unhandled paths, null safety?
2. **Quality:** Naming, conventions, DRY, complexity?
3. **Security:** Injection (SQL/XSS/command/path traversal), hardcoded secrets, missing auth checks, insecure deserialization, eval() with dynamic input?
4. **Tests:** Coverage for new functionality? Edge cases?

Skip sections with no attack surface in the changed code. Focus effort where the risk is.

## Output Format
### {Title}
- **Severity:** MUST_FIX | SHOULD_FIX | CRITICAL | HIGH | MEDIUM
- **File:** `path/to/file.ext:line`
- **Issue:** {what is wrong}
- **Suggestion:** {how to fix}

### Summary
- Findings by severity count
- Verdict: PASS | PASS_WITH_FIXES | FAIL

## Output Budget
Maximum 1,000 tokens. Focus on MUST_FIX/CRITICAL/HIGH. Omit OK/INFO items.
```

---

## Phase 6 — Code Review Prompt Template

```
Perform a thorough code review of recent changes implementing a user story.

## Context
Story: {story_title}
Description: {story_description}
Acceptance Criteria:
{acceptance_criteria_list}

## What Changed
The orchestrator passes a list of files modified during implementation.
Changed files: {changed_files_list}
Read each changed file. If the list is unavailable, run: `git diff --name-only $(git merge-base HEAD main)..HEAD`

## Review Checklist
Focus effort on sections relevant to the changed code. Skip sections with no applicable surface (e.g., skip Performance for pure config changes, skip Tests if no test files changed).

### 1. Correctness
- Does the code implement the acceptance criteria?
- Logic errors, off-by-one, incorrect conditions?
- All code paths handled (including error paths)?
- Null/undefined/None handled properly?

### 2. Quality
- Clear, descriptive naming?
- Readable without excessive comments?
- Reasonable complexity (no god functions)?
- DRY violations?
- Follows project conventions?

### 3. Performance
- Unnecessary allocations in hot paths?
- N+1 query patterns?
- Blocking I/O in async contexts?
- Memory leaks?

### 4. Tests
- Coverage for new functionality?
- Edge cases tested?
- Deterministic (no timing/random dependencies)?

### 5. Acceptance Criteria
For each criterion: fully implemented? Test validates it? Gaps?

### 6. Architecture Consistency
- Does new code follow project conventions (error handling, file structure, naming)?
- Are existing shared utilities used instead of duplicated logic?
- Does state management match established patterns?
- Are deprecated internal APIs or patterns avoided?
- Does the code integrate cleanly at the identified integration points?

## Output Format
### {Category}: {Title}
- **Severity:** MUST_FIX | SHOULD_FIX | CONSIDER | OK
- **File:** `path/to/file.ext:line`
- **Issue:** {what is wrong}
- **Suggestion:** {how to fix}

### Review Summary
- MUST_FIX: {count} | SHOULD_FIX: {count} | CONSIDER: {count}
- Overall: PASS | PASS_WITH_FIXES | FAIL

## Output Budget
Maximum 1,500 tokens. Focus on MUST_FIX and SHOULD_FIX findings. Omit OK items.
```

---

## Phase 7 — Security Audit Prompt Template

```
Perform a security audit of recent code changes.

## Context
Story: {story_title}
Changes implement: {story_description}

## Scope
The orchestrator passes a list of files modified during implementation.
Changed files: {changed_files_list}
Read each changed file thoroughly. If the list is unavailable, run: `git diff --name-only $(git merge-base HEAD main)..HEAD`

## Audit Checklist
Focus on sections with actual attack surface in the changed code. Skip sections that don't apply (e.g., skip Cryptography if no crypto code changed, skip AI Agent Risks if no LLM integration).

### 1. Injection (CWE-89, CWE-79, CWE-78, CWE-22)
- SQL injection: string concatenation in queries, missing parameterization
- XSS: innerHTML, dangerouslySetInnerHTML, unescaped template output
- Command injection: exec(), system(), shell=True with user input
- Path traversal: user input in file paths without canonicalization

### 2. Authentication & Authorization (CWE-284, CWE-287)
- Missing auth checks on new endpoints/routes
- Direct object reference without ownership validation
- Privilege escalation vectors
- Missing rate limiting on sensitive operations

### 3. Cryptography (CWE-327, CWE-338)
- Weak algorithms (MD5, SHA1 for security, DES, RC4)
- Math.random() for tokens/secrets
- Hardcoded encryption keys

### 4. Secrets (CWE-798)
- Hardcoded passwords, API keys, tokens
- Connection strings with credentials
- .env files or secrets in committed code

### 5. Data Handling (CWE-502, CWE-200)
- Insecure deserialization
- Sensitive data in logs, URLs, or client storage
- Missing input validation at system boundaries

### 6. Configuration (CWE-16, CWE-352, CWE-918)
- CORS misconfiguration, missing CSRF protection
- SSRF vectors (user-controlled URLs fetched server-side)
- Debug mode or verbose errors in production config

### 7. AI-Generated Code Anti-Patterns
- eval() with dynamic input
- innerHTML from untrusted sources
- subprocess with shell=True
- .unwrap() on user input (Rust)
- Missing error handling on external calls

### 8. AI Agent-Specific Risks (OWASP Agentic Top 10)
- Prompt injection vectors: if code processes LLM inputs, check for injection surfaces
- Tool misuse patterns: functions that execute arbitrary commands from user-controlled data
- Data leakage via logs: sensitive data (tokens, PII, credentials) written to logs or console
- Excessive autonomy: code that takes irreversible actions without confirmation gates

## Output Format
### [{severity}] {Vulnerability Title}
- **File:** `path/to/file.ext:line`
- **Type:** CWE-XXX: {name}
- **Description:** {what is wrong and why}
- **Remediation:** before/after code snippet

Severity: CRITICAL | HIGH | MEDIUM | LOW | INFO

### Security Summary
CRITICAL: {count} | HIGH: {count} | MEDIUM: {count} | LOW: {count} | INFO: {count}
Verdict: PASS | PASS_WITH_FIXES | FAIL

## Output Budget
Maximum 1,500 tokens. Focus on CRITICAL/HIGH findings. Omit INFO items unless no higher findings exist.
```

