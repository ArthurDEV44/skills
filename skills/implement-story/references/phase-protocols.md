# Phase Protocols — Agent Prompts and Coordination

## Agent Spawning Rules

All agents are spawned using the `Agent` tool. Key parameters:

```
Agent(
  description: "3-5 word summary",
  prompt: "Detailed instructions",
  subagent_type: "agent-type"
)
```

**Subagent type assignments:**

| Phase | Agent | subagent_type | Tools Available |
|-------|-------|---------------|-----------------|
| 2a | Web research | `agent-websearch` | WebSearch, WebFetch, Exa MCP |
| 2c | Codebase exploration | `agent-explore` | Read, Grep, Glob, Bash |
| 2c | Documentation lookup | `agent-docs` | Read, Grep, Glob, Context7 MCP |
| 5 | Code review | `agent-explore` | Read, Grep, Glob, Bash |
| 6 | Security audit | `agent-explore` | Read, Grep, Glob, Bash |

Review and security agents use `agent-explore` because they need read-only codebase access with no write tools.

---

## Phase 2a — Web Research Prompt Template

```
Research how to implement the following user story in a software project.

## User Story
Title: {story_title}
Description: {story_description}
Acceptance Criteria:
{acceptance_criteria_list}

## Technical Context
- Language/Framework: {detected_from_manifest}
- Project type: {detected_from_structure}

## Research Focus
1. Best practices for implementing this type of feature
2. Common pitfalls and how to avoid them
3. Libraries or tools commonly used for this
4. Security considerations specific to this feature
5. Testing strategies for this type of functionality

## Search Strategy
- Search for "{feature_type} implementation best practices {language/framework}"
- Search for "common mistakes {feature_type} {language/framework}"
- If auth/payment/security related: search for "OWASP {feature_type} security"
- Include year 2026 in searches for time-sensitive topics

## Output Requirements
Return findings in this structure:

### Key Findings
[Numbered list of 3-8 findings, most important first]

### Libraries & Frameworks Mentioned
[List with version numbers if found]

### Implementation Patterns
[Specific patterns recommended for this type of feature]

### Security Considerations
[Security risks specific to this feature type]

### Sources
[All URLs as markdown links]
```

---

## Phase 2c — Codebase Exploration Prompt Template

```
Explore the codebase to understand how to implement the following user story.

## User Story
Title: {story_title}
Description: {story_description}

## Research Context (from web research)
{compressed_phase_2a_output — max 500 words}

## Exploration Tasks
1. Find existing code related to this feature area (similar features, same domain)
2. Identify the project's patterns and conventions for:
   - File organization (where should new files go?)
   - Error handling (how does this project handle errors?)
   - Testing (where are tests? what framework? what patterns?)
   - State management (how is state passed around?)
3. Find the entry point where this feature would integrate
4. Check for existing types, interfaces, or models that this feature should use
5. Identify any shared utilities or helpers relevant to implementation
6. Look for configuration or environment variables this feature might need

## Output Requirements
Return findings with file:line references for every claim. Structure:

### Project Architecture
[How the project is organized, relevant to this feature]

### Existing Patterns to Follow
[Conventions this implementation should match, with file:line examples]

### Integration Points
[Where the new code connects to existing code, with file:line]

### Relevant Types and Interfaces
[Existing types to use or extend, with file:line]

### Dependencies and Config
[Relevant dependencies, env vars, configuration]
```

---

## Phase 2c — Documentation Lookup Prompt Template

```
Look up official documentation for libraries needed to implement this feature.

## User Story
{story_title}: {story_description}

## Libraries to Look Up
{library_list_with_versions}

## Context
Web research found these libraries are relevant. Look up:
1. Exact API signatures for functions we'll need
2. Code examples for this specific use case
3. Version-specific behavior or migration notes
4. Configuration or setup requirements

## Important
- Use Context7 two-step protocol: resolve-library-id first, then query-docs
- Maximum 3 Context7 calls total
- Focus on precise API details, not general overviews
```

---

## Phase 5 — Code Review Prompt Template

```
Perform a thorough code review of recent changes implementing a user story.

## Context
Story: {story_title}
Description: {story_description}
Acceptance Criteria:
{acceptance_criteria_list}

## What Changed
Run: git diff --name-only HEAD~1
Then read each changed file.

## Review Checklist

### 1. Correctness
- Does the code actually implement the acceptance criteria?
- Are there logic errors, off-by-one errors, or incorrect conditions?
- Are all code paths handled (including error paths)?
- Are null/undefined/None values handled properly?
- Do loops terminate correctly?

### 2. Quality
- Are variable/function names clear and descriptive?
- Is the code readable without excessive comments?
- Is complexity reasonable (no unnecessary nesting, no god functions)?
- Are there DRY violations (copy-pasted code blocks)?
- Does the code follow the project's existing conventions?

### 3. Performance
- Are there unnecessary allocations in hot paths?
- Are there N+1 query patterns?
- Is there blocking I/O in async contexts?
- Are there memory leaks (unclosed resources, growing collections)?
- Are expensive operations cached when appropriate?

### 4. Tests
- Is there test coverage for the new functionality?
- Are edge cases tested (empty input, boundary values, error conditions)?
- Are tests deterministic (no timing dependencies, no random)?
- Do test names describe what they verify?

### 5. Acceptance Criteria Verification
For each acceptance criterion, verify:
- Is it fully implemented?
- Is there a test that validates it?
- Are there gaps between the criterion and the implementation?

## Output Format
For each finding, use this format:

### {Category}: {Title}
- **Severity:** MUST_FIX | SHOULD_FIX | CONSIDER | OK
- **File:** `path/to/file.ext:line`
- **Issue:** {what is wrong}
- **Suggestion:** {how to fix it}

End with a summary:
### Review Summary
- MUST_FIX: {count} findings
- SHOULD_FIX: {count} findings
- CONSIDER: {count} findings
- Overall assessment: {PASS / PASS_WITH_FIXES / FAIL}
```

---

## Phase 6 — Security Audit Prompt Template

```
Perform a security audit of recent code changes.

## Context
Story: {story_title}
Changes implement: {story_description}

## Scope
Run: git diff --name-only HEAD~1
Read each changed file thoroughly.

## Security Audit Checklist

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
- Math.random() or similar for tokens/secrets
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
- CORS misconfiguration
- Missing CSRF protection
- SSRF vectors (user-controlled URLs fetched server-side)
- Debug mode or verbose errors in production config

### 7. AI-Generated Code Anti-Patterns
- eval() with dynamic input
- innerHTML from untrusted sources
- subprocess with shell=True
- .unwrap() on user input (Rust)
- Missing error handling on external calls

## Output Format
For each finding:

### [{severity}] {Vulnerability Title}
- **File:** `path/to/file.ext:line`
- **Type:** CWE-XXX: {name}
- **Description:** {what is wrong and why it's dangerous}
- **Remediation:**
  ```{lang}
  // Before (vulnerable)
  {code}

  // After (fixed)
  {code}
  ```

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO

End with:
### Security Summary
- CRITICAL: {count}
- HIGH: {count}
- MEDIUM: {count}
- LOW: {count}
- INFO: {count}
- **Verdict:** {PASS / PASS_WITH_FIXES / FAIL}
```

---

## Parallel Spawning Rules

### Phase 2c — Explore + Docs

When both codebase and libraries are detected, spawn in a SINGLE message:

```
Agent(
  description: "Explore codebase for {topic}",
  prompt: <Phase 2c explore template>,
  subagent_type: "agent-explore"
)

Agent(
  description: "Fetch docs for {library}",
  prompt: <Phase 2c docs template>,
  subagent_type: "agent-docs"
)
```

### Phases 5 + 6 — Review + Security

Always spawn both in a SINGLE message:

```
Agent(
  description: "Code review for {story}",
  prompt: <Phase 5 template>,
  subagent_type: "agent-explore"
)

Agent(
  description: "Security audit for {story}",
  prompt: <Phase 6 template>,
  subagent_type: "agent-explore"
)
```

---

## Orchestrator Responsibilities

The orchestrator (main Claude session) handles:

1. **Phase 1:** Parse PRD, extract story, confirm with user
2. **Phase 2:** Spawn agents, compress outputs, synthesize research
3. **Phase 3:** Generate plan from synthesis + criteria, get user approval
4. **Phase 4:** Implement the plan (the orchestrator DOES write code here)
5. **Phase 5+6:** Spawn review agents (orchestrator does NOT review — fresh-context agents do)
6. **Phase 7:** Fix issues flagged by agents, re-run validations
7. **Phase 8:** Stage, commit, push (with user confirmation)

The orchestrator writes code ONLY in Phase 4 and Phase 7. In all other phases, it orchestrates.

---

## Compressed Summary Format

When passing Phase 2a output to downstream agents, compress to this format:

```markdown
## Research Context (for downstream agents)

Key findings on implementing "{story_title}":
1. {finding_1}
2. {finding_2}
3. {finding_3}

Recommended patterns: {pattern_1}, {pattern_2}
Relevant libraries: {lib1} (v{version}), {lib2} (v{version})
Security notes: {key_security_consideration}
```

Target: <500 words. Strip URLs and detailed explanations. Keep only facts and identifiers.

---

## Conventional Commit Quick Reference

| Type | When |
|------|------|
| `feat` | New feature or user story implementation |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semi colons, etc. (no code change) |
| `perf` | Performance improvement |
| `chore` | Build process, auxiliary tools, dependencies |

**Format:** `type(scope): description`

- Scope = module or feature area (e.g., `auth`, `api`, `cart`)
- Description = imperative mood, lowercase, no period, <72 chars
- Body = bullet list of changes + acceptance criteria + PRD reference
- Footer = `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

**Breaking changes:** Add `!` after scope: `feat(api)!: change response format`
