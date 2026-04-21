# Output Format — Phase 6 Summary Report

## Report Template

```markdown
## Review Report: {PRD title or Story ID}

### Executive Summary

{2-3 sentences: overall verdict, critical blocking issues if any, key insight.}

**Verdict:** {ALL_CLEAR | PASS_WITH_FIXES | ISSUES_REMAINING}
**Phase results:** Intake {PASS} | Research {PASS} | Static {PASS/n warnings} | Validate {PASS/n false positives} | Review {PASS/FAIL} | Security {PASS/FAIL} | Remediation {PASS/PARTIAL}

---

**Mode:** {Single Story | Full PRD}
**Files reviewed:** {count} ({total_lines_changed} lines changed)
**Files filtered:** {count} (lock/generated/vendor)
**Stories reviewed:** {count}

### Acceptance Criteria Validation

| Story | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| US-001 | {criterion_1} | PASS | `file:line` |
| US-001 | {criterion_2} | FAIL | {what's missing} |
| US-001 | {criterion_3} | PARTIAL | {what's implemented, what's not} |
| ... | ... | ... | ... |

### Static Analysis Results

- **Errors fixed:** {count}
- **Warnings fixed:** {count}
- **Remaining:** {count} (with justification)

### Validation Results (Phase 4.5)

- **Findings received:** {total from Phase 3 + Phase 4}
- **CONFIRMED:** {count}
- **PARTIAL:** {count} (reclassified)
- **REFUTED:** {count} (false positives eliminated)
- **Signal ratio:** {confirmed + partial} / {total} ({percentage}%)

### Findings Summary

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Code Review | {n} | {n} | {n} | {n} |
| Security | {n} | {n} | {n} | {n} |
| Static Analysis | {n} | {n} | {n} | {n} |
| **Total** | **{n}** | **{n}** | **{n}** | **{n}** |

### Issues Fixed (auto-remediated — CRITICAL/HIGH/MUST_FIX only)

| ID | Severity | Confidence | Category | Description | Impact | File | Fix Applied |
|----|----------|------------|----------|-------------|--------|------|-------------|
| C-1 | CRITICAL | HIGH | security | {desc} | {why it matters} | `file:line` | {what was changed} |
| H-1 | HIGH | HIGH | correctness | {desc} | {why it matters} | `file:line` | {what was changed} |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Issues Requiring Human Review (if any)

| ID | Severity | Category | Description | Impact | Proposed Fix | Why Not Auto-Fixed |
|----|----------|----------|-------------|--------|-------------|-------------------|
| ... | ... | ... | ... | ... | ... | {touches auth / oscillates / novel issue} |

### Observations (SHOULD_FIX/CONSIDER — informational, not auto-fixed)

These findings are based on semantic judgment (naming, readability, architecture style) and have no mechanical oracle to confirm resolution. They are reported for context but were NOT entered into the remediation loop to prevent fix oscillation.

| ID | Tier | Category | Description | File |
|----|------|----------|-------------|------|
| O-1 | 2 | quality | {desc} | `file:line` |
| ... | ... | ... | ... | ... |

### Quality Gate Results

- {gate_1}: PASS / FAIL
- {gate_2}: PASS / FAIL
- Cognitive complexity: {max value found} (threshold: 10 warn, 20 gate)
- Test coverage for new code: {assessment}

### Research Insights Applied

- {insight_1 from Phase 2 that influenced a fix}
- {insight_2}

### Tracked for Later (out-of-scope findings)

- {finding that affects code outside the diff + 1 hop scope}
- {finding that would require architectural discussion}

### Recommendations

- {recommendation_1 — future improvement, not a blocking issue}
- {recommendation_2}

---
**Changes ready to commit:** {Yes — review `git diff` | No — see issues requiring human review}
```
