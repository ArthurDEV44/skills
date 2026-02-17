# Troubleshooting and Testing

## Testing Approach

### 1. Triggering Tests

**Goal:** Ensure skill loads at the right times.

Create a test suite:
```
Should trigger:
- "[Direct request matching skill purpose]"
- "[Paraphrased version of the same request]"
- "[Another variation using different keywords]"

Should NOT trigger:
- "[Unrelated topic A]"
- "[Unrelated topic B]"
- "[Similar-sounding but different task]"
```

Run 10-20 test queries. Track how many trigger the skill automatically vs. requiring explicit invocation.

### 2. Functional Tests

**Goal:** Verify correct outputs.

Test cases:
- Valid outputs generated
- API calls succeed (if applicable)
- Error handling works
- Edge cases covered

Example:
```
Test: Create project with 5 tasks
Given: Project name "Q4 Planning", 5 task descriptions
When: Skill executes workflow
Then:
    - Project created in ProjectHub
    - 5 tasks created with correct properties
    - All tasks linked to project
    - No API errors
```

### 3. Performance Comparison

**Goal:** Prove the skill improves results vs. baseline.

Compare same task with and without skill enabled. Count tool calls and total tokens consumed.

```
Without skill:
- User provides instructions each time
- 15 back-and-forth messages
- 3 failed API calls requiring retry
- 12,000 tokens consumed

With skill:
- Automatic workflow execution
- 2 clarifying questions only
- 0 failed API calls
- 6,000 tokens consumed
```

## Common Issues

### Skill won't upload

**Error: "Could not find SKILL.md in uploaded folder"**
- Cause: File not named exactly SKILL.md
- Solution: Rename to SKILL.md (case-sensitive). Verify with: `ls -la`

**Error: "Invalid frontmatter"**
- Cause: YAML formatting issue
- Common mistakes:
  ```yaml
  # Wrong - missing delimiters
  name: my-skill
  description: Does things

  # Wrong - unclosed quotes
  name: my-skill
  description: "Does things

  # Correct
  ---
  name: my-skill
  description: Does things
  ---
  ```

**Error: "Invalid skill name"**
- Cause: Name has spaces or capitals
  ```yaml
  # Wrong
  name: My Cool Skill

  # Correct
  name: my-cool-skill
  ```

### Skill doesn't trigger

**Symptom:** Skill never loads automatically.

**Fix:** Revise description field.

Quick checklist:
- Is it too generic? ("Helps with projects" won't work)
- Does it include trigger phrases users would actually say?
- Does it mention relevant file types if applicable?

**Debugging approach:** Ask Claude: "When would you use the [skill name] skill?" Claude will quote the description back. Adjust based on what's missing.

### Skill triggers too often

**Symptom:** Skill loads for unrelated queries.

Solutions:
1. **Add negative triggers:**
   ```yaml
   description: Advanced data analysis for CSV files. Use for
   statistical modeling, regression, clustering. Do NOT use for
   simple data exploration (use data-viz skill instead).
   ```

2. **Be more specific:**
   ```yaml
   # Too broad
   description: Processes documents

   # More specific
   description: Processes PDF legal documents for contract review
   ```

3. **Clarify scope:**
   ```yaml
   description: PayFlow payment processing for e-commerce. Use
   specifically for online payment workflows, not for general
   financial queries.
   ```

### Instructions not followed

**Symptom:** Skill loads but Claude doesn't follow instructions.

Common causes:

1. **Instructions too verbose**
   - Keep instructions concise
   - Use bullet points and numbered lists
   - Move detailed reference to separate files

2. **Instructions buried**
   - Put critical instructions at the top
   - Use ## Important or ## Critical headers
   - Repeat key points if needed

3. **Ambiguous language**
   ```markdown
   # Bad
   Make sure to validate things properly

   # Good
   CRITICAL: Before calling create_project, verify:
   - Project name is non-empty
   - At least one team member assigned
   - Start date is not in the past
   ```

4. **Model "laziness"** - Add explicit encouragement:
   ```markdown
   ## Performance Notes
   - Take your time to do this thoroughly
   - Quality is more important than speed
   - Do not skip validation steps
   ```

### MCP connection issues

**Symptom:** Skill loads but MCP calls fail.

Checklist:
1. **Verify MCP server is connected** - Check Settings > Extensions
2. **Check authentication** - API keys valid, proper permissions/scopes
3. **Test MCP independently** - Ask Claude to call MCP directly (without skill)
4. **Verify tool names** - Skill references correct MCP tool names (case-sensitive)

### Large context issues

**Symptom:** Skill seems slow or responses degraded.

Causes: Skill content too large, too many skills enabled simultaneously, all content loaded instead of progressive disclosure.

Solutions:
1. **Optimize SKILL.md size**
   - Move detailed docs to references/
   - Link to references instead of inline
   - Keep SKILL.md under 5,000 words
2. **Reduce enabled skills**
   - Evaluate if you have more than 20-50 skills enabled simultaneously
   - Recommend selective enablement
   - Consider skill "packs" for related capabilities

## Iteration Based on Feedback

Skills are living documents. Iterate based on:

**Undertriggering signals:**
- Skill doesn't load when it should
- Users manually enabling it
- Support questions about when to use it
- **Solution:** Add more detail and nuance to the description, include keywords

**Overtriggering signals:**
- Skill loads for irrelevant queries
- Users disabling it
- Confusion about purpose
- **Solution:** Add negative triggers, be more specific

**Execution issues:**
- Inconsistent results
- API call failures
- User corrections needed
- **Solution:** Improve instructions, add error handling

## Quality Metrics

### Quantitative
- Skill triggers on 90% of relevant queries
- Completes workflow in X tool calls
- 0 failed API calls per workflow

### Qualitative
- Users don't need to prompt Claude about next steps
- Workflows complete without user correction
- Consistent results across sessions
- New users accomplish tasks on first try with minimal guidance
