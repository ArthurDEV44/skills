# Agent Protocols — Prompt Templates

Exact Agent tool parameters and prompt templates for each agent in the meta-code pipeline. For pipeline logic and step descriptions, see [SKILL.md](../SKILL.md).

## Narrative Reframing Rule

When passing context to downstream agents (typed handoffs, research context), always frame prior findings in **third person**: "Web research found X", "A prior analysis identified Y" — never "I found X". This prevents the receiving agent from attributing prior findings to itself and avoids hallucinated action attribution (Google ADK production pattern).

## Five-Part Subagent Brief

Every agent prompt follows this structure:
1. **Objective** — what to find/answer
2. **Output format** — exact sections and structure expected
3. **Output budget** — max 1,500 tokens returned to orchestrator
4. **Source/tool guidance** — what tools to use, what NOT to do
5. **Task boundaries** — scope limits, what to skip

---

## Step 2: agent-websearch (RESEARCH)

```
Agent(
  description: "Web research on {topic}",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

### Prompt Template

```
Research the following development question thoroughly using web search.

## Objective
{user_question}

{# If complex query with sub-questions:}
## Sub-questions
{sub_questions_targeting_websearch}

## Done Criteria
Your answer is complete ONLY when it addresses ALL of these:
- Must answer: {must_answer_items_from_step_1b}
- Must include: {must_include_items_from_step_1b}
If you cannot address an item, explicitly note it as a gap — do NOT silently skip it.

## Search Strategy
- Search for current best practices, recent articles, and official guidance.
- If the question involves specific libraries, search for latest docs and changelogs.
- Use 2-3 complementary searches covering different angles.
- Include the current year in searches for time-sensitive topics.

## Dual-Perspective Protocol
For each of your top 3 findings, run ONE additional search:
  "{finding}" limitations OR problems OR alternatives
Do NOT only search for confirming evidence. Report both supporting and challenging evidence.

## Output Format
Return findings in this exact structure (max 1,500 tokens total):

### Key Findings
[Numbered list of 3-8 key findings, most important first]

### Libraries & Frameworks Mentioned
[Relevant libraries with version numbers if found. Format: "- library-name (vX.Y.Z if known)"]

### Best Practices
[Specific, actionable recommendations from authoritative sources]

### Contradictions Detected
[Cases where sources disagree. Format:
- Claim: "{claim}"
  - Source A ({url}): says X
  - Source B ({url}): says Y
If none: "None detected."]

### Sources
[All URLs consulted as markdown links with source tier annotation: T1 (official docs), T2 (major eng blog), T3 (community), T4 (AI-generated/SEO)]

## Output Budget
Return at most 1,500 tokens. Prioritize findings by relevance to the question, cut low-value details.

## Task Boundaries
- Do NOT read local code or files.
- Do NOT run ctx7 CLI or look up library docs.
- Focus on web-accessible information only.
```

### Typed Handoff Compression (orchestrator, after Step 2)

After Step 2 completes, the orchestrator compresses output into a typed handoff object (Step 3a). See [workflow-engine.md](workflow-engine.md) for the canonical Typed Handoff Format.

---

## Step 4: agent-explore (EXPLORE)

```
Agent(
  description: "Explore codebase for {topic}",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

### Prompt Template

```
Explore the codebase in the current working directory to find code, patterns, and architecture relevant to the following question.

## Objective
{user_question}

## Research Context (typed handoff)
{step_3a_typed_handoff_object}

## Done Criteria
Your answer is complete ONLY when it addresses ALL of these:
- Must answer: {must_answer_items_from_step_1b}
- Must include: {must_include_items_from_step_1b}
If you cannot address an item, explicitly note it as a gap — do NOT silently skip it.

## Exploration Focus
1. Find existing code that relates to the question — functions, types, modules, handlers.
2. Identify patterns and conventions that affect the approach.
3. Check for existing implementations (partial or complete).
4. Note architecture, dependencies, and module structure relevant to the question.
5. Focus exploration on relevant areas — do not do a full project scan.

## Output Format
Return findings with file:line references for every claim (max 1,500 tokens total).

At the end, include:
### Contradictions with Web Research
[Conflicts between web recommendations and codebase patterns. Note if deviations appear intentional. If none: "None detected."]

## Output Budget
Return at most 1,500 tokens. Prioritize findings by relevance, cut low-value details.

## Task Boundaries
- Do NOT fetch URLs or search the web.
- Do NOT run ctx7 CLI or look up library docs.
- Do NOT modify any files.
- Read-only exploration of the codebase.
```

---

## Step 4: docs via ctx7 CLI (DOCUMENT)

```
Agent(
  description: "Fetch docs for {library}",
  prompt: <template below>,
  subagent_type: "agent-docs"
)
```

### Prompt Template

```
Look up official documentation for the following libraries using the ctx7 CLI to answer a specific question.
This is a READ-ONLY research task. Do NOT modify any files.

## Objective
{user_question}

## Libraries to Look Up
{library_list_with_versions}

## ctx7 CLI Protocol (MUST follow)

Two-step process: resolve library ID first, then query docs.

### Step 1 — Resolve library ID
Run via Bash:
  bunx ctx7@latest library {library_name} "{user_question}"

Select the most relevant result by name similarity, snippet count, and reputation.
Note the library ID (format: /org/project).
If a specific version is needed, use /org/project/version.

### Step 2 — Query documentation
Run via Bash:
  bunx ctx7@latest docs {library_id} "{focused_query}"

Use a specific, descriptive query — not single words.

### Hard limits
- Maximum 3 ctx7 calls total (combining library + docs calls).
- If you cannot find what you need after 3 attempts, use the best result you have.

## Research Context (typed handoff)
{step_3a_typed_handoff_object}

## Done Criteria
Your answer is complete ONLY when it addresses ALL of these:
- Must answer: {must_answer_items_from_step_1b}
- Must include: {must_include_items_from_step_1b}
If you cannot address an item, explicitly note it as a gap — do NOT silently skip it.

Focus on:
1. Exact API signatures and types relevant to the question
2. Official code examples for the specific use case
3. Version-specific behavior, deprecations, or migration notes
4. Configuration or setup requirements

## Contradiction Check
Web research claimed:
{relevant_web_research_claims_from_handoff}

If any claims conflict with official documentation, note the discrepancy.

## Output Format (max 1,500 tokens total)

### Answer
[Direct answer based on official docs]

### Code Examples
[Runnable snippets from official docs]

### Key API Details
[Signatures, types, parameters]

### Version Notes
[Version-specific behavior. If none: "N/A."]

### Contradictions with Web Research
[Conflicts between docs and web claims. If none: "None detected."]

### Sources
[ctx7 library IDs used and relevant doc sections]

## Output Budget
Return at most 1,500 tokens. Prioritize precise API details over general information.

## Task Boundaries
- Use ctx7 CLI two-step protocol: library first, then docs.
- Maximum 3 ctx7 calls total.
- Do NOT fetch arbitrary URLs or search the web.
- Do NOT read local code beyond manifest files for version info.
- Do NOT modify any files — this is read-only research.
- Do NOT repeat general information — focus on precise API details.
```

---

## Step 6: Challenge Agent (CHALLENGE)

```
Agent(
  description: "Challenge claims on {topic}",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

**Skip for `simple` queries.**

The challenge agent is an independent adversarial reviewer with a fresh context window. It never sees the full synthesis draft — only the extracted claims.

### Prompt Template

```
You are an independent reviewer. Your job is to challenge the following claims by searching for counter-evidence, known limitations, and edge cases.

## Claims to Challenge
{extracted_top_3_5_claims_with_sources}

## Original Question Context
{user_question_summary}

## Skepticism Protocol
You are a SKEPTICAL reviewer, not a helpful assistant. Your default stance is doubt.
- Assume each claim is wrong until evidence proves otherwise.
- A claim supported by a single source is WEAK, not confirmed.
- "No counter-evidence found" after 2 searches per claim = CONFIRMED, not before.
- Do NOT soften your verdicts. WEAKENED means WEAKENED. REFUTED means REFUTED.
- Do NOT give the benefit of the doubt — that is the synthesizer's job, not yours.

## Your Task
For EACH claim above:
1. Search for counter-evidence: known limitations, edge cases, or contradicting sources.
2. Search for recency: is this claim still current, or has it been superseded?
3. Assess strength: does the cited source actually support the claim as stated?

## Output Format (max 1,500 tokens total)

For each claim, return ONE of:
- **CONFIRMED** — counter-search found no contradicting evidence. Claim is solid.
- **WEAKENED** — found limitations or nuance. Detail: {what was found} | Source: {url}
- **REFUTED** — found strong counter-evidence. Detail: {what was found} | Source: {url}

### Summary
[1-3 sentences: overall assessment of claim quality]

## Output Budget
Return at most 1,500 tokens. Be concise — verdict + evidence per claim.

## Task Boundaries
- Do NOT read local code or files.
- Do NOT see or evaluate the full synthesis draft — only the claims listed above.
- Search specifically for counter-evidence, not confirming evidence.
- If you find no counter-evidence after 2 searches per claim, mark as CONFIRMED.
```

---

## Step 2: agent-websearch — Critical Angle (complex queries only)

Spawned in parallel with the supportive-angle agent. Both use a SINGLE Agent tool message.

```
Agent(
  description: "Web research on {topic} — critical angle",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

### Prompt Template

```
Research the LIMITATIONS, PROBLEMS, and ALTERNATIVES for the following development question.

## Objective
{user_question}

## Critical Angle
You are specifically searching for:
1. Known limitations, gotchas, and edge cases
2. Common problems and failure modes reported by practitioners
3. Alternative approaches that competing sources recommend
4. Deprecations, breaking changes, or upcoming replacements
5. Performance issues, scalability concerns, or security warnings

Do NOT search for best practices or recommendations — the other agent handles that.

## Done Criteria
Your answer is complete ONLY when it addresses ALL of these:
- Must answer: {must_answer_items_from_step_1b}
- Must include: {must_include_items_from_step_1b}
If you cannot address an item, explicitly note it as a gap.

## Search Strategy
- Use searches like: "{topic} problems", "{topic} alternatives to", "{topic} deprecated", "{topic} known issues"
- Include the current year in searches for time-sensitive topics.
- Prioritize practitioner experience reports over marketing content.

## Output Format
Return findings in this exact structure (max 1,500 tokens total):

### Limitations & Problems
[Known issues, gotchas, edge cases — most impactful first]

### Alternatives
[Competing approaches with trade-offs]

### Deprecations & Breaking Changes
[If any. "None found." if none.]

### Sources
[All URLs with tier annotations]

## Output Budget
Return at most 1,500 tokens.

## Task Boundaries
- Do NOT read local code or files.
- Do NOT run ctx7 CLI.
- Do NOT search for best practices — only critical/challenging evidence.
```

---

## Step 7d: Evaluator Agent (complex queries only)

Independent evaluator that receives the synthesis but has NOT generated it. Enforces generator-evaluator separation.

```
Agent(
  description: "Evaluate synthesis on {topic}",
  prompt: <template below>,
  subagent_type: "general-purpose"
)
```

### Prompt Template

```
You are an independent evaluator. You did NOT generate the synthesis below. Your job is to assess its quality against explicit success criteria.

## Success Criteria (from Step 1)
expected_output_type: {expected_output_type}
must_answer: {must_answer_items}
must_include: {must_include_items}

## Synthesis to Evaluate
{synthesis_draft}

## Invariant Check Results
{invariant_results_from_step_7b}

## Your Task
For EACH must_answer item:
1. Is it addressed in the synthesis? (YES/NO)
2. Is the answer sourced? (YES/NO/PARTIALLY)
3. Is the source credible (T1-T2)? (YES/NO/MIXED)

For EACH must_include item:
1. Is it present? (YES/NO)
2. Is it accurate and useful? (YES/NO)

## Output Format (max 1,500 tokens)

### Per-Item Verdict
[For each must_answer and must_include: item | verdict | source quality]

### Gaps Identified
[Specific gaps that need filling. Format: gap_description | target_agent | target_query]

### Recommendation
- **PASS** — synthesis meets criteria, proceed to output
- **REFINE** — synthesis has addressable gaps, proceed to Step 8 with listed targets

### Overall Assessment
[1-3 sentences: quality, completeness, confidence level recommendation]

## Output Budget
Return at most 1,500 tokens.

## Task Boundaries
- Do NOT search the web for new evidence.
- Do NOT modify the synthesis.
- Evaluate ONLY against the criteria provided.
- Be honest — a REFINE verdict is more valuable than a false PASS.
```

---

## Step 8: Refinement Prompts (Anti-Sycophancy Protocol)

Targeted gap-filling. Narrower than original prompts — one specific gap per agent.

**Critical: Anti-sycophancy ordering:**
1. First, send the sources and quality rubric — NOT the current draft
2. Ask the agent to evaluate independently what a complete answer looks like
3. Only then, provide the current draft for targeted revision

**Anonymization:** In the "Already Known" section, present claims WITHOUT source URLs first. The agent evaluates on substance before knowing source authority. This prevents authority bias where T1 sources are automatically accepted regardless of claim quality.

### agent-websearch refinement

```
Agent(
  description: "Refine: {gap_short}",
  prompt: "A prior research pass on '{user_question}' identified a gap that needs filling.

## Quality Rubric
The answer must address: {must_answer_items_for_this_gap}
It must include: {must_include_items_for_this_gap}

## Gap to Fill
{gap_description}

## Already Known (claims only — sources withheld to prevent authority bias)
These claims were established: {established_claims_list_without_urls}

## Task
Search specifically for: {target_query}
Do NOT repeat broad research. Focus narrowly on filling this gap.
Return: Key Findings, Sources, Contradictions Detected.

## Output Budget
Return at most 1,500 tokens.",
  subagent_type: "agent-websearch"
)
```

### agent-explore refinement

```
Agent(
  description: "Refine: {gap_short}",
  prompt: "A prior codebase exploration on '{user_question}' missed a specific area.

## Gap to Fill
{gap_description}

## Task
Look specifically for: {target_query}
Focus narrowly. Return findings with file:line references.

## Output Budget
Return at most 1,500 tokens.",
  subagent_type: "agent-explore"
)
```

### docs refinement (ctx7 CLI)

```
Agent(
  description: "Refine: {gap_short}",
  prompt: "A prior documentation lookup on '{user_question}' missed a specific detail.

## Gap to Fill
{gap_description}

## Task
Look up specifically: {target_query}
Use remaining ctx7 CLI calls (bunx ctx7@latest docs {library_id} '{target_query}').
Do NOT modify any files. Return findings with ctx7 library IDs.

## Output Budget
Return at most 1,500 tokens.",
  subagent_type: "agent-docs"
)
```
