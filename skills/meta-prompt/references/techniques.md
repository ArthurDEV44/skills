# Prompt Engineering Techniques Catalog

Consolidated best practices from Anthropic, OpenAI, and Google research, optimized for Claude Code prompt generation.

## 1. Clarity and Directness

The single most impactful technique. Claude responds best to clear, explicit instructions.

**Principles:**
- Be specific about desired output format and constraints
- Use numbered steps when order matters
- State what you WANT, not what you DON'T want (then add anti-patterns separately)
- Think of the prompt as a spec for a new employee: no implied knowledge

**Pattern:**
```
WEAK: "Create a good dashboard"
STRONG: "Create an analytics dashboard with: (1) a sidebar navigation with 5 sections, (2) a main content area with a 3-column card grid showing KPIs, (3) a line chart for revenue over time using Recharts, (4) a data table with sorting and pagination for recent transactions"
```

**Claude Code specific:** Add modifiers like "Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation." to unlock Claude's full capability.

## 2. XML Tag Structuring

XML tags eliminate ambiguity when prompts mix instructions, context, examples, and data.

**Essential tags for Claude Code prompts:**
- `<context>` — Background info, project state, existing codebase description
- `<requirements>` — What must be built (functional requirements)
- `<stack>` — Technologies, frameworks, libraries, versions
- `<constraints>` — Rules, limitations, performance targets
- `<examples>` / `<example>` — Demonstration of desired output
- `<anti-patterns>` — What to explicitly avoid
- `<acceptance-criteria>` — Definition of done

**Pattern:**
```xml
<context>
This is an existing Next.js 15 app with App Router, Tailwind CSS v4, and Clerk auth already configured.
The project uses TypeScript strict mode and Drizzle ORM with PostgreSQL on Neon.
</context>

<requirements>
1. Add a user settings page at /dashboard/settings
2. Include profile editing (name, avatar, bio)
3. Add notification preferences (email, push, in-app)
4. Include account deletion with confirmation modal
</requirements>

<constraints>
- Use existing UI components from the project's /components/ui directory
- All forms must use server actions, not API routes
- Must be fully responsive (mobile-first)
- No client-side state management libraries — use React Server Components where possible
</constraints>
```

## 3. Role Assignment (Persona)

Setting a specific expert persona focuses Claude's behavior and activates domain-specific knowledge.

**Pattern:**
```
You are a senior full-stack engineer specializing in Next.js App Router, TypeScript, and Tailwind CSS. You write production-ready, type-safe code with proper error handling. You follow the principle of least surprise and prefer simplicity over abstraction.
```

**Effective roles for Claude Code:**
- "Senior full-stack engineer" — web apps
- "Staff backend engineer" — APIs, databases, system design
- "Principal frontend engineer" — UI/UX, design systems
- "DevOps/SRE engineer" — infrastructure, CI/CD, deployment
- "Security engineer" — auth, data protection, OWASP compliance

## 4. Few-Shot Examples

Examples are the most reliable way to steer output format, tone, and structure.

**Rules:**
- 3-5 examples for best results
- Make examples diverse (cover edge cases)
- Wrap in `<examples>` / `<example>` tags
- Show the PATTERN, not just one case

**When to use in meta-prompts:**
- When the output format is specific (JSON schema, component API, file structure)
- When tone/style matters (error messages, UI copy, commit messages)
- When the behavior has subtle rules (validation logic, routing patterns)

## 5. Chain of Thought (CoT)

Encourage step-by-step reasoning for complex decisions.

**Pattern:**
```
Before implementing, think through the following:
1. What is the data model? Define entities and relationships
2. What are the page routes and their purposes?
3. What components are needed and how do they compose?
4. What are the edge cases and error states?
Then implement based on your analysis.
```

**Claude Code specific:** Claude 4.6 has adaptive thinking built in. For complex prompts, use phrases like "Think through the architecture before writing code" or "Plan the file structure first, then implement."

## 6. Task Decomposition

Break complex projects into sequential phases.

**Pattern:**
```
Implement this in phases:

Phase 1 — Data Layer:
- Define the database schema with Drizzle ORM
- Create migration files
- Set up seed data

Phase 2 — API Layer:
- Implement server actions for CRUD operations
- Add input validation with Zod
- Handle errors with typed error responses

Phase 3 — UI Layer:
- Build page layouts and components
- Wire up forms to server actions
- Add loading and error states

Phase 4 — Polish:
- Add animations and transitions
- Implement responsive design
- Run through accessibility checklist
```

## 7. Anti-Pattern Specification

Explicitly state what Claude should NOT do. This is critical for Claude Code, which can over-engineer.

**Essential anti-patterns to include:**
```xml
<anti-patterns>
- DO NOT over-engineer: no abstractions for one-time operations
- DO NOT add features that weren't requested
- DO NOT use placeholder/dummy data — use realistic sample data
- DO NOT create generic "AI-looking" UI (no purple gradients on white, no Inter font)
- DO NOT add unnecessary error handling for impossible scenarios
- DO NOT create helper utilities for single-use operations
- DO NOT add comments to self-explanatory code
- DO NOT create README or documentation files unless requested
- DO NOT use deprecated APIs or patterns
</anti-patterns>
```

## 8. Output Format Specification

Define exactly what Claude should produce.

**Pattern for file-based output:**
```
Create the following file structure:
src/
  app/
    dashboard/
      page.tsx        — Main dashboard with KPI cards and charts
      layout.tsx      — Dashboard layout with sidebar nav
      settings/
        page.tsx      — User settings form
    api/
      webhooks/
        stripe/
          route.ts    — Stripe webhook handler
  components/
    ui/
      data-table.tsx  — Reusable data table with sorting/filtering
      chart.tsx       — Line chart wrapper component
  lib/
    db/
      schema.ts       — Drizzle schema definitions
      queries.ts      — Database query functions
```

## 9. Context Window Optimization

For Claude Code, how you structure context matters.

**Rules:**
- Place long reference material (existing code, docs) at the TOP of the prompt
- Place the actual instruction/query at the BOTTOM
- This improves response quality by up to 30% (per Anthropic testing)
- Use XML tags to clearly separate reference material from instructions

**Pattern:**
```xml
<existing-code>
[paste relevant existing code here]
</existing-code>

<existing-schema>
[paste database schema here]
</existing-schema>

Now, based on the existing code and schema above, implement the following new feature:
[specific instructions]
```

## 10. Grounding and Verification

Reduce hallucinations by requiring Claude to verify before acting.

**Pattern:**
```
Before making changes:
1. Read the existing files in the project to understand the current architecture
2. Check what dependencies are already installed
3. Follow the patterns established in the existing codebase
4. Do not invent APIs or libraries — verify they exist first
```

## 11. Iterative Refinement Instructions

Tell Claude how to self-correct.

**Pattern:**
```
After implementing:
1. Review your code for type safety — fix any TypeScript errors
2. Check that all imports resolve to real modules
3. Verify responsive design works at mobile (375px), tablet (768px), desktop (1280px)
4. Test that all interactive elements have proper hover/focus/active states
5. Ensure no accessibility issues (proper ARIA labels, keyboard navigation, color contrast)
```

## 12. Scope Control

Prevent Claude from expanding beyond the request.

**Pattern:**
```
SCOPE: Only modify the files listed below. Do not create new files, refactor existing code,
or add features beyond what is explicitly requested.

Files to modify:
- src/app/dashboard/page.tsx
- src/components/ui/chart.tsx
- src/lib/db/queries.ts
```

## 13. Parallel Tool Use Optimization

For Claude Code specifically, enable efficient tool usage.

**Pattern (included in system prompts):**
```
When exploring the codebase, read multiple files in parallel rather than one at a time.
When creating multiple independent files, write them in parallel.
Prioritize understanding the existing codebase structure before making changes.
```

## 14. Realistic Data and Design

Prevent the "AI slop" aesthetic.

**Pattern:**
```
Use realistic, contextually appropriate sample data — never placeholder text like "Lorem ipsum"
or generic names like "John Doe". For an ecommerce site, use real-sounding product names,
realistic prices, and varied product descriptions.

For UI design:
- Choose a distinctive font from Google Fonts (not Inter, Roboto, or Arial)
- Use a cohesive color palette with a strong primary color and sharp accents
- Add micro-interactions and transitions that feel polished
- Design for the specific domain — a fintech app should look different from a social network
```

## 15. Error Handling Strategy

Define how errors should be handled rather than leaving it implicit.

**Pattern:**
```
Error handling strategy:
- Use typed error responses (never throw generic Error)
- Show user-friendly error messages in the UI (toast notifications for actions, inline for forms)
- Log detailed errors server-side for debugging
- Handle network failures gracefully with retry logic where appropriate
- Add proper loading states for all async operations
- Include empty states for lists/tables with no data
```

## 16. Security-First Instructions

Embed security awareness into the prompt.

**Pattern:**
```
Security requirements:
- Validate all user input on the server (never trust client-side validation alone)
- Use parameterized queries (Drizzle handles this, but verify)
- Implement proper CSRF protection for form submissions
- Sanitize any user-generated content before rendering
- Use HttpOnly cookies for sensitive tokens
- Apply rate limiting to authentication endpoints
- Follow the principle of least privilege for database queries
```
