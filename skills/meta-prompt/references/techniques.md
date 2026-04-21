# Prompt Engineering Techniques Catalog

Techniques for generating high-quality Claude Code prompts. Each technique includes a pattern you can embed directly in generated prompts.

## 1. Clarity and Directness

Be specific about what to build. Treat the prompt as a specification — no implied knowledge, no ambiguity.

**Principles:**
- Use numbered steps when order matters
- State what you want first, then complement with pitfalls to avoid
- Every instruction should be concrete enough to verify

**Pattern:**
```
Weak: "Create a good dashboard"
Strong: "Create an analytics dashboard with: (1) a sidebar navigation with 5 sections, (2) a main content area with a 3-column card grid showing KPIs, (3) a line chart for revenue over time using Recharts, (4) a data table with sorting and pagination for recent transactions"
```

## 2. XML Tag Structuring

XML tags organize prompts into unambiguous sections when mixing instructions, context, and examples. Use consistently — do not mix XML and Markdown delimiters in the same prompt.

**Standard tag vocabulary:**
- `<context>` — Background info, project state, existing codebase
- `<requirements>` — Functional requirements (numbered)
- `<stack>` — Technologies, frameworks, libraries with versions
- `<constraints>` — Rules and limitations with motivations
- `<verification>` — Concrete checks to validate the output
- `<examples>` / `<example>` — Demonstration of desired output
- `<plan>` — Architecture planning instruction

**Pattern:**
```xml
<context>
Existing Next.js 15 app with App Router, Tailwind CSS v4, and Clerk auth.
TypeScript strict mode, Drizzle ORM with PostgreSQL on Neon.
</context>

<requirements>
1. Add a user settings page at /dashboard/settings
2. Include profile editing (name, avatar, bio)
3. Add notification preferences (email, push, in-app)
4. Include account deletion with confirmation modal
</requirements>

<constraints>
- Use existing UI components from /components/ui — avoids style fragmentation
- Use server actions for mutations — enables progressive enhancement and type-safe form handling
- Mobile-first responsive — 60%+ of users are on mobile
</constraints>

<verification>
- `next build` succeeds with zero errors
- All forms submit and persist data correctly
- Pages render correctly at 375px, 768px, 1440px
- Account deletion requires confirmation and cascades properly
</verification>
```

## 3. Role Assignment

A single sentence focusing Claude's expertise is sufficient. Elaborate multi-paragraph personas add noise without improving output.

**Pattern:**
```
You are a senior full-stack engineer specializing in Next.js App Router, TypeScript, and Tailwind CSS.
```

**Effective roles by domain:**
- "Senior full-stack engineer" — web apps
- "Staff backend engineer" — APIs, databases, system design
- "Principal frontend engineer" — UI/UX, design systems
- "DevOps/SRE engineer" — infrastructure, CI/CD
- "Security engineer" — auth, data protection

## 4. Few-Shot Examples

Examples are the most reliable way to steer output format, tone, and structure. Include 1-2 examples by default when output format matters; use 3-5 for complex or subtle patterns.

**When to include:**
- Output format is specific (JSON schema, component API, file structure)
- Tone/style matters (error messages, UI copy)
- Behavior has subtle rules (validation logic, routing patterns)

**Rules:**
- Formatting must be identical across all examples (whitespace, tags, delimiters)
- Show the pattern, not just one case — cover at least one edge case
- Wrap in `<examples>` / `<example>` tags

## 5. Planning Before Code

For complex builds, instruct Claude to plan the architecture before writing code. A good plan enables one-shot implementation.

**Pattern:**
```
Before writing any code, think through:
1. The data model — entities, relationships, and constraints
2. The page routes and their purposes
3. The component hierarchy and how pieces compose
4. Edge cases and error states

Then implement based on your analysis.
```

**For Full-level prompts, use a `<plan>` tag:**
```xml
<plan>
Before implementing, plan the architecture. Think through the data model, page routes, component hierarchy, and state management approach. Then build systematically.
</plan>
```

## 6. Task Decomposition

Break complex projects into sequential phases. Each phase produces a testable milestone.

**Pattern:**
```
Implement in phases:

Phase 1 — Data Layer:
- Define the database schema with Drizzle ORM
- Create migration files
- Set up seed data with realistic content

Phase 2 — Server Layer:
- Implement server actions for CRUD operations
- Add input validation with Zod
- Handle errors with typed responses

Phase 3 — UI Layer:
- Build page layouts and components
- Wire up forms to server actions
- Add loading and error states

Phase 4 — Verification:
- Run `next build` — fix all errors
- Test responsive at 375px, 768px, 1440px
- Check accessibility (ARIA labels, keyboard nav)
```

## 7. Positive Directives with Pitfall Complement

State what to do first. Then add a short list of specific pitfalls to avoid. This produces better generalization than pure negative constraints — Claude extrapolates better from motivations than from prohibition lists.

**Pattern:**
```xml
<constraints>
- Keep components focused — each handles one concern
- Use realistic sample data that fits the domain
- Design a distinctive visual identity for this specific product
- Use React Server Components by default, client components only for interactivity
</constraints>

Avoid these specific pitfalls:
- Placeholder content ("Lorem ipsum", "Coming soon", generic user avatars)
- Generic AI aesthetic (purple gradients, Inter/Roboto fonts, card-heavy layouts)
- Abstractions for one-time operations
- Features not explicitly requested
```

## 8. Verification Loop

The highest-leverage quality multiplier. Every prompt should include concrete checks Claude runs to validate its output. This creates a feedback loop that 2-3x the quality of the final result.

**Pattern for web apps:**
```xml
<verification>
After implementing, verify:
- `next build` (or equivalent) succeeds with zero errors and zero warnings
- All pages render correctly at 375px, 768px, and 1440px
- Interactive elements have hover/focus/active states
- Forms submit correctly and show validation errors for invalid input
- Empty states are handled for lists and tables with no data
- Accessibility: proper ARIA labels, keyboard navigation works, color contrast passes
</verification>
```

**Pattern for APIs:**
```xml
<verification>
- All endpoints return correct status codes (200, 201, 400, 401, 404)
- Invalid input returns 400 with descriptive error messages
- Unauthorized requests return 401
- API docs are generated and accessible at /docs
- Rate limiting triggers at configured thresholds
</verification>
```

**Pattern for CLI tools:**
```xml
<verification>
- All commands work with valid input
- Invalid input produces helpful, non-technical error messages
- `--help` output is comprehensive for every command
- Exit codes: 0 for success, 1 for errors
- Config file is created on first run if missing
</verification>
```

## 9. Outcome-Based Prompting

For tasks on existing codebases, give the goal and let Claude investigate — this often outperforms prescriptive step-by-step instructions. State what success looks like, not the exact steps to get there.

**Pattern:**
```
The checkout flow currently fails when a user applies a discount code after adding items.
Fix this so that: (1) discount codes apply correctly regardless of when they're entered,
(2) the cart total updates in real-time, (3) the existing test suite passes.

Read the relevant code first to understand the current implementation before making changes.
```

**When to use:** Bug fixes, refactors, feature additions on existing code. When Claude needs investigative freedom.

**When NOT to use:** Greenfield builds where you want a specific architecture. Use prescriptive decomposition instead.

## 10. Context Window Optimization

How you structure context affects output quality. Place reference material at the top and the task instruction at the bottom.

**Pattern:**
```xml
<existing-code>
[paste relevant existing code here]
</existing-code>

<existing-schema>
[paste database schema here]
</existing-schema>

Based on the code and schema above, implement the following:
[specific instructions]
```

For long prompts with critical instructions, place key constraints at both the beginning and end — instructions near the end take precedence when conflicts exist.

## 11. Grounding and Anti-Hallucination

Reduce hallucinations by requiring Claude to verify before acting.

**Pattern:**
```
Before making changes:
1. Read existing files to understand the current architecture
2. Check what dependencies are already installed
3. Follow patterns established in the existing codebase
4. Verify that libraries and APIs exist before using them — do not invent packages
```

## 12. Scope Control

Prevent Claude from expanding beyond the request. Scope creep is one of the most common failure modes.

**Pattern:**
```
Scope: Only modify the files listed below. Do not create new files, refactor existing code,
or add features beyond what is explicitly requested.

Files to modify:
- src/app/dashboard/page.tsx
- src/components/ui/chart.tsx
- src/lib/db/queries.ts
```

## 13. Realistic Data and Design

Prevent generic AI output by specifying domain-appropriate data and visual identity.

**Pattern:**
```
Use realistic, domain-appropriate sample data — never "Lorem ipsum" or "John Doe".
For an ecommerce site: real-sounding product names, realistic prices, varied descriptions.

Visual identity:
- Choose a distinctive font from Google Fonts (avoid Inter, Roboto, Space Grotesk, Arial)
- Use a cohesive color palette with a strong primary color and sharp accents
- Add micro-interactions and transitions
- Design for the specific domain — a fintech app looks different from a social network
```

## 14. Agentic Reminders

For full-app builds, three system-prompt reminders improve completion quality. Include these in Full-level prompts.

**Pattern:**
```
1. Persistence: Keep working until the entire implementation is complete. Do not stop with partial features or TODO comments.
2. Anti-hallucination: Verify that all libraries, APIs, and imports exist before using them. Do not guess.
3. Planning: Think through the architecture before writing code. Plan the data model and component hierarchy first.
```

## 15. WHY Over WHAT

Explain the motivation behind constraints so Claude can generalize beyond the literal rule. A single explained constraint is more effective than multiple unexplained ones.

**Pattern:**
```
Weak: "Use server actions for mutations"
Strong: "Use server actions for mutations — they enable progressive enhancement, type-safe form handling, and eliminate the need for manual API route boilerplate"

Weak: "No client-side state management"
Strong: "Avoid client-side state management libraries — React Server Components handle data fetching natively with less complexity and better performance"
```

## 16. Error Handling Strategy

Define the error handling approach rather than leaving it to defaults.

**Pattern:**
```
Error handling:
- Use typed error responses with error codes, not generic Error throws
- Show user-friendly messages in the UI (toasts for actions, inline for forms)
- Log detailed errors server-side for debugging
- Handle network failures with retry logic where appropriate
- Include loading states for all async operations
- Include empty states for lists/tables with no data
```

## 17. Security Requirements

Embed security awareness into the prompt for any app handling user data.

**Pattern:**
```
Security:
- Validate all user input on the server — client-side validation is for UX only
- Use parameterized queries for all database operations
- Sanitize user-generated content before rendering
- Use HttpOnly cookies for sensitive tokens
- Apply rate limiting to authentication endpoints
- Follow the principle of least privilege for database queries
```

## 18. Self-Critique

For complex builds, instruct Claude to review its own output against the original constraints before finalizing. This catches mismatches between intent and implementation.

**Pattern:**
```
After completing the implementation, review your work:
1. Does each requirement from the spec have a corresponding implementation?
2. Do the constraints hold — no violations of the rules above?
3. Are there any hardcoded values that should be configurable?
4. Would a new developer understand this code without additional context?
```
