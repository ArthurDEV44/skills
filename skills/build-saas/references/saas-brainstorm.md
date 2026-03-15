# SaaS Brainstorm Protocols — Agent Prompts and Question Patterns

## Phase 2 — Research Prompt Template

```
Research the market and competitive landscape for a SaaS product.

## SaaS Idea
{user_saas_description}

## Domain
{detected_domain}

## SaaS Type
{B2B | B2C | B2B2C | marketplace | platform | tool}

## Research Priorities (IN ORDER)

1. **Direct Competitors** — Find 3-5 SaaS products that solve the same or similar problem.
   For each: name, URL, pricing model, price points, key features, target user,
   estimated size (users/revenue if public), strengths, weaknesses.

2. **Indirect Competitors** — Find 2-3 products that solve the problem differently
   (e.g., spreadsheets, manual processes, different tools cobbled together).

3. **Market Sizing** — TAM (total addressable market), SAM (serviceable), SOM (obtainable).
   Look for industry reports, growth rates, trends.

4. **Pricing Intelligence** — How do competitors price? Per-seat, usage-based, flat?
   What's the average price point? What tier structure is common?

5. **Feature Expectations** — What features are table stakes (every competitor has them)?
   What are differentiators? What's missing from the market?

6. **User Feedback** — Search for reviews, complaints, feature requests about competitors.
   What do users love? What do they hate? What's missing?

7. **Technical Landscape** — How are similar products built? Any known tech stacks?
   What APIs or integrations are expected?

8. **Monetization Patterns** — What conversion rates are typical? Freemium vs trial?
   What's the typical LTV and churn rate for this type of SaaS?

9. **Common Failure Modes** — Why do SaaS products in this domain fail?
   Technical debt, pricing, competition, market timing?

## Search Strategy
- "{domain} SaaS competitors comparison 2025 2026"
- "{domain} software market size"
- "{competitor_name} pricing" (for each identified competitor)
- "{domain} SaaS common mistakes"
- "{domain} software reviews complaints"
- "best {domain} tools {year}"

## Output Requirements

### Direct Competitors
[For each: name, pricing, features, target user, strengths, weaknesses]

### Indirect Competitors
[Alternative solutions users currently use]

### Market Data
[Size, growth, trends]

### Pricing Landscape
[Models, price points, tier structures]

### Feature Matrix
[Table stakes vs differentiators vs missing]

### User Feedback Themes
[What users want, what frustrates them]

### Technical Patterns
[How competitors are built, integrations expected]

### Failure Modes
[Why similar products fail]

### Sources
[All URLs]
```

---

## Phase 2b — Codebase Exploration (if existing project)

```
Explore the existing codebase to understand current state and constraints.

## Context
We're planning a SaaS product: {description}

## Exploration Tasks
1. What's the current tech stack? (framework, language, dependencies)
2. Is there existing auth? What provider?
3. Is there a database? What ORM/driver?
4. What's the project structure? (monorepo, app router, etc.)
5. Are there existing components/UI patterns?
6. What tests exist? What framework?
7. Are there existing PRDs or specs?
8. What environment variables are configured?

## Output
[Tech stack, structure, patterns, constraints — with file:line refs]
```

---

## Research-Informed Question Patterns

### The SaaS-Specific Rule

For SaaS brainstorming, questions fall into 4 categories:

1. **Market position** (informed by competitor research)
2. **Business model** (informed by pricing research)
3. **Feature scope** (informed by feature matrix)
4. **Architecture** (informed by stack defaults + feature decisions)

Categories 1-3 are genuinely open. Category 4 is mostly predetermined — only the open decisions from stack-defaults.md are asked.

---

### Round 1 — Vision & Market Position

```markdown
## {SaaS Idea} — Market Research Results

{Present research summary: competitors, market, pricing}

---

**Let's shape your vision based on what we found:**

1. The market currently has {N} notable players:
   - {Competitor A}: {brief — strength + weakness}
   - {Competitor B}: {brief — strength + weakness}

   Where do you position yourself?
   A. Direct competitor to {A} — better {specific advantage}
   B. Niche player — focus on {underserved segment from research}
   C. Platform play — {broader vision}
   D. Different positioning — [describe]

2. Users reviewing competitors frequently mention: "{specific complaint from research}"
   Is solving this pain point central to your product?
   A. Yes — core value proposition
   B. One of several value props
   C. No — my core value is different: [describe]

3. The market is {growing at X% | stable | consolidating}.
   {Market-specific question based on trends}
   A. {Option informed by trend}
   B. {Counter-option}
   C. {Other}
```

---

### Round 2 — Business Model

```markdown
Based on your positioning, let's define the business model:

**Competitor pricing for reference:**
| Competitor | Model | Free Tier | Starter | Pro | Enterprise |
|------------|-------|-----------|---------|-----|-----------|
| {A} | {model} | {yes/no} | ${X}/mo | ${Y}/mo | Custom |
| {B} | {model} | {yes/no} | ${X}/mo | ${Y}/mo | Custom |

1. Pricing model (research shows {model} is most common in {domain}):
   A. Freemium — {when research says this works, conversion rate benchmarks}
   B. Free trial ({N} days) → paid — {when this works}
   C. Usage-based — {when this works}
   D. Per-seat (B2B) — {when this works}
   E. Flat rate — {when this works}

2. Price positioning:
   A. Below market ({specific price range}) — compete on price
   B. Market rate ({specific price range}) — compete on features/UX
   C. Premium ({specific price range}) — compete on quality/brand
   D. Let me decide later

3. Payment provider:
   A. **Stripe** — best for: complex billing, usage-based, B2B invoicing, existing Stripe infra
   B. **Lemon Squeezy** — best for: simplicity, automatic tax/VAT handling, indie/solo, global sales

4. Teams/organizations from day one?
   A. Yes — Clerk Organizations for multi-tenant (B2B play)
   B. Later — individual users first, teams in v2
   C. Never — purely individual SaaS
```

---

### Round 3 — Feature Scope (MoSCoW)

```markdown
Based on competitor analysis and your vision, here's the feature matrix:

**Table Stakes** (every competitor has these):
{list features marked as table stakes from research}

**Differentiators** (some competitors have these):
{list features that differentiate competitors}

**Missing from Market** (nobody does this well):
{list gaps identified from research}

---

Rate each with MoSCoW (Must/Should/Could/Won't):

| # | Feature | Market Context | Your Priority |
|---|---------|---------------|---------------|
| 1 | {Feature} | Table stakes — all competitors have it | |
| 2 | {Feature} | {Competitor A}'s key differentiator | |
| 3 | {Feature} | Market gap — no one does this well | |
| 4 | {Feature} | Users frequently request this | |
| ... | ... | ... | |

Any features I missed that you want to add?
```

---

### Round 4 — Architecture (stack-constrained)

Only questions from the "Decisions That Remain Open" table in stack-defaults.md:

```markdown
Technical decisions (your stack is already defined — see below):

**Locked:** Next.js, TypeScript, Tailwind, bun, Clerk, Neon, Drizzle,
CossUI + BaseUI + shadcn/ui, Tabler Icons, Vitest, Zod, Resend + React Email

**Open decisions based on your features:**

1. Separate backend needed?
   Based on your Must-Have features, I {see / don't see} a need for a separate backend.
   {If see: "{specific feature} requires {heavy computation | WebSockets | etc.}"}
   A. No — Next.js API routes + Server Actions handle everything (recommended for your scope)
   B. Yes — {detected reason} → Rust (Axum) | Go | Python | C# | C/C++

2. Data layer:
   {Based on features requiring real-time or caching}
   A. Neon + Drizzle only (sufficient for your features)
   B. Add Upstash Redis (you need: {rate limiting | caching | queues})
   C. Use Convex for real-time (you need: {live sync | collaboration})
   D. Drizzle + Upstash + Convex (complex data needs)

3. Desktop app?
   A. No — web only (recommended unless offline/native is core)
   B. Yes — Tauri for {specific use case}
   C. Plan for later — architect but don't implement

4. 3D / advanced visuals?
   A. No — standard web UI
   B. Landing page only — R3F for marketing visuals
   C. Core product feature — R3F integrated in the app
```

---

### Quality Gates (MANDATORY)

```markdown
Quality gates for every story (bun-based):

Recommended default:
- `bun run typecheck` — TypeScript strict checking
- `bun run lint` — ESLint
- `bun run test` — Vitest

Accept this default?
A. Yes — all three gates
B. Add more: [specify]
C. Remove test gate for non-logic stories
D. Custom: [specify]
```

---

### Devil's Advocate (MANDATORY)

```markdown
Before finalizing, let me challenge the concept:

1. **Competition:** {Competitor A} has {X} funding / {Y} users / {Z} years head start.
   Your edge is {stated differentiator}. Is this defensible?
   → What happens if {Competitor A} copies your differentiator?

2. **Pricing:** You're pricing at {price} while {Competitor B} offers {comparison}.
   Research shows {domain} SaaS has {X}% typical churn.
   → Is your pricing sustainable given expected acquisition cost?

3. **Scope:** The MVP has {N} stories across {M} epics.
   Based on typical AI-agent velocity (~3-5 stories/day well-scoped):
   → This is roughly {estimate} of development. Comfortable?

4. **Technical:** {Specific technical concern based on chosen architecture}
   → {Risk and potential mitigation}

5. **Missing:** Research mentioned {thing we haven't addressed}.
   → Is this relevant? Should we add a story for it?
```

---

## Compressed Research Format

For passing to downstream phases:

```markdown
## SaaS Research Brief

### Market Position
- Top competitor: {name} at ${price}/mo, {N} users
- Market size: {TAM}
- Our positioning: {differentiation}

### Feature Set (validated)
Must Have: {list}
Should Have: {list}
Won't Have: {list}

### Business Model
- Model: {freemium | trial | usage-based}
- Price: {range}
- Payment: {Stripe | Lemon Squeezy}

### Architecture
- Type: {monolith | monolith + backend}
- Backend: {if separate: language}
- Real-time: {yes/no}
- Cache: {yes/no}
- Desktop: {yes/no}

### Risks
- {Risk 1}: {mitigation}
- {Risk 2}: {mitigation}
```
