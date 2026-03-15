---
model: opus
name: build-saas
description: "End-to-end SaaS creation workflow from idea to PRD. Researches via /meta-code, brainstorms with research-backed questions, validates the concept, defines architecture using the user's preferred tech stack, and generates a complete PRD with epics, stories, and status tracking. Invoke with /build-saas [saas idea description]."
argument-hint: "[saas idea description]"
---

# build-saas — SaaS From Idea to PRD

Build a SaaS for: $ARGUMENTS

## Overview

Full SaaS creation pipeline that takes a raw idea through research-informed brainstorming, concept validation, architecture definition, and complete PRD generation. Uses the user's predefined tech stack (see [references/stack-defaults.md](references/stack-defaults.md)) and only asks questions where genuine decisions remain.

**Key principles:**
- Research before brainstorming — every question backed by competitor/market data
- Stack is predefined — don't ask "what framework?", ask "does this need a separate backend?"
- Validate before committing — check market fit, feasibility, differentiation
- SaaS-native thinking — auth, billing, multi-tenancy, and onboarding are first-class concerns

**Outputs:**
1. Complete PRD with SaaS-specific sections (`./tasks/prd-[name].md`)
2. JSON status tracking file (`./tasks/prd-[name]-status.json`)
3. Architecture decision record within the PRD

## Execution Flow

```
$ARGUMENTS -> [saas idea]
       |
       v
+---------------+
|  Phase 1:     |
|   INTAKE      |  <- Parse idea, detect SaaS type and domain
+-------+-------+
        |
        v
+---------------+
|  Phase 2:     |
|  RESEARCH     |  <- /meta-code: competitors, market, pricing, tech
+-------+-------+
        | research synthesis
        v
+---------------+
|  Phase 3:     |
|  BRAINSTORM   |  <- Research-informed Q&A (3-5 rounds)
|  (decisions)  |  <- Only genuinely open questions
+-------+-------+
        | decisions locked
        v
+---------------+
|  Phase 4:     |
|  VALIDATE     |  <- Concept check: market, feasibility, differentiation
+-------+-------+
        | concept validated
        v
+---------------+
|  Phase 5:     |
|  ARCHITECTURE |  <- Apply stack defaults + open decisions
+-------+-------+
        | architecture defined
        v
+---------------+
|  Phase 6:     |
|  WRITE PRD    |  <- Full SaaS PRD + status.json
+-------+-------+
        |
        v
+---------------+
|  Phase 7:     |
|  FINALIZE     |  <- User review, adjustments, save
+-------+-------+
```

## Phase-by-Phase Execution

### Phase 1 — INTAKE

**1a. Parse the idea:**

Extract from `$ARGUMENTS`:
- **Domain:** What problem space? (productivity, finance, health, dev-tools, etc.)
- **SaaS type:** B2B, B2C, B2B2C, marketplace, platform, tool
- **Keywords:** For competitor and market research

**1b. Classify SaaS dimensions:**

| Dimension | Detection |
|-----------|-----------|
| Multi-tenant? | B2B with orgs/teams → yes |
| Real-time? | Chat, collab, live data → yes |
| Data-heavy? | Analytics, processing, ML → yes |
| Content platform? | UGC, media, publishing → yes |
| API-first? | Developer tools, integrations → yes |
| Desktop needed? | Offline use, system access → maybe Tauri |
| 3D/visual? | Design tools, data viz, tech branding → maybe R3F |

**1c. Check for existing project:**

```
Glob: package.json
Glob: tasks/prd-*.md
```

If existing project found, note constraints and avoid conflicts.

**GATE:** Domain and SaaS type identified.

---

### Phase 2 — RESEARCH (mandatory /meta-code pipeline)

**2a. Spawn agent-websearch:**

```
Agent(
  description: "Research {domain} SaaS landscape",
  prompt: <see references/saas-brainstorm.md — Research prompt>,
  subagent_type: "agent-websearch"
)
```

Research focus (SaaS-specific):
1. **Competitors** — Top 5 competitors, their pricing, features, tech stack if known
2. **Market sizing** — TAM/SAM/SOM, growth trajectory, trends
3. **Pricing models** — What competitors charge, freemium vs paid, per-seat vs usage-based
4. **User expectations** — What's table stakes, what's a differentiator
5. **Technical patterns** — How similar SaaS products are built
6. **Monetization** — What works in this domain (Stripe vs Lemon Squeezy patterns)
7. **Common failures** — Why SaaS products in this domain fail

Wait for completion. Compress to <500 words.

**2b. Spawn agent-explore + agent-docs in parallel (if applicable):**

```
// If codebase exists:
Agent(
  description: "Explore existing codebase",
  prompt: <explore current architecture, patterns, dependencies>,
  subagent_type: "agent-explore"
)

// If key libraries identified:
Agent(
  description: "Fetch docs for {libraries}",
  prompt: <check latest APIs for Clerk, Stripe, Drizzle, etc.>,
  subagent_type: "agent-docs"
)
```

**2c. Synthesize research brief** — organized for brainstorming questions.

**GATE:** Research complete. Competitor and market data available.

---

### Phase 3 — BRAINSTORM (research-informed)

Every question references research findings. Load stack defaults from [references/stack-defaults.md](references/stack-defaults.md) — do NOT ask about predetermined choices.

**3a. Present research summary:**

```markdown
## Research: {SaaS Idea} Market

### Competitors
- **{Competitor A}:** {pricing}, {features}, {users}, {strengths/weaknesses}
- **{Competitor B}:** {pricing}, {features}, {users}, {strengths/weaknesses}
- **Market gap:** {unmet need}

### Market
- Size: {TAM/SAM/SOM if found}
- Trend: {growing/stable/declining}
- Key insight: {what research revealed}

### Pricing Landscape
- {Competitor A}: {model and price points}
- {Competitor B}: {model and price points}
- Average: {typical pricing for this type of SaaS}

*{N} sources consulted*
```

**3b. Round 1 — Vision & Positioning:**

```
Based on our research:

1. {Competitor A} targets {audience A}, while {Competitor B} serves {audience B}.
   Who is YOUR primary user?
   A. {User type A} — {market size from research}
   B. {User type B} — {market size from research}
   C. Both, segmented by {criterion}
   D. Different audience — [describe]

2. The main gap in the market is: {gap from research}.
   Is this your core differentiator?
   A. Yes — build around this gap
   B. Partially — my differentiator is actually [describe]
   C. No — I'm competing on {execution/price/UX/other}

3. Research shows {trend} in this space.
   How does this affect your timeline?
   A. Launch fast to capture first-mover advantage
   B. Take time to build a robust product
   C. MVP first, iterate based on feedback
```

**3c. Round 2 — Business Model (informed by pricing research):**

```
Competitors price their products like this:
- {Competitor A}: {model} at {price points}
- {Competitor B}: {model} at {price points}

1. Pricing model:
   A. Freemium (free tier + paid plans) — {when research says this works}
   B. Free trial → paid (no free tier) — {when this works}
   C. Usage-based pricing — {when this works}
   D. Flat rate — {when this works}
   E. Per-seat/team pricing — {when this works for B2B}

2. Payment processor:
   A. Stripe — full control, webhooks, complex billing, usage-based
   B. Lemon Squeezy — simpler, merchant of record (handles tax/VAT), better for indie

3. Do you plan to offer team/organization features?
   A. Yes from day one (Clerk Organizations)
   B. Yes but for a later version
   C. No, individual users only
```

**3d. Round 3 — Core Features & MVP:**

```
Based on your vision and what's standard in the market:

Here are the capabilities I've identified. Rate each:

| # | Capability | Market Context | M/S/C/W? |
|---|-----------|---------------|----------|
| 1 | {Feature} | {all competitors have this} | |
| 2 | {Feature} | {differentiator} | |
| 3 | {Feature} | {nice-to-have, {Competitor A} has it} | |
| ... | ... | ... | |

M = Must Have, S = Should Have, C = Could Have, W = Won't Have
```

**3e. Round 4 — Architecture Decision:**

This is the ONLY technical round because the stack is predefined. The question is:

```
Based on the features you selected:

1. Does this SaaS need a separate backend?
   (Next.js API routes handle most cases. A separate backend is needed for:
    heavy computation, WebSocket servers, background job processing,
    system-level access, or extreme performance requirements)

   A. No — Next.js handles everything (Vercel deployment)
   B. Yes — needs {detected reason from features}
      → Which language?
      a. Rust (performance-critical, systems-level)
      b. Go (concurrent services, microservices)
      c. Python (ML/AI, data processing)
      d. C#/.NET (enterprise integrations)
      e. C/C++ (extreme performance, embedded)

2. Data needs:
   A. Standard relational (Neon + Drizzle is sufficient)
   B. Need caching/rate-limiting (add Upstash Redis)
   C. Need real-time sync (consider Convex instead of/alongside Drizzle)
   D. B + C (Upstash + Convex)

3. Does this need a desktop app?
   A. Yes — Tauri (offline access, system integration)
   B. No — web only
   C. Later — plan for it but don't build yet

4. Does this need 3D/advanced visuals?
   A. Yes — React Three Fiber for {specific use case}
   B. No — standard web UI
   C. Only for landing page / marketing
```

**3f. Quality Gates (MANDATORY):**

```
Quality commands (bun-based, as per stack defaults):

1. Standard gates:
   A. bun run typecheck && bun run lint (default)
   B. Additional: bun run test (Vitest)
   C. Both A + B
   D. Custom: [specify]

2. For UI stories, include visual verification?
   A. Yes, verify in browser
   B. No, automated tests sufficient
```

**3g. Devil's Advocate (MANDATORY):**

```
Before we finalize, research flagged these risks:

1. **{Competitor risk}:** {Competitor A} has {X} users and {Y} funding.
   How do you plan to compete?

2. **{Technical risk}:** {risk specific to this SaaS type}
   → Mitigation: {option from research}

3. **{Market risk}:** {market concern from research}
   → Should we scope down the MVP?

4. **{Monetization risk}:** {pricing concern}
   → Research suggests {alternative}
```

**GATE:** All decisions made. Scope defined.

---

### Phase 4 — VALIDATE

Concept validation checklist — evaluate before writing the PRD.

**4a. Market Fit Check:**

| Check | Status | Evidence |
|-------|--------|----------|
| Clear target user? | PASS/FAIL | {who} |
| Identified pain point? | PASS/FAIL | {what} |
| Differentiation from competitors? | PASS/FAIL | {how} |
| Viable pricing model? | PASS/FAIL | {model + evidence} |
| Market large enough? | PASS/FAIL | {TAM/SAM data} |

**4b. Technical Feasibility Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Stack supports all Must-Have features? | PASS/FAIL | |
| No impossible requirements? | PASS/FAIL | |
| Infrastructure supports expected scale? | PASS/FAIL | |
| Third-party integrations available? | PASS/FAIL | |

**4c. MVP Scope Check:**

| Check | Status |
|-------|--------|
| MVP has 3-8 epics? | PASS/FAIL |
| Each epic has 2-8 stories? | PASS/FAIL |
| Total stories < 30 for v1? | PASS/FAIL |
| Can launch within reasonable timeline? | PASS/FAIL |

**4d. Present validation to user:**

If any check FAILS, discuss with user and adjust before proceeding.

**GATE:** All validation checks pass (or user explicitly acknowledges risks).

---

### Phase 5 — ARCHITECTURE

Apply stack defaults and architecture decisions from brainstorming.

**5a. Load stack defaults:**

Read [references/stack-defaults.md](references/stack-defaults.md) — apply ALL rules.

**5b. Generate architecture based on decisions:**

**If monolith (Next.js only):**
```
Frontend + Backend: Next.js (App Router) on Vercel
Database: Neon (Postgres) + Drizzle ORM
Auth: Clerk
Payments: {Stripe | Lemon Squeezy}
Email: Resend + React Email
Cache: {Upstash Redis if needed}
Real-time: {Convex if needed, else TanStack Query polling}
UI: CossUI + BaseUI + shadcn/ui (BaseUI)
Icons: Tabler Icons
Testing: Vitest
Package Manager: bun
Validation: Zod
Desktop: {Tauri if needed}
3D: {React Three Fiber if needed}
```

**If separate backend needed:**
```
Frontend: Next.js (App Router) on Vercel
Backend: {Rust | Go | Python | C# | C/C++} on VPS Hostinger
Database: Neon (Postgres) + {Drizzle (Next.js) | SeaORM (Rust)}
Auth: Clerk (shared between frontend + backend)
... (same services as above)
```

**5c. Write architecture section** for inclusion in PRD.

**GATE:** Architecture defined.

---

### Phase 6 — WRITE PRD

Generate the complete SaaS PRD following [references/saas-prd-template.md](references/saas-prd-template.md).

**6a. PRD sections (SaaS-specific extensions):**

Standard sections from `/write-prd`:
- Overview, Goals, Target Users, Research Findings, Quality Gates
- Epics & User Stories, Functional Requirements, Non-Functional Requirements
- Non-Goals, Technical Considerations, Success Metrics, Open Questions

SaaS-specific sections:
- **Tech Stack** — complete stack definition from Phase 5
- **Business Model** — pricing, tiers, monetization from Phase 3
- **Infrastructure** — hosting, services, third-party integrations
- **Authentication & Authorization** — Clerk config, roles, permissions
- **Billing & Subscriptions** — payment flow, webhooks, plan management
- **Email System** — transactional emails, templates
- **SaaS Epics** — standard SaaS epics always included (see 6b)

**6b. Standard SaaS epics (always include relevant ones):**

| Epic | When to Include |
|------|----------------|
| EP-001: Project Setup & Infrastructure | Always |
| EP-002: Authentication & User Management | Always |
| EP-003: Billing & Subscriptions | If monetized |
| EP-004: Onboarding | Always |
| EP-005: Core Feature (domain-specific) | Always (1-3 epics) |
| EP-006: Dashboard & Analytics | If B2B or data-heavy |
| EP-007: Team/Org Management | If multi-tenant |
| EP-008: Email System | If transactional emails needed |
| EP-009: Admin Panel | If B2B or marketplace |
| EP-010: Landing Page & Marketing | Always |

Only include epics relevant to the validated scope. Not all are mandatory.

**EP-001: Project Setup** always includes these baseline stories:
- US-001: Initialize Next.js project with bun, TypeScript strict, Tailwind
- US-002: Configure Neon DB + Drizzle ORM schema
- US-003: Set up Clerk authentication
- US-004: Configure Vitest + testing utilities
- US-005: Set up CI quality gates (typecheck, lint, test)

**6c. Write status tracking file:**

JSON at `./tasks/prd-[name]-status.json` following the schema in [references/saas-prd-template.md](references/saas-prd-template.md).

**6d. Save both files:**

```
./tasks/prd-[saas-name].md            — the PRD
./tasks/prd-[saas-name]-status.json   — the status tracker
```

**GATE:** Both files written.

---

### Phase 7 — FINALIZE

Present the PRD for user review.

**7a. Display summary:**

```markdown
## SaaS PRD Summary: {Name}

**Stack:** {Next.js monolith | Next.js + {backend}}
**Database:** Neon + {Drizzle | SeaORM}
**Auth:** Clerk | **Payments:** {Stripe | Lemon Squeezy}
**Hosting:** {Vercel | Vercel + Hostinger VPS}

### Epics ({N} total)
| ID | Title | Stories | Priority |
|----|-------|---------|----------|
| EP-001 | Project Setup | {n} | P0 |
| EP-002 | Auth & Users | {n} | P0 |
| ... | ... | ... | ... |

### Total: {N} stories across {M} epics

### Next Steps
1. `/implement-story tasks/prd-{name}.md US-001` — start with project setup
2. Implement stories in order: EP-001 → EP-002 → EP-003 → core features
3. `/review-story tasks/prd-{name}.md` — review after each epic
```

**7b. Accept modifications** — adjust PRD and status JSON.

**7c. Mark as READY** — update status file.

**GATE:** User approves.

---

## Hard Rules

1. Phase 2 (RESEARCH) is MANDATORY — never brainstorm without market data.
2. Stack defaults are IMMUTABLE — never suggest alternatives to predefined choices (see stack-defaults.md).
3. Only ask questions where genuine decisions exist — do NOT ask about TypeScript vs JavaScript.
4. Every brainstorm question must reference a specific research finding.
5. Quality gates question is MANDATORY.
6. Devil's Advocate phase is MANDATORY.
7. Phase 4 (VALIDATE) must pass before writing the PRD.
8. EP-001 (Project Setup) is always the first epic with baseline stories.
9. Story format: `US-NNN` (compatible with `/implement-story` and `/review-story`).
10. PRD wrapped in `[PRD]...[/PRD]` markers.
11. Status file is JSON.
12. `Files NOT to Modify` section is MANDATORY if codebase exists.
13. Total stories < 30 for MVP — suggest phasing if larger.

## Error Handling

- **agent-websearch fails:** Ask structured questions without competitive context. Note reduced quality.
- **No competitors found:** The idea may be novel — flag this as both opportunity and risk.
- **User wants unsupported tech:** Refer to stack-defaults.md. Only the predefined options are available.
- **Scope too large (>30 stories):** Suggest splitting into v1/v2 releases.
- **Validation fails:** Do not proceed to PRD. Discuss with user and iterate on scope/positioning.

## DO NOT

- Ask about predetermined stack choices (TypeScript, Tailwind, bun, etc.) — these are FIXED.
- Suggest RadixUI, Lucide React, CSS vanilla, npm/yarn/pnpm, or any excluded technology.
- Skip research because "it's a simple SaaS" — even simple products need competitive context.
- Write the PRD before validation passes.
- Include more than 30 stories in the MVP PRD — phase the rest.
- Forget the standard SaaS epics (setup, auth, billing, onboarding).
- Assume single-tenant unless explicitly confirmed.

## Done When

- [ ] Domain and SaaS type identified (Phase 1)
- [ ] Research completed with competitor and market data (Phase 2)
- [ ] All brainstorm rounds completed including quality gates and devil's advocate (Phase 3)
- [ ] Validation checks pass — market fit, feasibility, MVP scope (Phase 4)
- [ ] Architecture defined using stack defaults (Phase 5)
- [ ] PRD file written at `./tasks/prd-[name].md` wrapped in `[PRD]...[/PRD]`
- [ ] Status JSON written at `./tasks/prd-[name]-status.json`
- [ ] User approves the PRD (Phase 7)

## Constraints (Three-Tier)

### ALWAYS
- Run research (Phase 2) before brainstorming — every question backed by data
- Use predefined stack defaults — never suggest alternatives to predetermined choices
- Run validation (Phase 4) before writing the PRD
- Include standard SaaS epics (setup, auth, billing, onboarding)

### ASK FIRST
- Architecture decisions: separate backend, data needs, desktop app, 3D (Phase 3e)
- Proceed when validation check fails (Phase 4)

### NEVER
- Ask about predetermined stack choices (TypeScript, Tailwind, bun, etc.)
- Suggest excluded technologies (RadixUI, Lucide React, npm/yarn/pnpm)
- Write the PRD before validation passes
- Include more than 30 stories in the MVP

## References

- [Stack Defaults](references/stack-defaults.md) — immutable tech stack rules, all predetermined choices, decision trees for open questions
- [SaaS Brainstorm Protocols](references/saas-brainstorm.md) — agent prompts, SaaS-specific question patterns, competitor analysis templates
- [SaaS PRD Template](references/saas-prd-template.md) — extended PRD format with SaaS sections, standard epic templates, status file schema
