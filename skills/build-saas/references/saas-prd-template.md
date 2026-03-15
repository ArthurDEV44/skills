# SaaS PRD Template — Extended Format

## Complete Template

```markdown
[PRD]
# PRD: {SaaS Product Name}

## Overview

{2-3 paragraphs: What is this SaaS? What problem does it solve? Who is it for?
Include the key positioning decision from brainstorming.
Mention the primary differentiator identified through research.}

## Goals

- {Measurable business objective — e.g., "Acquire 100 paying users within 3 months of launch"}
- {Product objective — e.g., "Achieve <2s page load for all core flows"}
- {User objective — e.g., "Reduce user's time spent on {task} by 50%"}

## Target Users

- **{Primary Persona}:** {Role, company size, pain point, current solution, willingness to pay}
- **{Secondary Persona}:** {Role, needs, how they differ from primary}

## Research Findings

### Competitive Landscape
- **{Competitor A}:** {pricing at ${X}/mo, {N} users, strength: {X}, weakness: {Y}}
- **{Competitor B}:** {pricing, features, position}
- **Our Differentiation:** {what we do differently and why it matters}

### Market Context
- Market size: {TAM/SAM if available}
- Trend: {growing/stable/declining}
- Key insight: {the insight that validates this product}

### Risks Identified
- {Risk 1}: mitigated by {approach}
- {Risk 2}: mitigated by {approach}

## Tech Stack

### Core
| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | TypeScript (strict) | Never plain JS |
| Framework | Next.js (App Router) | Server Components default |
| Styling | TailwindCSS v4 | CSS-first config |
| UI | CossUI + BaseUI + shadcn/ui (BaseUI) | Never RadixUI |
| Icons | Tabler Icons | Never Lucide |
| Validation | Zod | All inputs and schemas |
| Package Manager | bun | Exclusively |
| Testing | Vitest | Co-located tests |

### Data & Services
| Service | Technology | Notes |
|---------|-----------|-------|
| Database | Neon (Postgres) | Serverless, branching for dev |
| ORM | Drizzle ORM | {Or SeaORM if Rust backend} |
| Auth | Clerk | {+ Organizations if multi-tenant} |
| Payments | {Stripe / Lemon Squeezy} | {Reason for choice} |
| Email | Resend + React Email | Transactional emails |
| Cache | {Upstash Redis / None} | {If needed: reason} |
| Real-time | {TanStack Query / Convex / None} | {Reason for choice} |

### Infrastructure
| Component | Service |
|-----------|---------|
| Frontend + API | Vercel |
| {Backend if separate} | {VPS Hostinger} |
| Database | Neon |
| {Cache if needed} | {Upstash} |
| {Desktop if needed} | {Tauri} |
| {3D if needed} | {React Three Fiber} |

## Business Model

### Pricing
- **Model:** {Freemium / Free trial / Usage-based / Per-seat / Flat rate}
- **Tiers:**

| Tier | Price | Features | Target |
|------|-------|----------|--------|
| {Free / Trial} | $0 | {limited features} | {acquisition} |
| {Starter} | ${X}/mo | {core features} | {individuals} |
| {Pro} | ${Y}/mo | {full features} | {power users / small teams} |
| {Enterprise} | Custom | {everything + support} | {large orgs} |

### Monetization Notes
- {Payment flow description}
- {Webhook handling: Stripe/LS → Neon DB sync}
- {Subscription management approach}

## Quality Gates

These commands must pass for every user story:
- `bun run typecheck` - TypeScript strict checking
- `bun run lint` - ESLint
- `bun run test` - Vitest test suite

{Additional gates if specified:}
- {gate}

## Epics & User Stories

### EP-001: Project Setup & Infrastructure

Set up the foundational project structure, database, auth, and CI pipeline.

**Definition of Done:** Project runs locally, deploys to Vercel, all quality gates pass on empty project.

#### US-001: Initialize Next.js Project
**Description:** As a developer, I want a properly configured Next.js project so that I can start building features on a solid foundation.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** None

**Acceptance Criteria:**
- [ ] Next.js App Router project created with `bun create next-app`
- [ ] TypeScript strict mode enabled in tsconfig.json
- [ ] TailwindCSS v4 configured with CSS-first approach
- [ ] ESLint configured with TypeScript rules
- [ ] Tabler Icons installed (`@tabler/icons-react`)
- [ ] CossUI + BaseUI + shadcn/ui (BaseUI variant) installed and configured
- [ ] `bun run dev` starts the dev server without errors
- [ ] `bun run build` completes without errors
- [ ] `bun run typecheck` passes
- [ ] `bun run lint` passes

#### US-002: Configure Database
**Description:** As a developer, I want Neon DB connected with Drizzle ORM so that I can define and query the data model.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**
- [ ] Neon database created and connection string configured in `.env.local`
- [ ] Drizzle ORM installed and configured with Neon driver
- [ ] `drizzle.config.ts` configured with schema path and migrations directory
- [ ] Initial schema file created at `src/db/schema/index.ts`
- [ ] `bun run db:push` successfully applies schema to Neon
- [ ] `bun run db:studio` opens Drizzle Studio
- [ ] Database utility at `src/db/index.ts` exports typed `db` instance

#### US-003: Set Up Authentication
**Description:** As a user, I want to sign up and log in so that I can access the application securely.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**
- [ ] Clerk installed and configured with environment variables
- [ ] `middleware.ts` with `clerkMiddleware()` protecting app routes
- [ ] Sign-in page at `/sign-in` using Clerk components
- [ ] Sign-up page at `/sign-up` using Clerk components
- [ ] Protected layout wraps authenticated routes
- [ ] `auth()` works in Server Components and Server Actions
- [ ] User avatar and sign-out available in navigation
- [ ] Clerk webhook endpoint syncs user data to Neon DB

#### US-004: Configure Testing
**Description:** As a developer, I want Vitest configured so that I can write and run tests.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**
- [ ] Vitest installed and configured in `vitest.config.ts`
- [ ] `@vitest/coverage-v8` installed for coverage reports
- [ ] Test utilities configured (React Testing Library if UI tests needed)
- [ ] `bun run test` runs without errors
- [ ] Sample test passes to verify setup

#### US-005: CI Quality Gates
**Description:** As a developer, I want automated quality checks so that code quality is enforced consistently.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by US-001, US-004

**Acceptance Criteria:**
- [ ] `bun run typecheck` script configured in package.json
- [ ] `bun run lint` script configured in package.json
- [ ] `bun run test` script configured in package.json
- [ ] All three commands pass on the initial project

---

### EP-002: Authentication & User Management

{Clerk-based auth with user profiles and optional org management.}

**Definition of Done:** Users can sign up, log in, manage profile, and (if B2B) create/join organizations.

#### US-006: User Profile
**Description:** As a user, I want to view and edit my profile so that I can manage my account information.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-003

**Acceptance Criteria:**
- [ ] Profile page at `/settings/profile`
- [ ] Displays user name, email, avatar from Clerk
- [ ] User can update display name
- [ ] Changes sync to Clerk and local database
- [ ] Page uses CossUI components with warm-toned design

{If multi-tenant, add:}
#### US-007: Organization Management
**Description:** As a team admin, I want to create and manage an organization so that my team can collaborate.

...

---

### EP-003: Billing & Subscriptions

{Stripe or Lemon Squeezy integration for subscription management.}

**Definition of Done:** Users can subscribe, manage their plan, and billing works end-to-end.

#### US-0XX: Pricing Page
...

#### US-0XX: Checkout Flow
...

#### US-0XX: Subscription Management
...

#### US-0XX: Webhook Handler
...

---

### EP-004: Onboarding

{First-run experience for new users.}

---

### EP-005+: Core Feature Epics

{Domain-specific epics — these come entirely from the brainstorming phase.}

---

### EP-0XX: Landing Page & Marketing

{Public-facing landing page with the defined design language.}

**Definition of Done:** Landing page deployed, responsive, passes Lighthouse 90+ scores.

#### US-0XX: Landing Page
**Description:** As a visitor, I want to see a compelling landing page so that I understand the product and want to sign up.

**Acceptance Criteria:**
- [ ] Hero section with mesh gradient background and warm color palette
- [ ] Glass-morphism card components (Apple-style)
- [ ] Grain texture overlay on gradient sections
- [ ] Feature showcase section
- [ ] Pricing section matching business model tiers
- [ ] CTA buttons leading to /sign-up
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Light theme by default
- [ ] {If R3F: 3D visual element in hero section}
- [ ] Lighthouse performance score > 90

---

## Functional Requirements

- FR-01: {System-level requirement}
- FR-02: {System-level requirement}
- ...

## Non-Functional Requirements

- **Performance:** Core pages load in <2s on 3G, API responses <200ms p95
- **Security:** All data encrypted in transit (TLS), Clerk handles auth tokens, no secrets in client code
- **Accessibility:** WCAG 2.1 AA compliant, keyboard navigation, screen reader support
- **SEO:** Landing page SSR, proper meta tags, sitemap, robots.txt

## Non-Goals

- {What v1 explicitly will NOT include}
- {Features deferred to v2}
- {Adjacent products not being built}

## Files NOT to Modify

{Only if extending an existing codebase.}

## Technical Considerations

- **Architecture:** {Monolith on Vercel / + backend on Hostinger}
- **Data Model:** {Key entities and relationships}
- **API Design:** {REST via Route Handlers / Server Actions / tRPC}
- **Third-Party Integrations:** {List of external services}

## Design Language

- **Style:** Modern minimalist
- **Gradients:** Mesh gradients with warm color palette
- **Colors:** Warm tones (amber, coral, warm gray, soft purple)
- **Effects:** Grain texture overlays, glass-morphism (Apple-style)
- **Theme:** Light theme default, dark theme toggle
- **Components:** CossUI + BaseUI + shadcn/ui (BaseUI variant)
- **Icons:** Tabler Icons exclusively
- **Typography:** System font stack or modern sans-serif

## Success Metrics

- {Metric 1: acquisition — e.g., "100 sign-ups in first month"}
- {Metric 2: activation — e.g., "60% of sign-ups complete onboarding"}
- {Metric 3: revenue — e.g., "10 paying customers in first month"}
- {Metric 4: retention — e.g., "<5% monthly churn after 3 months"}

## Open Questions

- {Question 1 — what depends on the answer}
- {Question 2 — who should answer}
[/PRD]
```

---

## Status File Schema (same as /write-prd)

```json
{
  "prd": {
    "file": "tasks/prd-{name}.md",
    "title": "{SaaS Name}",
    "created_at": "{YYYY-MM-DD}",
    "status": "DRAFT | READY | IN_PROGRESS | DONE"
  },
  "stack": {
    "type": "monolith | monolith+backend",
    "frontend": "next.js",
    "backend": "next.js | rust | go | python | csharp | c | cpp",
    "database": "neon",
    "orm": "drizzle | seaorm",
    "auth": "clerk",
    "payments": "stripe | lemonsqueezy",
    "cache": "upstash | none",
    "realtime": "tanstack | convex | none",
    "desktop": "tauri | none",
    "hosting_frontend": "vercel",
    "hosting_backend": "vercel | hostinger"
  },
  "epics": [
    {
      "id": "EP-001",
      "title": "Project Setup & Infrastructure",
      "status": "TODO",
      "priority": "P0",
      "stories_total": 5,
      "stories_done": 0
    }
  ],
  "stories": [
    {
      "id": "US-001",
      "title": "Initialize Next.js Project",
      "epic": "EP-001",
      "status": "TODO",
      "priority": "P0",
      "size": "S",
      "blocked_by": [],
      "started_at": null,
      "completed_at": null,
      "reviewed_at": null
    }
  ]
}
```

The `stack` section is unique to `/build-saas` — it records the architecture decisions so other skills can reference the tech stack without re-reading the full PRD.

---

## Standard SaaS Epic Registry

| Epic | ID Pattern | When | Stories (typical) |
|------|-----------|------|-------------------|
| Project Setup | EP-001 | Always | 5 (init, db, auth, test, ci) |
| Auth & Users | EP-002 | Always | 3-5 (profile, settings, orgs?) |
| Billing | EP-003 | If monetized | 4-6 (pricing, checkout, manage, webhooks) |
| Onboarding | EP-004 | Always | 2-4 (welcome, setup wizard, first action) |
| Core Features | EP-005+ | Always | Varies by product |
| Dashboard | EP-0XX | If B2B / data | 3-5 (overview, charts, filters) |
| Team/Org | EP-0XX | If multi-tenant | 3-4 (create, invite, roles, switch) |
| Email | EP-0XX | If transactional | 2-3 (templates, triggers, preferences) |
| Admin | EP-0XX | If B2B / marketplace | 3-5 (user mgmt, analytics, config) |
| Landing Page | EP-0XX (last) | Always | 2-4 (hero, features, pricing, responsive) |

EP-001 through EP-004 are always present. EP-005+ are domain-specific from brainstorming. Landing page is always last.
