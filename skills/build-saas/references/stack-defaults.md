# Stack Defaults — Immutable Technology Choices

These are HARD RULES. Never suggest alternatives. Never ask the user to choose between these and other options. If a technology is marked NEVER, do not include it under any circumstance.

---

## Core Stack

### Language
- **ALWAYS:** TypeScript (strict mode)
- **NEVER:** JavaScript (plain JS), CoffeeScript, or any non-typed variant
- `tsconfig.json`: `"strict": true`, `"noUncheckedIndexedAccess": true`

### Package Manager
- **ALWAYS:** bun
- **NEVER:** npm, yarn, pnpm
- All commands use `bun`: `bun install`, `bun run dev`, `bun run build`, `bun run test`
- Lock file: `bun.lock`

### Validation
- **ALWAYS:** Zod
- Use for: API input validation, form validation, env var parsing, config schemas
- Pattern: define schema → infer type with `z.infer<typeof schema>`

---

## Frontend

### Framework
- **ALWAYS:** Next.js (App Router, latest stable)
- Server Components by default, `"use client"` only when needed
- File conventions: `page.tsx`, `layout.tsx`, `loading.tsx`, `error.tsx`, `not-found.tsx`

### Styling
- **ALWAYS:** TailwindCSS (v4+)
- **NEVER:** CSS vanilla, CSS modules, Sass, styled-components, Emotion
- CSS-first config with `@theme`, `@utility`, `@custom-variant` directives

### UI Components
- **ALWAYS:** CossUI + BaseUI + shadcn/ui (BaseUI variant only)
- **NEVER:** RadixUI (standalone), MUI, Ant Design, Chakra UI
- shadcn/ui components MUST use BaseUI primitives, not Radix primitives
- CossUI for the component layer on top of BaseUI

### Icons
- **ALWAYS:** Tabler Icons (`@tabler/icons-react`)
- **NEVER:** Lucide React, Heroicons, FontAwesome, Material Icons
- Import pattern: `import { IconName } from '@tabler/icons-react'`

### State & Data
- **DEFAULT:** TanStack ecosystem (Query, Form, Table, Store, Router)
- **ALTERNATIVE:** Convex (when real-time sync is a core requirement)
- Decision tree:
  - Standard CRUD + API calls → TanStack Query + Drizzle
  - Real-time collaboration, live cursors, instant sync → Convex
  - Hybrid (some real-time, mostly CRUD) → TanStack Query + Convex for real-time parts

### 3D / Advanced Visuals
- **IF NEEDED:** React Three Fiber (@react-three/fiber + @react-three/drei)
- Use for: landing page visuals, product 3D features, data visualization, tech branding
- Not required for every project — only when explicitly chosen

### Design Language
- **Style:** Modern minimalist
- **Gradients:** Mesh gradients (warm color palettes)
- **Colors:** Warm tones (ambers, corals, warm grays, soft purples)
- **Effects:** Grain texture overlays, glass-morphism (Apple-style)
- **Theme:** Light theme by default (dark theme as optional toggle)
- **Inspiration:** Apple design language, Linear app, Vercel dashboard

---

## Backend

### Default (Monolith)
- **ALWAYS:** Next.js API Routes (Route Handlers) + Server Actions
- Hosted on Vercel
- No separate backend needed for most SaaS products

### Separate Backend (only when required)
**Trigger:** Heavy computation, WebSocket servers, persistent connections, background job processing, system-level access, ML inference, extreme performance requirements

**Language options (ranked by preference):**
1. **Rust** — performance-critical, systems-level, high concurrency (use Axum framework)
2. **Go** — microservices, concurrent services, DevOps tools
3. **Python** — ML/AI, data processing, scripting-heavy
4. **C#** — enterprise integrations, .NET ecosystem
5. **C/C++** — extreme performance, embedded, real-time systems

**Hosting for separate backend:** VPS on Hostinger

**ORM for Rust backend:** SeaORM (not Drizzle — Drizzle is TypeScript only)

---

## Database

### Primary Database
- **ALWAYS:** Neon (serverless Postgres)
- Connection pooling enabled
- Branching for dev/staging environments

### ORM
- **Next.js projects:** Drizzle ORM
- **Rust backend:** SeaORM
- Schema-first approach: define schema in code, generate migrations

### Cache / KV Store
- **IF NEEDED:** Upstash Redis (serverless)
- **NEVER:** Self-hosted Redis, Memcached
- Use for: rate limiting, session caching, job queues, feature flags
- Only include if the product needs caching or rate limiting

---

## Authentication

- **ALWAYS:** Clerk
- Clerk Organizations for multi-tenant B2B
- Clerk Webhooks for user sync to database
- Middleware: `clerkMiddleware()` in `middleware.ts`
- Server: `auth()` from `@clerk/nextjs/server`
- Client: `useUser()`, `useAuth()` from `@clerk/nextjs`

---

## Payments

- **Option A:** Stripe — complex billing, usage-based, enterprise, existing Stripe users
- **Option B:** Lemon Squeezy — simpler, handles tax/VAT as merchant of record, indie/solo

**Decision tree:**
- Usage-based billing, metered, complex tiers → Stripe
- Simple subscription tiers, global tax handling needed → Lemon Squeezy
- B2B with invoicing, custom contracts → Stripe
- Solo founder, want simplicity → Lemon Squeezy

---

## Email

- **ALWAYS:** Resend (email sending API)
- **ALWAYS:** React Email (email template components)
- Templates in: `src/emails/` or `emails/`
- Pattern: React components → rendered to HTML by React Email → sent via Resend

---

## Testing

- **ALWAYS:** Vitest
- **NEVER:** Jest, Mocha, Jasmine
- Config: `vitest.config.ts`
- Pattern: `*.test.ts` / `*.test.tsx` co-located with source
- Coverage: `@vitest/coverage-v8`

---

## Desktop (Optional)

- **IF NEEDED:** Tauri (v2)
- Use when: offline access, system-level integration, file system access, native performance
- Stack: Next.js frontend + Tauri shell + Rust backend commands
- Not needed for most SaaS — only when explicitly requested

---

## Hosting

| Component | Service |
|-----------|---------|
| Next.js frontend + API | Vercel |
| Separate backend (if any) | VPS Hostinger |
| Database | Neon |
| Cache (if any) | Upstash Redis |
| Auth | Clerk (cloud) |
| Payments | Stripe or Lemon Squeezy (cloud) |
| Email | Resend (cloud) |
| Real-time (if Convex) | Convex (cloud) |

---

## Decisions That Remain Open

These are the ONLY technical questions to ask during brainstorming:

| Decision | Options | Depends On |
|----------|---------|------------|
| Separate backend? | No (Next.js only) / Yes (+ language) | Feature complexity |
| Which payment provider? | Stripe / Lemon Squeezy | Billing complexity |
| TanStack or Convex? | TanStack / Convex / Both | Real-time needs |
| Need Upstash Redis? | Yes / No | Caching/rate-limiting needs |
| Need desktop app? | Yes (Tauri) / No | Offline/native needs |
| Need 3D visuals? | Yes (R3F) / No / Landing only | Product type |
| Multi-tenant? | Yes (Clerk Orgs) / No | B2B vs B2C |
