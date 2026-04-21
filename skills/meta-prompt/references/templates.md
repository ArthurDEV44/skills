# Domain-Specific Prompt Templates

Proven prompt structures for common project types. Each template follows the tag vocabulary from SKILL.md and incorporates verification loops, plan-before-code, WHY-based constraints, and positive-first directives.

## Template 1: SaaS Application

```prompt
You are a senior full-stack engineer. Build a production-ready SaaS application.

<context>
Project: [NAME] — [one-line description]
Target users: [who uses this and why]
Business model: [free tier / paid tiers / enterprise]
</context>

<stack>
- Framework: Next.js 15+ App Router with TypeScript (strict mode)
- Styling: Tailwind CSS v4
- Auth: Clerk (with organizations for multi-tenant)
- Database: PostgreSQL on Neon with Drizzle ORM
- Payments: Stripe (Checkout Sessions + Customer Portal + Webhooks)
- Deployment: Vercel
</stack>

<plan>
Before writing any code, think through:
1. The data model — entities, relationships, indexes, and constraints
2. The page routes, layouts, and their data requirements
3. The Stripe integration flow — which webhooks, how subscription state syncs to your DB
4. Component hierarchy — what's shared, what's page-specific
Then implement systematically, phase by phase.
</plan>

<requirements>
## Core Features
1. [Feature 1 with details]
2. [Feature 2 with details]
3. [Feature 3 with details]

## Pages and Routes
- / — Marketing landing page with hero, features, pricing, CTA
- /sign-in, /sign-up — Clerk auth pages
- /dashboard — Main authenticated view with [describe content]
- /dashboard/settings — User profile and preferences
- /dashboard/billing — Stripe Customer Portal integration

## Data Model
- [Entity 1]: [fields and relationships]
- [Entity 2]: [fields and relationships]

## Billing
- Free tier: [limits]
- Pro tier ($X/mo): [features]
- Enterprise: [features]
- Stripe webhooks: checkout.session.completed, customer.subscription.updated, customer.subscription.deleted
</requirements>

<constraints>
- Mobile-first responsive — 60%+ of SaaS users access from mobile
- React Server Components by default, client components only for interactivity — reduces bundle size and simplifies data fetching
- Server Actions for all mutations — enables progressive enhancement and type-safe form handling
- Zod validation on all forms and server actions — catches invalid data before it reaches the database
- Proper loading.tsx and error.tsx for every route group — prevents blank screens during navigation
- Realistic sample data that fits the domain — never placeholder text
</constraints>

Avoid these specific pitfalls:
- Placeholder UI ("Coming soon", "Lorem ipsum", generic user avatars)
- Generic AI aesthetic (purple gradients, Inter/Roboto fonts, excessive card layouts)
- Abstractions for one-time operations — keep components focused and flat
- Features not explicitly requested
- Client-side state management libraries — RSC handles this natively

Keep working until the entire implementation is complete. Do not stop with partial features or TODO comments. Verify that all libraries and APIs exist before using them.

<verification>
After implementing, verify:
- `next build` succeeds with zero errors
- All pages render correctly at 375px, 768px, and 1440px
- Auth flow works end-to-end (sign up → sign in → sign out)
- Stripe checkout creates a subscription and webhook updates the database
- Dashboard displays real data from the database
- Forms show validation errors for invalid input
- Empty states are handled for lists with no data
</verification>
```

## Template 2: Ecommerce Platform

```prompt
You are a senior full-stack engineer specializing in ecommerce. Build a production-ready online store.

<context>
Store: [NAME] — [what they sell]
Target market: [audience description]
Scale: [expected catalog size, traffic level]
</context>

<stack>
- Framework: Next.js 15+ App Router with TypeScript
- Styling: Tailwind CSS v4
- Database: PostgreSQL on Neon with Drizzle ORM
- Payments: Stripe (Payment Intents for checkout)
- Search: [built-in / Algolia / Meilisearch]
- Images: Next.js Image optimization + [Cloudinary / Uploadthing]
- Deployment: Vercel
</stack>

<plan>
Before writing code, plan:
1. Product data model — how variants, categories, inventory, and images relate
2. Cart architecture — localStorage for guests, DB-synced for authenticated users
3. Checkout flow — address → shipping → payment → confirmation, with each step validated
4. Admin dashboard — CRUD operations, order status workflow, analytics queries
Then implement phase by phase.
</plan>

<requirements>
## Storefront
- Product catalog with grid/list views and filtering
- Product detail pages with image gallery, variants, reviews
- Search with autocomplete and faceted filtering
- Shopping cart (persisted in localStorage + synced when authenticated)
- Checkout flow with address, shipping, payment steps

## Admin Dashboard
- Product management (CRUD with image upload)
- Order management with status tracking
- Customer list and order history
- Revenue analytics with charts

## Data Model
- Products: name, description, price, images[], variants[], categories[], inventory
- Orders: items[], status, shipping_address, payment_intent_id, total
- Customers: auth_id, orders[], addresses[], cart
- Reviews: product_id, customer_id, rating, text, created_at

## Performance
- Static generation for product pages (ISR with revalidation) — keeps pages fast while allowing content updates
- Optimized images with blur placeholders
- Edge middleware for geolocation-based pricing (if applicable)
</requirements>

<constraints>
- SEO-optimized with proper meta tags, structured data (JSON-LD), and sitemap — ecommerce lives or dies by search visibility
- WCAG 2.1 AA accessible — legal requirement in many markets
- Core Web Vitals green scores — directly affects search ranking
- Realistic product data (15-20 products with varied descriptions and realistic prices)
</constraints>

Avoid these specific pitfalls:
- Placeholder product images or "Lorem ipsum" descriptions
- Missing empty states (empty cart, no search results, no reviews)
- Checkout flow without proper error handling at each step
- Admin dashboard without confirmation for destructive actions (delete product, cancel order)

Keep working until the entire implementation is complete. Verify that all libraries and APIs exist before using them.

<verification>
- Full purchase flow works: browse → add to cart → checkout → payment → order confirmation
- Admin can create/edit/delete products with image upload
- Product pages pass Google's Rich Results Test (structured data)
- Page load under 2s on simulated 4G connection
- Cart persists across page reloads and browser sessions
- Search returns relevant results and handles empty queries
</verification>
```

## Template 3: Landing Page / Marketing Site

```prompt
You are a principal frontend engineer. Build a conversion-optimized landing page with a distinctive visual identity.

<context>
Product: [NAME] — [what it does]
Target audience: [who and what pain point]
Goal: [sign-ups / waitlist / demo requests / purchases]
Tone: [professional / playful / bold / minimalist]
</context>

<stack>
- Framework: Next.js 15+ App Router (or pure HTML/CSS/JS if simpler)
- Styling: Tailwind CSS v4
- Animations: Framer Motion (or CSS-only if lightweight)
- Icons: Lucide React
- Fonts: [specific Google Fonts that match the tone — not Inter, Roboto, Space Grotesk, or Arial]
</stack>

<requirements>
## Sections (in order)
1. **Hero**: Compelling headline, subheadline, primary CTA, optional hero image/animation
2. **Social proof**: Logos, testimonials, or usage stats
3. **Features**: 3-6 key features with icons and descriptions
4. **How it works**: 3-step process visualization
5. **Pricing**: [tiers if applicable]
6. **FAQ**: 5-8 common questions
7. **Final CTA**: Strong closing section with email capture or sign-up button
8. **Footer**: Links, social icons, legal

## Design Direction
- Distinctive visual identity designed for this specific product — not a generic template
- Smooth scroll animations (elements animate in on viewport entry)
- Micro-interactions on buttons and interactive elements
- Dark/light mode support
- Mobile-first, fully responsive
</requirements>

<constraints>
- Choose a distinctive font pairing from Google Fonts — typography sets the personality
- Use a cohesive color palette with a strong primary and sharp accents — not muted grays
- Use CSS gradients, patterns, or abstract shapes instead of stock photos — removes the template feel
- Every section should be scannable — short paragraphs, visual hierarchy, clear headings
</constraints>

Avoid these specific pitfalls:
- Generic SaaS template aesthetic (purple/blue gradients on white, card-heavy layouts)
- Inter, Roboto, Arial, or system fonts — these signal "default"
- Walls of text without visual breaks
- Animations that feel gratuitous rather than purposeful

<verification>
- Page loads in under 1.5s
- Lighthouse performance score > 95
- All animations are smooth at 60fps (no jank on scroll)
- CTA buttons are visible and tappable on all screen sizes
- Page tells a clear narrative from hero to final CTA
- Dark/light mode toggle works with no flash on load
</verification>
```

## Template 4: REST API Backend

```prompt
You are a staff backend engineer. Build a production-ready REST API.

<context>
API: [NAME] — [what it serves]
Consumers: [web app / mobile app / third-party integrations]
Scale: [expected request volume]
</context>

<stack>
- Runtime: [Node.js with Express/Hono | Rust with Axum | Python with FastAPI]
- Database: [PostgreSQL with Drizzle/SQLAlchemy/SeaORM]
- Auth: [JWT / API keys / OAuth2]
- Validation: [Zod / Pydantic / serde]
- Documentation: OpenAPI/Swagger auto-generated
- Testing: [Vitest / pytest / cargo test]
</stack>

<plan>
Before implementing, plan:
1. Resource hierarchy and URL design — RESTful naming, versioning strategy
2. Auth flow — token lifecycle, refresh strategy, permission model
3. Error response format — consistent structure across all endpoints
4. Database schema and migration strategy
Then implement endpoint by endpoint.
</plan>

<requirements>
## Endpoints
[List each endpoint with method, path, request/response shape]

### Authentication
- POST /auth/register — Create account
- POST /auth/login — Get access token
- POST /auth/refresh — Refresh token

### Resources
- GET /api/v1/[resource] — List with cursor-based pagination, filtering, sorting
- GET /api/v1/[resource]/:id — Get by ID
- POST /api/v1/[resource] — Create
- PUT /api/v1/[resource]/:id — Update
- DELETE /api/v1/[resource]/:id — Soft delete

## Cross-Cutting Concerns
- Rate limiting per IP and per API key — prevents abuse without blocking legitimate traffic
- Request/response logging — essential for debugging production issues
- Input validation on all endpoints — rejects malformed data before it reaches business logic
- Consistent error format: `{ error: { code, message, details? } }`
- CORS configuration for known consumers
</requirements>

<constraints>
- Validate all input on the server — client-side validation is for UX only
- Use parameterized queries for all database operations — prevents SQL injection
- Apply rate limiting to auth endpoints with stricter limits — protects against credential stuffing
- Follow principle of least privilege for database queries — select only needed columns
</constraints>

Verify that all libraries and APIs exist before using them. Do not guess package names.

<verification>
- All endpoints return correct HTTP status codes (200, 201, 400, 401, 404, 429)
- Invalid input returns 400 with descriptive error messages following the error format
- Unauthorized requests return 401
- Rate limiting triggers at configured thresholds
- API docs are auto-generated and accessible at /docs
- All endpoints have integration tests with passing assertions
</verification>
```

## Template 5: CLI Tool

```prompt
You are a senior developer specializing in CLI tools. Build a polished command-line application.

<context>
Tool: [NAME] — [what it does]
Target users: [developers / devops / data engineers]
Distribution: [npm / cargo / pip / standalone binary]
</context>

<stack>
- Language: [TypeScript with Commander.js | Rust with Clap | Python with Click/Typer]
- Output: Colored terminal output with [chalk / colored / rich]
- Config: [TOML / YAML / JSON] config file support
</stack>

<requirements>
## Commands
- [command 1]: [description, flags, arguments]
- [command 2]: [description, flags, arguments]

## UX
- Helpful --help output for every command and subcommand
- Progress bars/spinners for long operations
- Colored output: green for success, yellow for warnings, red for errors
- Graceful error messages — no stack traces for user errors
- Confirmation prompts for destructive operations

## Configuration
- Config file at ~/.config/[tool-name]/config.[ext]
- Environment variable overrides — useful for CI/CD
- Command-line flags override everything — principle of least surprise
</requirements>

<constraints>
- Error messages should tell the user what to do, not just what went wrong
- Exit codes must be consistent: 0 for success, 1 for user errors, 2 for system errors
- Config file creation on first run should be non-interactive with sensible defaults
</constraints>

Verify that all libraries exist before importing them. Do not guess package names.

<verification>
- All commands work correctly with valid input
- Invalid input produces helpful, actionable error messages
- `--help` is comprehensive and well-formatted for every command
- Config file is created on first run if missing
- Exit codes are correct: 0 success, 1 user error, 2 system error
- Destructive operations require confirmation (with --force to skip)
</verification>
```

## Template 6: Full-Stack App with Real-Time Features

```prompt
You are a senior full-stack engineer. Build a real-time collaborative application.

<context>
App: [NAME] — [description]
Real-time needs: [what updates in real-time and why]
</context>

<stack>
- Framework: Next.js 15+ App Router with TypeScript
- Real-time: [WebSockets via Socket.io | Server-Sent Events | Supabase Realtime | Pusher]
- Database: PostgreSQL with Drizzle ORM
- Auth: Clerk
- Styling: Tailwind CSS v4
</stack>

<plan>
Before implementing, plan:
1. Real-time data flow — what events propagate, who receives them, consistency model
2. Optimistic update strategy — how to apply changes immediately and roll back on failure
3. Offline/reconnection behavior — queue management, conflict resolution
4. Connection state management — how the UI reflects connectivity status
Then implement the data layer first, real-time layer second, UI last.
</plan>

<requirements>
## Real-Time Features
- [Feature 1]: [what updates, who sees it, latency requirement]
- [Feature 2]: [description]

## Offline/Reconnection
- Queue actions when offline — user should not lose work
- Sync on reconnection with conflict detection
- Show connection status indicator — users need to know when they're offline

## Optimistic Updates
- Apply changes immediately in UI — perceived latency under 50ms
- Roll back on server error with user notification
- Show sync status per item (synced / syncing / conflict)
</requirements>

<constraints>
- Real-time updates must not cause full-page re-renders — use granular subscriptions
- Optimistic updates must be reversible — never persist optimistic state to DB before confirmation
- Connection loss must be graceful — no error modals, just a status indicator and automatic reconnection
</constraints>

Keep working until the entire implementation is complete. Verify that all libraries exist before using them.

<verification>
- Changes appear for other users within 500ms
- Offline actions sync correctly on reconnection
- Optimistic updates roll back cleanly on server error
- No data loss on connection interruptions (disconnect WiFi and reconnect test)
- Works with 10+ concurrent users on the same resource
- Connection status indicator reflects actual state accurately
</verification>
```

## Universal Template

Use this structure when transforming any vague prompt into an optimized one:

```prompt
You are a [specific expert role]. Build a [production-ready/polished/complete] [thing].

<context>
[What exists, who it's for, why it matters]
</context>

<stack>
[Technologies, frameworks, libraries — with versions when relevant]
</stack>

<plan>
Before implementing, think through:
1. [Key architectural decision for this domain]
2. [Data model or structure]
3. [Component/module hierarchy]
Then build systematically.
</plan>

<requirements>
[Numbered list of specific, measurable features]
[Organized by category: Core, UI, Data, Integration, etc.]
</requirements>

<constraints>
[Hard rules with motivations — why each constraint matters]
[Performance targets, accessibility standards, responsive breakpoints]
</constraints>

Avoid these specific pitfalls:
[3-5 concrete pitfalls relevant to this domain]

Keep working until complete. Verify libraries and APIs exist before using them.

<verification>
[Concrete, testable checks Claude runs to validate its output]
[Both functional and non-functional: build succeeds, pages render, flows work end-to-end]
</verification>
```
