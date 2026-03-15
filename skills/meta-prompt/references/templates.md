# Domain-Specific Prompt Templates

Battle-tested prompt structures for common project types. Use these as starting points and customize based on the user's specific requirements.

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
- Implement Stripe webhooks for: checkout.session.completed, customer.subscription.updated, customer.subscription.deleted
</requirements>

<constraints>
- Mobile-first responsive design
- Use React Server Components by default, client components only when needed
- Server Actions for all mutations (no API routes for CRUD)
- Zod validation on all forms and server actions
- Proper loading.tsx and error.tsx for every route group
- Real sample data, not placeholders
</constraints>

<anti-patterns>
- No over-engineering: keep components focused and flat
- No placeholder UI ("Coming soon", "Lorem ipsum")
- No generic AI aesthetic (purple gradients, Inter font, card soup)
- No unnecessary abstractions or utility files
- No client-side state management libraries
</anti-patterns>

<acceptance-criteria>
- App builds without errors (next build succeeds)
- All pages render correctly at 375px, 768px, and 1440px
- Auth flow works end-to-end (sign up, sign in, sign out)
- Stripe checkout creates a subscription
- Webhooks update the database correctly
- Dashboard shows real data from the database
</acceptance-criteria>
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
- Static generation for product pages (ISR with revalidation)
- Optimized images with blur placeholders
- Edge middleware for geolocation-based pricing (if applicable)
</requirements>

<constraints>
- SEO-optimized: proper meta tags, structured data (JSON-LD), sitemap
- Accessible: WCAG 2.1 AA compliant
- Performance: Core Web Vitals green scores
- Realistic product data (15-20 products with varied descriptions and realistic prices)
</constraints>

<acceptance-criteria>
- Full purchase flow works: browse > add to cart > checkout > payment > order confirmation
- Admin can create/edit/delete products
- Product pages are indexed correctly (test with structured data validator)
- Page load under 2s on 4G connection
- Cart persists across page reloads
</acceptance-criteria>
```

## Template 3: Landing Page / Marketing Site

```prompt
You are a principal frontend engineer and creative developer. Build a stunning, conversion-optimized landing page.

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
- Fonts: [specific Google Fonts that match the tone — NOT Inter or Roboto]
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

## Design Requirements
- Distinctive visual identity (NOT generic SaaS template)
- Smooth scroll animations (elements animate in as they enter viewport)
- Micro-interactions on buttons and interactive elements
- Dark/light mode support
- Mobile-first, fully responsive
</requirements>

<anti-patterns>
- No generic "AI slop" aesthetic
- No stock photo placeholders — use CSS gradients, patterns, or abstract shapes
- No cookie-cutter component layouts
- No walls of text — every section should be scannable
- No Inter, Roboto, Arial, or system fonts
</anti-patterns>

<acceptance-criteria>
- Page loads in under 1.5s
- Lighthouse performance score > 95
- All animations are smooth (no jank)
- CTA buttons are visible and accessible on all screen sizes
- Page tells a clear story from top to bottom
</acceptance-criteria>
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

<requirements>
## Endpoints
[List each endpoint with method, path, request/response shape]

### Authentication
- POST /auth/register — Create account
- POST /auth/login — Get access token
- POST /auth/refresh — Refresh token

### Resources
- GET /api/v1/[resource] — List with pagination, filtering, sorting
- GET /api/v1/[resource]/:id — Get by ID
- POST /api/v1/[resource] — Create
- PUT /api/v1/[resource]/:id — Update
- DELETE /api/v1/[resource]/:id — Soft delete

## Cross-Cutting Concerns
- Rate limiting (per IP and per API key)
- Request/response logging
- Input validation on all endpoints
- Consistent error response format: { error: { code, message, details? } }
- Pagination: cursor-based for lists
- CORS configuration
</requirements>

<acceptance-criteria>
- All endpoints return correct status codes
- Invalid input returns 400 with descriptive errors
- Unauthorized requests return 401
- Rate limiting triggers at configured thresholds
- API docs are auto-generated and accessible at /docs
- All endpoints have integration tests
</acceptance-criteria>
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
- Graceful error messages (no stack traces for user errors)
- Confirmation prompts for destructive operations

## Configuration
- Config file at ~/.config/[tool-name]/config.[ext]
- Environment variable overrides
- Command-line flags override everything
</requirements>

<acceptance-criteria>
- All commands work correctly with valid input
- Invalid input produces helpful error messages
- --help is comprehensive and well-formatted
- Config file is created on first run if missing
- Tool exits with appropriate exit codes (0 success, 1 error)
</acceptance-criteria>
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

<requirements>
## Real-Time Features
- [Feature 1]: [what updates, who sees it, latency requirement]
- [Feature 2]: [description]

## Offline/Reconnection
- Queue actions when offline
- Sync on reconnection
- Show connection status indicator

## Optimistic Updates
- Apply changes immediately in UI
- Roll back on server error
- Show sync status per item
</requirements>

<acceptance-criteria>
- Changes appear for other users within 500ms
- Offline actions sync correctly on reconnection
- Optimistic updates roll back cleanly on error
- No data loss on connection interruptions
- Works with 10+ concurrent users on same resource
</acceptance-criteria>
```

## Meta-Template: The Universal Enhancer

Use this structure when transforming ANY vague prompt into an optimized one:

```prompt
You are a [specific expert role]. [Build/Create/Implement] a [production-ready/polished/complete] [thing].

<context>
[What exists, who it's for, why it matters]
</context>

<stack>
[Every technology, framework, and library — with versions when relevant]
</stack>

<requirements>
[Numbered list of specific, measurable features]
[Organized by category: Core, UI, Data, Integration, etc.]
</requirements>

<constraints>
[Hard rules that must be followed]
[Performance targets, accessibility standards, responsive breakpoints]
</constraints>

<anti-patterns>
[What to explicitly avoid — prevents Claude from over-engineering or producing generic output]
</anti-patterns>

<acceptance-criteria>
[Concrete, testable conditions that define "done"]
[Include both functional and non-functional requirements]
</acceptance-criteria>
```
