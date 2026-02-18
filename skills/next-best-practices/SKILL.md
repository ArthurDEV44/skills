---
name: next-best-practices
description: >
  Next.js best practices for the App Router (v14/v15/v16+). Use when writing, reviewing, or
  refactoring Next.js code: (1) File conventions - page, layout, loading, error, not-found, route,
  template, middleware/proxy, (2) RSC boundaries - use client, composition, serialization rules,
  (3) Data fetching - caching, revalidation, ISR, streaming, Suspense, generateStaticParams, PPR,
  (4) Metadata - generateMetadata, Open Graph, Twitter cards, robots, sitemap, OG images,
  (5) Error handling - error.tsx, global-error, redirect/notFound try-catch gotchas, useActionState,
  (6) Route handlers - CORS, streaming, cookies/headers, (7) Server Actions - revalidatePath,
  revalidateTag, useFormStatus, (8) Middleware - matcher, NextResponse, auth patterns,
  (9) next/image - sizes, priority, fill, blur, (10) next/font - Google, local, Tailwind v4,
  (11) Bundling - turbopack, serverExternalPackages, dynamic ssr:false,
  (12) Async APIs - params, searchParams, cookies, headers as Promises in v15+.
---

# Next.js Best Practices (App Router)

Comprehensive best practices for Next.js App Router targeting v14, v15, and v16+.

## Quick Reference

| Topic | Reference File |
|-------|---------------|
| Project structure, special files, routing | [references/file-conventions.md](references/file-conventions.md) |
| Server/Client component boundary rules | [references/rsc-boundaries.md](references/rsc-boundaries.md) |
| Async params, searchParams, cookies, headers (v15+) | [references/async-patterns.md](references/async-patterns.md) |
| Node.js vs Edge runtime | [references/runtime-selection.md](references/runtime-selection.md) |
| `'use client'`, `'use server'`, `'use cache'` | [references/directives.md](references/directives.md) |
| Navigation hooks, server functions, generate functions | [references/functions.md](references/functions.md) |
| error.tsx, global-error, not-found, redirect gotchas | [references/error-handling.md](references/error-handling.md) |
| Server Components vs Server Actions vs Route Handlers | [references/data-patterns.md](references/data-patterns.md) |
| route.ts basics, HTTP methods, CORS | [references/route-handlers.md](references/route-handlers.md) |
| Static/dynamic metadata, OG images, sitemaps | [references/metadata.md](references/metadata.md) |
| next/image, responsive sizes, blur, priority | [references/image.md](references/image.md) |
| next/font, Google/local fonts, Tailwind integration | [references/font.md](references/font.md) |
| Server-incompatible packages, ESM/CJS, bundle analysis | [references/bundling.md](references/bundling.md) |
| next/script, loading strategies, Google Analytics | [references/scripts.md](references/scripts.md) |
| Hydration mismatch causes and fixes | [references/hydration-error.md](references/hydration-error.md) |
| useSearchParams/usePathname Suspense requirements | [references/suspense-boundaries.md](references/suspense-boundaries.md) |
| @slot modals, intercepting routes, default.tsx | [references/parallel-routes.md](references/parallel-routes.md) |
| Docker, standalone output, ISR cache handlers | [references/self-hosting.md](references/self-hosting.md) |
| MCP dev endpoint, debug-build-paths | [references/debug-tricks.md](references/debug-tricks.md) |

## Core Principles

### 1. Server Components by Default

All components in `app/` are Server Components. Only add `'use client'` when you need interactivity (hooks, event handlers, browser APIs). Push client boundaries as deep as possible.

### 2. Async APIs in v15+

`params`, `searchParams`, `cookies()`, `headers()`, and `draftMode()` are **Promises** — always `await` them:

```tsx
export default async function Page({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  // ...
}
```

### 3. Data Fetching Decision Tree

- **Server Component reads**: Fetch directly (no API layer needed)
- **Mutations from UI**: Use Server Actions (`'use server'`)
- **External/public APIs**: Use Route Handlers (`route.ts`)
- **Client Component reads**: Pass data from Server Component parent, or fetch from Route Handler

### 4. Error Handling: Never Wrap Navigation in try-catch

`redirect()`, `notFound()`, `forbidden()`, `unauthorized()` throw internally. Call them outside `try-catch` blocks, or use `unstable_rethrow()`.

### 5. Metadata: Server Components Only

`metadata` object and `generateMetadata` only work in Server Components. Use `React.cache()` to deduplicate fetches shared between metadata and page.

### 6. Image and Font Optimization

- Always use `next/image` over `<img>` — set `priority` on LCP images and `sizes` on responsive images
- Always use `next/font` over `<link>` tags — self-hosts fonts with zero layout shift
- Use CSS `variable` option and `display: 'swap'` for best performance

### 7. Streaming and Suspense

Wrap slow async components in `<Suspense>` boundaries with skeleton fallbacks. Use `loading.tsx` for route-level loading states. This enables progressive rendering and better perceived performance.

### 8. Bundle Safety

- Use `dynamic(() => import('...'), { ssr: false })` for browser-only packages
- Use `serverExternalPackages` for native Node.js bindings
- Use `transpilePackages` for ESM/CJS compatibility issues
