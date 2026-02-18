# Middleware Strategies

## File Location

Always place at the project root: `middleware.ts` (or `middleware.js`).

## Matcher Config (Always Required)

```typescript
export const config = {
  matcher: [
    // Skip Next.js internals and static files
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
}
```

## Strategy 1: Protected-First (Recommended for Apps)

Block everything by default, whitelist specific public routes. Best for dashboards, SaaS apps, internal tools.

```typescript
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isPublicRoute = createRouteMatcher([
  '/',
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/webhooks(.*)',
  '/api/public(.*)',
])

export default clerkMiddleware(async (auth, req) => {
  if (!isPublicRoute(req)) {
    await auth.protect()
  }
})
```

## Strategy 2: Public-First (Marketing Sites)

Allow everything by default, protect specific routes. Best for marketing sites, blogs, docs with gated content.

```typescript
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isProtectedRoute = createRouteMatcher([
  '/dashboard(.*)',
  '/settings(.*)',
  '/api/private(.*)',
])

export default clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) {
    await auth.protect()
  }
})
```

## Role-Based Route Protection

Protect routes based on organization roles or permissions:

```typescript
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isAdminRoute = createRouteMatcher(['/admin(.*)'])
const isDashboardRoute = createRouteMatcher(['/dashboard(.*)'])
const isPublicRoute = createRouteMatcher(['/', '/sign-in(.*)', '/sign-up(.*)'])

export default clerkMiddleware(async (auth, req) => {
  if (isAdminRoute(req)) {
    await auth.protect({ role: 'org:admin' })
  }

  if (isDashboardRoute(req)) {
    await auth.protect()
  }
})
```

## Permission-Based Protection

```typescript
export default clerkMiddleware(async (auth, req) => {
  if (isAdminRoute(req)) {
    await auth.protect((has) => {
      return has({ permission: 'org:admin:access' }) || has({ role: 'org:admin' })
    })
  }
})
```

## Multi-Tenant Org Routes

```typescript
const isOrgRoute = createRouteMatcher(['/orgs/:slug(.*)'])

export default clerkMiddleware(async (auth, req) => {
  if (isOrgRoute(req)) {
    await auth.protect() // Ensures user is authenticated
    // Org slug validation should happen in the page/layout, not middleware
  }
})
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not including `/api/webhooks` in public routes | Webhook verification fails with 401 |
| Forgetting `(.*)` suffix on route patterns | Sub-paths like `/sign-in/sso-callback` blocked |
| Missing API matcher `/(api\|trpc)(.*)` | API routes bypass auth entirely |
| Putting middleware in `src/` when not using `src/` dir | Place in project root next to `next.config` |
| Using `src/middleware.ts` when project has `src/` dir | Correct - must be at root of source |

[Docs](https://clerk.com/docs/reference/nextjs/clerk-middleware)
