---
name: clerk-best-practices
description: >-
  Clerk authentication best practices for Next.js (App Router). Comprehensive guide covering
  setup, middleware, Server Components, Server Actions, API routes, organizations, RBAC,
  webhooks, caching, custom UI, and testing. Use when writing, reviewing, or refactoring
  Next.js code with Clerk auth: (1) Setting up ClerkProvider and environment variables,
  (2) Configuring clerkMiddleware with createRouteMatcher for public-first or protected-first
  strategies, (3) Using auth() and currentUser() in Server Components, (4) Protecting Server
  Actions and API routes, (5) Implementing organizations with role-based access control using
  has(), (6) Syncing data with verifyWebhook, (7) Auth-aware caching with unstable_cache,
  (8) Customizing Clerk UI appearance, (9) E2E testing with Playwright or Cypress. This skill
  does NOT cover Clerk for non-Next.js frameworks (Remix, Astro, Vue) or the Clerk Backend API
  directly.
license: MIT
metadata:
  author: arthur
  version: "1.0.0"
---

# Clerk Best Practices for Next.js

Comprehensive guide for building production-grade authentication with Clerk in Next.js App Router applications.

## Mental Model

Server vs Client = different auth APIs. Never mix imports.

| Context | Import from | Auth function | Notes |
|---------|-------------|---------------|-------|
| Server Component | `@clerk/nextjs/server` | `await auth()` | **Must await** |
| Server Action | `@clerk/nextjs/server` | `await auth()` | Always verify first |
| API Route | `@clerk/nextjs/server` | `await auth()` | Return 401/403 |
| Client Component | `@clerk/nextjs` | `useAuth()` / `useUser()` | Sync hooks |
| Middleware | `@clerk/nextjs/server` | `auth.protect()` | Route-level gate |
| Webhooks | `@clerk/nextjs/webhooks` | `verifyWebhook(req)` | Signature check |

## References

| Reference | Impact | When to read |
|-----------|--------|--------------|
| `references/setup.md` | CRITICAL | Adding Clerk to a project |
| `references/middleware.md` | CRITICAL | Configuring route protection |
| `references/server-components.md` | CRITICAL | Server vs Client auth patterns |
| `references/server-actions.md` | HIGH | Protecting mutations |
| `references/api-routes.md` | HIGH | Building API endpoints |
| `references/organizations.md` | HIGH | B2B multi-tenant with RBAC |
| `references/webhooks.md` | HIGH | Syncing data to your database |
| `references/caching.md` | MEDIUM | Auth-aware caching strategies |
| `references/custom-ui.md` | MEDIUM | Theming and appearance |
| `references/testing.md` | MEDIUM | E2E testing setup |

## Quick Start (Protected-First)

### 1. Install

```bash
npm install @clerk/nextjs
```

### 2. Environment Variables

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
```

### 3. ClerkProvider (root layout)

```tsx
// app/layout.tsx
import { ClerkProvider } from '@clerk/nextjs'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>{children}</body>
      </html>
    </ClerkProvider>
  )
}
```

### 4. Middleware (protected-first)

```typescript
// middleware.ts
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isPublicRoute = createRouteMatcher([
  '/',
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/webhooks(.*)',
])

export default clerkMiddleware(async (auth, req) => {
  if (!isPublicRoute(req)) {
    await auth.protect()
  }
})

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
```

### 5. Server Component

```tsx
import { auth, currentUser } from '@clerk/nextjs/server'

export default async function DashboardPage() {
  const { userId } = await auth()
  if (!userId) return <p>Not signed in</p>

  const user = await currentUser()
  return <h1>Welcome, {user?.firstName}!</h1>
}
```

## Critical Rules

1. **Always `await auth()`** - Forgetting `await` returns `undefined` for all fields
2. **Never import `@clerk/nextjs/server` in Client Components** - Use `useAuth()` / `useUser()` instead
3. **Never expose `CLERK_SECRET_KEY`** - Only `NEXT_PUBLIC_*` keys are safe for client code
4. **Always protect Server Actions** - They are public endpoints; check auth at the top
5. **Always include userId/orgId in cache keys** - Prevents data leaking between users
6. **Make webhook routes public** - Exclude `/api/webhooks(.*)` from middleware protection
7. **Use `has()` for RBAC** - Check permissions/roles, not string comparisons on orgRole
8. **Return proper HTTP codes** - 401 = not authenticated, 403 = no permission

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `undefined` userId in Server Component | Missing `await` | `const { userId } = await auth()` |
| Auth not working on API routes | Missing middleware matcher | Add `'/(api\|trpc)(.*)'` |
| Cache returns wrong user's data | userId missing from cache key | Include `userId` in `unstable_cache` key |
| Mutations bypass auth | Unprotected Server Action | Check `auth()` at top of action |
| Wrong HTTP error code | Confused 401/403 | 401 = unauthenticated, 403 = unauthorized |
| Webhook verification fails | Wrong import or protected route | Use `@clerk/nextjs/webhooks`, make route public |
| Org role check always fails | Not awaiting `auth()` | Add `await` before `auth()` |
| `orgSlug` undefined | No active organization | Check if user selected an org |

## Documentation

- [Next.js SDK Reference](https://clerk.com/docs/reference/nextjs/overview)
- [Middleware Reference](https://clerk.com/docs/reference/nextjs/clerk-middleware)
- [Auth Helper](https://clerk.com/docs/reference/nextjs/auth)
- [Server Actions](https://clerk.com/docs/reference/nextjs/app-router/server-actions)
- [Organizations](https://clerk.com/docs/guides/organizations/overview)
- [Webhooks](https://clerk.com/docs/guides/development/webhooks/overview)
- [Testing](https://clerk.com/docs/guides/development/testing/overview)
