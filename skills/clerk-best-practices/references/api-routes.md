# API Routes (Route Handlers)

## Basic Auth Check

```typescript
// app/api/data/route.ts
import { auth } from '@clerk/nextjs/server'

export async function GET() {
  const { userId } = await auth()
  if (!userId) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const data = await db.data.findMany({ where: { userId } })
  return Response.json(data)
}
```

## 401 vs 403

- **401 Unauthorized** - User is not authenticated (not signed in)
- **403 Forbidden** - User is authenticated but lacks permission

```typescript
export async function DELETE(req: Request) {
  const { userId, has } = await auth()

  // Not signed in
  if (!userId) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Signed in but not admin
  const isAdmin = has({ role: 'org:admin' })
  if (!isAdmin) {
    return Response.json({ error: 'Forbidden' }, { status: 403 })
  }

  // Authorized - proceed
  return Response.json({ success: true })
}
```

## Org-Scoped Route Protection

Verify user belongs to the requested organization:

```typescript
// app/api/orgs/[orgId]/data/route.ts
export async function GET(
  req: Request,
  { params }: { params: Promise<{ orgId: string }> }
) {
  const { userId, orgId } = await auth()
  const { orgId: requestedOrgId } = await params

  if (!userId) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Ensure user's active org matches the requested org
  if (orgId !== requestedOrgId) {
    return Response.json({ error: 'Forbidden' }, { status: 403 })
  }

  const data = await db.orgData.findMany({ where: { organizationId: orgId } })
  return Response.json(data)
}
```

## Permission-Based Protection

```typescript
export async function POST(req: Request) {
  const { userId, has } = await auth()
  if (!userId) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 })
  }

  if (!has({ permission: 'org:invoices:create' })) {
    return Response.json({ error: 'Forbidden' }, { status: 403 })
  }

  const body = await req.json()
  const invoice = await db.invoices.create({ data: { ...body, createdBy: userId } })
  return Response.json(invoice, { status: 201 })
}
```

## Public API Routes

For routes that don't require auth (e.g., health checks), ensure they're in the public matcher:

```typescript
// middleware.ts
const isPublicRoute = createRouteMatcher([
  '/api/webhooks(.*)',
  '/api/health',
  '/api/public(.*)',
])
```

## Rules

1. **Always check auth at the start** even if middleware protects the route (defense in depth)
2. **Use proper HTTP status codes** - 401 for unauthenticated, 403 for unauthorized
3. **Scope data queries by userId/orgId** - Never return data the user shouldn't see
4. **Include API routes in middleware matcher** - `/(api|trpc)(.*)` must be in matcher

[Docs](https://clerk.com/docs/reference/nextjs/auth)
