# Organizations (B2B / Multi-Tenant)

> **Prerequisite**: Enable Organizations in Clerk Dashboard first.

## Getting Org Info (Server)

```typescript
import { auth } from '@clerk/nextjs/server'

export default async function OrgPage() {
  const { orgId, orgSlug, orgRole } = await auth()

  if (!orgId) return <div>Select an organization</div>
  return <div>Org: {orgSlug} (Role: {orgRole})</div>
}
```

## Role-Based Access Control with `has()`

### Server Component

```typescript
import { auth } from '@clerk/nextjs/server'

export default async function AdminPage() {
  const { has } = await auth()

  if (!has({ role: 'org:admin' })) {
    return <div>Admin access required</div>
  }

  return <h1>Admin Dashboard</h1>
}
```

### Client Component

```tsx
'use client'
import { useAuth } from '@clerk/nextjs'

export default function AdminPanel() {
  const { has } = useAuth()

  if (!has({ role: 'org:admin' })) return <p>Admin only</p>
  return <div>Admin content</div>
}
```

### Permission Check

```typescript
const { has } = await auth()

// Check specific permission
if (!has({ permission: 'org:team_settings:manage' })) {
  return null
}

// Check role
if (!has({ role: 'org:admin' })) {
  redirect('/dashboard')
}
```

## Default Roles

| Role | Key | Permissions |
|------|-----|-------------|
| Admin | `org:admin` | Full access, manage members and settings |
| Member | `org:member` | Limited access, read-only |

Custom roles can be created in Dashboard -> Organizations -> Roles.

## Default Permissions

| Permission | Default role |
|-----------|-------------|
| `org:create` | Any user |
| `org:manage_members` | Admin |
| `org:manage_roles` | Admin |
| `org:update_metadata` | Admin |

## Dynamic Routes with Org Slug

```
app/orgs/[slug]/page.tsx
app/orgs/[slug]/settings/page.tsx
app/orgs/[slug]/members/page.tsx
```

```typescript
import { auth } from '@clerk/nextjs/server'
import { redirect } from 'next/navigation'

export default async function OrgDashboard({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { orgSlug, has } = await auth()
  const { slug } = await params

  // Verify user is in the correct org
  if (orgSlug !== slug) {
    redirect('/dashboard')
  }

  return <div>Dashboard for {orgSlug}</div>
}
```

## OrganizationSwitcher

Let users switch between organizations:

```tsx
import { OrganizationSwitcher } from '@clerk/nextjs'

export function Header() {
  return (
    <header>
      <OrganizationSwitcher />
    </header>
  )
}
```

## Protect Component for Orgs

```tsx
import { Protect } from '@clerk/nextjs'

export function OrgSettings() {
  return (
    <Protect role="org:admin" fallback={<p>Admin access required</p>}>
      <SettingsForm />
    </Protect>
  )
}
```

## Server Action with Org Check

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'

export async function updateOrgSettings(formData: FormData) {
  const { userId, orgId, has } = await auth()
  if (!userId || !orgId) throw new Error('Must be in an organization')
  if (!has({ role: 'org:admin' })) throw new Error('Admin only')

  await db.organizations.update({
    where: { clerkOrgId: orgId },
    data: { name: formData.get('name') as string },
  })
}
```

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `orgSlug` undefined | No active org selected | User must select org via OrganizationSwitcher |
| Role check always fails | Not awaiting `auth()` | Add `await` |
| Users access other orgs' data | Not checking orgId matches | Verify `orgId` in data queries |
| Org not in switcher | Not enabled in Dashboard | Enable Organizations in Clerk Dashboard |

[Docs](https://clerk.com/docs/guides/organizations/overview)
