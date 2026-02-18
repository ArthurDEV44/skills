# Server Components and Client Components

## CRITICAL: Always `await auth()`

```tsx
// WRONG - returns undefined for all fields
const { userId } = auth()

// CORRECT
const { userId } = await auth()
```

## Import Rules

```tsx
// Server Components, Server Actions, API Routes
import { auth, currentUser } from '@clerk/nextjs/server'

// Client Components only
'use client'
import { useAuth, useUser, useClerk } from '@clerk/nextjs'
```

**Never mix these.** Server imports in Client Components will throw errors.

## Server Component Pattern

```tsx
import { auth, currentUser } from '@clerk/nextjs/server'

export default async function DashboardPage() {
  const { userId } = await auth()
  if (!userId) return <div>Please sign in</div>

  // currentUser() fetches full user object from Clerk API
  // Note: counts toward rate limits - prefer auth() when you only need userId
  const user = await currentUser()
  return <h1>Welcome, {user?.firstName}!</h1>
}
```

## `auth()` vs `currentUser()`

| Helper | Returns | Rate limit | Use when |
|--------|---------|------------|----------|
| `auth()` | `{ userId, orgId, orgRole, orgSlug, has, sessionClaims }` | No API call | You need userId, org info, or permission checks |
| `currentUser()` | Full Backend User object | Counts toward API rate limit | You need user profile data (name, email, image) |

**Prefer `auth()` whenever possible.** Only use `currentUser()` when you need profile fields.

## `auth()` Return Object

```typescript
const {
  userId,           // string | null
  orgId,            // string | null
  orgRole,          // string | null (e.g., 'org:admin')
  orgSlug,          // string | null
  sessionId,        // string | null
  sessionClaims,    // JWT claims object
  has,              // (params) => boolean - RBAC check
  isAuthenticated,  // boolean (convenience check)
} = await auth()
```

## Client Component Pattern

```tsx
'use client'
import { useUser, useAuth } from '@clerk/nextjs'

export function UserDashboard() {
  const { isLoaded, isSignedIn, user } = useUser()
  const { signOut, has } = useAuth()

  if (!isLoaded) return <div>Loading...</div>
  if (!isSignedIn) return <div>Not signed in</div>

  return (
    <div>
      <p>Hello, {user.firstName}!</p>
      <button onClick={() => signOut()}>Sign out</button>
    </div>
  )
}
```

## Hybrid Pattern (Server Data + Client Interactivity)

```tsx
// Server Component: fetch data
import { currentUser } from '@clerk/nextjs/server'
import { ProfileForm } from './ProfileForm'

export default async function ProfilePage() {
  const user = await currentUser()
  if (!user) return <div>Please sign in</div>
  return <ProfileForm initialData={{ firstName: user.firstName }} />
}
```

```tsx
// Client Component: handle interactions
'use client'
import { useUser } from '@clerk/nextjs'

export function ProfileForm({ initialData }: { initialData: { firstName: string | null } }) {
  const { user } = useUser()
  // Use initialData for SSR, user for client-side updates
  return <form>...</form>
}
```

## Control Components

Render conditionally based on auth state (works in both Server and Client Components):

```tsx
import { SignedIn, SignedOut, SignInButton, UserButton } from '@clerk/nextjs'

export function Header() {
  return (
    <header>
      <SignedIn>
        <UserButton />
      </SignedIn>
      <SignedOut>
        <SignInButton />
      </SignedOut>
    </header>
  )
}
```

## Protect Component

Gate content with `<Protect>`:

```tsx
import { Protect } from '@clerk/nextjs'

export function AdminPanel() {
  return (
    <Protect role="org:admin" fallback={<p>Admin access required</p>}>
      <AdminDashboard />
    </Protect>
  )
}
```

```tsx
<Protect permission="org:invoices:create" fallback={<p>No access</p>}>
  <CreateInvoiceForm />
</Protect>
```

[Docs](https://clerk.com/docs/reference/nextjs/auth)
