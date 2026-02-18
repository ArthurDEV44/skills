# Server Actions

**Server Actions are public HTTP endpoints.** Always verify authentication at the top of every action.

## Basic Protection

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'
import { revalidatePath } from 'next/cache'

export async function createPost(formData: FormData) {
  const { userId } = await auth()
  if (!userId) throw new Error('Unauthorized')

  const title = formData.get('title') as string
  await db.posts.create({ data: { title, authorId: userId } })
  revalidatePath('/posts')
}
```

## With `isAuthenticated` Check

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'

export async function addToCart(formData: FormData) {
  const { isAuthenticated } = await auth()
  if (!isAuthenticated) {
    throw new Error('You must be signed in')
  }

  console.log('add item', formData)
}
```

## Permission Check (RBAC)

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'

export async function deleteProject(projectId: string) {
  const { userId, has } = await auth()
  if (!userId) throw new Error('Unauthorized')

  const canDelete = has({ permission: 'org:project:delete' })
  if (!canDelete) throw new Error('Missing permission')

  await db.projects.delete({ where: { id: projectId } })
}
```

## Org + Role Check (B2B)

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'

export async function createTeamProject(formData: FormData) {
  const { userId, orgId } = await auth()
  if (!userId || !orgId) throw new Error('Must be in an organization')

  const { has } = await auth()
  if (!has({ role: 'org:admin' })) throw new Error('Admin access required')

  const name = formData.get('name') as string
  await db.projects.create({ data: { name, organizationId: orgId } })
}
```

## Full Auth + Permission Pattern (Returning HTTP-like Status)

```typescript
'use server'
import { auth } from '@clerk/nextjs/server'

export async function manageTeamSettings(formData: FormData) {
  const { isAuthenticated, has, userId } = await auth()

  if (!isAuthenticated) {
    return { error: 'User is not signed in', status: 401 }
  }

  if (!has({ permission: 'org:team_settings:manage' })) {
    return { error: 'Missing permissions', status: 403 }
  }

  // Proceed with authorized logic
  return { success: true }
}
```

## Using with Client Components

Pass Server Actions as props to Client Components:

```tsx
// actions.ts
'use server'
import { auth, currentUser } from '@clerk/nextjs/server'

export async function addHobby(formData: FormData) {
  const { isAuthenticated } = await auth()
  const user = await currentUser()

  if (!isAuthenticated) throw new Error('Sign in required')

  const data = {
    hobby: formData.get('hobby'),
    userId: user.id,
  }
  // save to database...
}
```

```tsx
// page.tsx (Server Component)
import { addHobby } from './actions'
import { HobbyForm } from './HobbyForm'

export default function Page() {
  return <HobbyForm addHobby={addHobby} />
}
```

```tsx
// HobbyForm.tsx (Client Component)
'use client'

export function HobbyForm({ addHobby }: { addHobby: (fd: FormData) => Promise<void> }) {
  return (
    <form action={addHobby}>
      <input type="text" name="hobby" />
      <button type="submit">Add</button>
    </form>
  )
}
```

## Rules

1. **Every Server Action must check auth** - No exceptions, even if middleware protects the route
2. **Use `auth()` not `currentUser()`** when you only need userId - Avoids unnecessary API calls
3. **Return error objects instead of throwing** when the caller needs to handle errors gracefully
4. **Call `revalidatePath()` or `revalidateTag()`** after mutations to bust cache

[Docs](https://clerk.com/docs/reference/nextjs/app-router/server-actions)
