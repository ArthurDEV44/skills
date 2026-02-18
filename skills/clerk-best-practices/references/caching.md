# Auth-Aware Caching

**CRITICAL**: Cache keys MUST include userId or orgId to prevent data leaking between users.

## User-Scoped Cache

```typescript
import { auth } from '@clerk/nextjs/server'
import { unstable_cache } from 'next/cache'

export default async function ProfilePage() {
  const { userId } = await auth()
  if (!userId) return <div>Not signed in</div>

  const getUserData = unstable_cache(
    () => db.users.findUnique({ where: { id: userId } }),
    [`user-${userId}`],
    { revalidate: 60, tags: [`user-${userId}`] }
  )

  const userData = await getUserData()
  return <div>{userData?.name}</div>
}
```

## Org-Scoped Cache

```typescript
const { orgId } = await auth()

const getOrgData = unstable_cache(
  () => db.orgData.findMany({ where: { organizationId: orgId } }),
  [`org-${orgId}-data`],
  { revalidate: 300, tags: [`org-${orgId}`] }
)
```

## Revalidate After Mutations

```typescript
'use server'
import { revalidateTag } from 'next/cache'
import { auth } from '@clerk/nextjs/server'

export async function updateProfile(formData: FormData) {
  const { userId } = await auth()
  if (!userId) throw new Error('Unauthorized')

  await db.users.update({
    where: { id: userId },
    data: { name: formData.get('name') as string },
  })

  revalidateTag(`user-${userId}`)
}
```

## Rules

1. **Always include userId/orgId in cache key** - `[`user-${userId}-posts`]` not `['posts']`
2. **Use tags for targeted revalidation** - `revalidateTag(`user-${userId}`)` after mutations
3. **Never cache auth state itself** - `auth()` is already optimized per-request
4. **Scope revalidation narrowly** - Don't `revalidatePath('/')` when you can tag

[Docs](https://nextjs.org/docs/app/building-your-application/caching)
