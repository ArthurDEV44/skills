# Authentication & Row-Level Security (RLS)

Neon provides two approaches for authentication: **Neon Authorize** (bring your own auth + RLS) and **Neon Auth** (built-in authentication with Stack Auth).

## Neon Authorize — Row-Level Security with JWT

Neon Authorize integrates with third-party auth providers (Clerk, Auth.js, Stytch, etc.) to enforce authorization at the database level using Postgres Row-Level Security (RLS).

### How It Works

1. User authenticates with your auth provider (e.g., Clerk)
2. Auth provider issues a JWT
3. Your app passes the JWT to Neon via connection parameters
4. Neon validates the JWT against the provider's JWKS endpoint
5. RLS policies use JWT claims to filter data

### Setup — SQL

```sql
-- 1. Enable RLS on tables
ALTER TABLE todos ENABLE ROW LEVEL SECURITY;

-- 2. Create policies using JWT claims
-- The JWT 'sub' claim (user ID) is available via current_setting()
CREATE POLICY "Users can view own todos"
  ON todos FOR SELECT
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY "Users can insert own todos"
  ON todos FOR INSERT
  WITH CHECK (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY "Users can update own todos"
  ON todos FOR UPDATE
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY "Users can delete own todos"
  ON todos FOR DELETE
  USING (user_id = current_setting('app.current_user_id')::uuid);

-- Role-based policy example
CREATE POLICY "Admins can view all"
  ON todos FOR SELECT
  USING (
    current_setting('app.current_user_role') = 'admin'
    OR user_id = current_setting('app.current_user_id')::uuid
  );
```

### Setup — Configure JWKS in Neon

Register your auth provider's JWKS URL in the Neon Console or via Terraform:

```hcl
resource "neon_jwks_url" "clerk_jwks" {
  project_id    = neon_project.app.id
  role_names    = [neon_project.app.database_user]
  provider_name = "clerk"
  jwks_url      = "https://your-app.clerk.accounts.dev/.well-known/jwks.json"
}
```

### Passing JWT Claims to Postgres

```typescript
// src/db/index.ts
import { Pool } from '@neondatabase/serverless'
import { auth } from '@clerk/nextjs/server'

export async function getAuthenticatedDb() {
  const { getToken, userId } = await auth()
  const token = await getToken()

  const pool = new Pool({ connectionString: process.env.DATABASE_URL })
  const client = await pool.connect()

  // Set the user context for RLS policies
  await client.query("SELECT set_config('app.current_user_id', $1, true)", [userId])

  return { client, pool }
}
```

## Auth.js (NextAuth) Integration

```typescript
// auth.ts
import NextAuth from 'next-auth'
import Resend from 'next-auth/providers/resend'
import PostgresAdapter from '@auth/pg-adapter'
import { Pool } from '@neondatabase/serverless'

// IMPORTANT: Create Pool inside the factory function, not at module scope
export const { handlers, auth, signIn, signOut } = NextAuth(() => {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL })
  return {
    adapter: PostgresAdapter(pool),
    providers: [Resend({ from: 'noreply@example.com' })],
  }
})
```

```typescript
// app/api/auth/[...nextauth]/route.ts
import { handlers } from '@/auth'
export const { GET, POST } = handlers
```

## Clerk + Drizzle + Neon

```typescript
// app/actions.ts
'use server'

import { currentUser } from '@clerk/nextjs/server'
import { db } from '@/db'
import { userMessages } from '@/db/schema'
import { eq } from 'drizzle-orm'
import { redirect } from 'next/navigation'

export async function createMessage(formData: FormData) {
  const user = await currentUser()
  if (!user) throw new Error('Unauthorized')

  const message = formData.get('message') as string
  await db.insert(userMessages).values({
    userId: user.id,
    message,
  })
  redirect('/')
}

export async function deleteMessage() {
  const user = await currentUser()
  if (!user) throw new Error('Unauthorized')

  await db.delete(userMessages).where(eq(userMessages.userId, user.id))
  redirect('/')
}
```

## Neon Auth — Built-in Authentication

Neon Auth provides authentication out of the box via Stack Auth, including pre-built React/Next.js components.

### Quick Start

```bash
# Initialize Neon Auth in your Next.js project
npx @stackframe/init-stack@latest --no-browser
```

This sets up authentication routes, layout wrappers, and handlers automatically.

### React Components

```tsx
import {
  StackProvider,
  StackTheme,
  StackHandler,
} from '@stackframe/stack'

// Wrap your app
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <StackProvider>
      <StackTheme>
        {children}
      </StackTheme>
    </StackProvider>
  )
}
```

### Sign In / Sign Up Pages

```tsx
// app/sign-in/page.tsx
import { SignIn } from '@stackframe/stack'

export default function SignInPage() {
  return <SignIn />
}
```

```tsx
// app/sign-up/page.tsx
import { SignUp } from '@stackframe/stack'

export default function SignUpPage() {
  return <SignUp />
}
```

### User Profile & Auth State

```tsx
'use client'

import { useUser, useStackApp, UserButton } from '@stackframe/stack'

export default function ProfilePage() {
  const user = useUser({ or: 'redirect' })
  const app = useStackApp()

  return (
    <div>
      <UserButton />
      <h1>Welcome, {user.displayName || 'User'}</h1>
      <p>Email: {user.primaryEmail}</p>
      <button onClick={() => user.signOut()}>Sign Out</button>
    </div>
  )
}
```

### Available Components

| Component | Purpose |
|-----------|---------|
| `<SignIn />` | Sign-in form |
| `<SignUp />` | Sign-up form |
| `<UserButton />` | User avatar with dropdown menu |
| `<OAuthButton provider="google" />` | Single OAuth provider button |
| `<OAuthButtonGroup />` | All configured OAuth providers |
| `<AccountSettings />` | User account settings page |
| `<SelectedTeamSwitcher />` | Team/org switcher |
| `<StackHandler />` | Route handler for auth callbacks |

### Backend JWT Verification

```typescript
import * as jose from 'jose'

// Cache this — refresh with low frequency
const jwks = jose.createRemoteJWKSet(
  new URL('https://api.stack-auth.com/api/v1/projects/<project-id>/.well-known/jwks.json')
)

export async function verifyToken(accessToken: string) {
  try {
    const { payload } = await jose.jwtVerify(accessToken, jwks)
    return { userId: payload.sub, valid: true }
  } catch {
    return { userId: null, valid: false }
  }
}
```

### Create Neon Auth via API

```bash
curl --request POST \
  --url 'https://console.neon.tech/api/v2/projects/auth/create' \
  --header "Authorization: Bearer $NEON_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
    "auth_provider": "stack",
    "project_id": "<project-id>",
    "branch_id": "<branch-id>",
    "database_name": "neondb",
    "role_name": "neondb_owner"
  }'
```

## RLS Best Practices

1. **Always enable RLS** on tables with user data — even if you also validate in application code
2. **Use `current_setting()` for JWT claims** — Access user context set by Neon Authorize or your app
3. **Create a policy per operation** — Separate SELECT, INSERT, UPDATE, DELETE policies for fine-grained control
4. **Test policies with different roles** — Verify that unauthorized access is blocked
5. **Use `WITH CHECK` for write policies** — Prevents users from inserting/updating data they shouldn't own
6. **Don't forget the default deny** — When RLS is enabled, all access is denied unless a policy explicitly allows it
7. **Index columns used in policies** — RLS conditions are evaluated per-row; index `user_id` and similar columns
