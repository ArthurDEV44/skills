# Setup and Configuration

## Installation

```bash
npm install @clerk/nextjs
```

## Environment Variables

```env
# Required
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

# Optional - custom sign-in/up URLs
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

# Optional - after sign-in/up redirects
NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=/dashboard
NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL=/dashboard

# Webhooks
CLERK_WEBHOOK_SIGNING_SECRET=whsec_...
```

**Security**: Never expose `CLERK_SECRET_KEY` in client code. Only `NEXT_PUBLIC_*` prefixed keys are safe.

## Keyless Development

On first SDK initialization, Clerk auto-generates dev keys and shows a "Claim your application" popover. No manual key setup needed for development.

## ClerkProvider

Must wrap the entire app in the root layout:

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

### With Dynamic Config

```tsx
<ClerkProvider
  signInUrl="/sign-in"
  signUpUrl="/sign-up"
  signInFallbackRedirectUrl="/dashboard"
  signUpFallbackRedirectUrl="/onboarding"
>
  {children}
</ClerkProvider>
```

## Sign-In / Sign-Up Pages

### Using Clerk Components

```tsx
// app/sign-in/[[...sign-in]]/page.tsx
import { SignIn } from '@clerk/nextjs'

export default function SignInPage() {
  return <SignIn />
}
```

```tsx
// app/sign-up/[[...sign-up]]/page.tsx
import { SignUp } from '@clerk/nextjs'

export default function SignUpPage() {
  return <SignUp />
}
```

### Catch-all Routes

Use `[[...slug]]` to handle sub-paths like `/sign-in/factor-one`, `/sign-in/sso-callback`.

## Prebuilt UI Components

| Component | Purpose |
|-----------|---------|
| `<SignIn />` | Sign-in form |
| `<SignUp />` | Sign-up form |
| `<UserButton />` | User avatar with dropdown menu |
| `<UserProfile />` | Full user profile management |
| `<OrganizationSwitcher />` | Switch between organizations |
| `<OrganizationProfile />` | Manage organization settings |
| `<SignedIn>` | Render children only when signed in |
| `<SignedOut>` | Render children only when signed out |

```tsx
import { SignedIn, SignedOut, UserButton, SignInButton } from '@clerk/nextjs'

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

## Migration from Other Auth Providers

Check `package.json` for existing auth:
- `next-auth` / `@auth/core` -> NextAuth/Auth.js
- `@supabase/supabase-js` -> Supabase Auth
- `firebase` -> Firebase Auth
- `@auth0/nextjs-auth0` -> Auth0

Migration steps:
1. Audit all auth touchpoints (pages, middleware, DB, OAuth)
2. Export users via Clerk Backend API (password hashes upgraded transparently)
3. Store legacy IDs as `external_id` in Clerk
4. Choose big bang (simpler) or trickle migration (safer)

Guide: https://clerk.com/docs/guides/development/migrating/overview

[Docs](https://clerk.com/docs/nextjs/getting-started/quickstart)
