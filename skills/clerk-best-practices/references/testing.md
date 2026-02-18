# E2E Testing

## Mental Model

Test auth = isolated session state. Each test needs fresh auth context.

- `clerkSetup()` initializes test environment
- `setupClerkTestingToken()` bypasses bot detection
- `storageState` persists auth between tests for speed

## Playwright Setup

### Install

```bash
npm install @clerk/testing --save-dev
```

### Global Setup

```typescript
// global-setup.ts
import { clerkSetup } from '@clerk/testing/playwright'

export default async function globalSetup() {
  await clerkSetup()
}
```

### Playwright Config

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test'

export default defineConfig({
  globalSetup: require.resolve('./global-setup'),
  use: {
    baseURL: 'http://localhost:3000',
  },
})
```

### Test Example

```typescript
import { test, expect } from '@playwright/test'
import { setupClerkTestingToken } from '@clerk/testing/playwright'

test('authenticated user can access dashboard', async ({ page }) => {
  await setupClerkTestingToken({ page })
  await page.goto('/dashboard')
  await expect(page.locator('h1')).toContainText('Dashboard')
})
```

### Save Auth State for Speed

```typescript
// Save auth state after sign-in
await page.context().storageState({ path: 'auth-state.json' })

// Reuse in other tests
test.use({ storageState: 'auth-state.json' })
```

## Cypress Setup

### Install

```bash
npm install @clerk/testing --save-dev
```

### Support File

```typescript
// cypress/support/e2e.ts
import { addClerkCommands } from '@clerk/testing/cypress'

addClerkCommands({ Cypress, cy })
```

### Test Example

```typescript
describe('Dashboard', () => {
  it('shows dashboard for authenticated user', () => {
    cy.clerkSignIn({ strategy: 'password', identifier: 'test@example.com', password: 'test123' })
    cy.visit('/dashboard')
    cy.get('h1').should('contain', 'Dashboard')
  })
})
```

## Best Practices

1. **Always use test API keys** - `pk_test_*` and `sk_test_*` only
2. **Call `setupClerkTestingToken()`** before navigating to auth pages
3. **Use `storageState`** to persist auth and speed up tests
4. **Wait for Clerk components** with `page.waitForSelector('[data-clerk-component]')`
5. **Create dedicated test users** in Clerk Dashboard for E2E tests
6. **Never use production keys** in test environments

## Anti-Patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| Production keys in tests | Security risk | Use `pk_test_*` keys |
| No `setupClerkTestingToken()` | Auth fails | Call before navigation |
| UI-based sign-in every test | Slow | Use `storageState` |
| Shared test user state | Flaky tests | Isolate test users |

[Docs](https://clerk.com/docs/guides/development/testing/overview)
