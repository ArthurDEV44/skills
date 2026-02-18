# Webhooks

> Webhooks are asynchronous. Use for background tasks (sync, notifications), not synchronous flows.

## Setup

1. Create endpoint at `app/api/webhooks/route.ts`
2. Use `verifyWebhook(req)` from `@clerk/nextjs/webhooks`
3. Dashboard -> Webhooks -> Add Endpoint with your URL
4. Set `CLERK_WEBHOOK_SIGNING_SECRET` in `.env`
5. **Make the route public** - exclude from middleware protection

## Make Route Public

```typescript
// middleware.ts
const isPublicRoute = createRouteMatcher([
  '/',
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/webhooks(.*)',  // <-- Must be public
])
```

## Webhook Handler

```typescript
// app/api/webhooks/route.ts
import { verifyWebhook } from '@clerk/nextjs/webhooks'
import { NextRequest } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const evt = await verifyWebhook(req)

    const { id } = evt.data
    const eventType = evt.type
    console.log(`Webhook ${id}: ${eventType}`)

    return new Response('OK', { status: 200 })
  } catch (err) {
    console.error('Webhook verification failed:', err)
    return new Response('Error', { status: 400 })
  }
}
```

## Sync Users to Database

```typescript
import { verifyWebhook } from '@clerk/nextjs/webhooks'
import { NextRequest } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const evt = await verifyWebhook(req)

    if (evt.type === 'user.created') {
      const { id, email_addresses, first_name, last_name, image_url } = evt.data
      const email = email_addresses.find(
        (e) => e.id === evt.data.primary_email_address_id
      )?.email_address

      await db.users.create({
        data: { clerkId: id, email, firstName: first_name, lastName: last_name, imageUrl: image_url },
      })
    }

    if (evt.type === 'user.updated') {
      const { id, first_name, last_name, image_url } = evt.data
      await db.users.update({
        where: { clerkId: id },
        data: { firstName: first_name, lastName: last_name, imageUrl: image_url },
      })
    }

    if (evt.type === 'user.deleted') {
      const { id } = evt.data
      await db.users.delete({ where: { clerkId: id } })
    }

    return new Response('OK', { status: 200 })
  } catch (err) {
    console.error('Webhook error:', err)
    return new Response('Error', { status: 400 })
  }
}
```

## Supported Events

**User**: `user.created`, `user.updated`, `user.deleted`

**Organization**: `organization.created`, `organization.updated`, `organization.deleted`

**Membership**: `organizationMembership.created`, `organizationMembership.updated`, `organizationMembership.deleted`

**Invitation**: `organizationInvitation.created`, `organizationInvitation.accepted`, `organizationInvitation.revoked`

**Session**: `session.created`, `session.ended`, `session.removed`, `session.revoked`

**Roles/Permissions**: `role.created`, `role.updated`, `role.deleted`, `permission.created`, `permission.updated`, `permission.deleted`

## When to Sync

**Do sync**: Social features needing other users' data, custom fields, notifications, integrations.

**Don't sync**: Only need current user (use session), no custom fields, need immediate consistency (webhooks are eventual).

## Reliability

- **Retries**: Svix retries failed webhooks for up to 3 days
- **Success**: Return 2xx to acknowledge receipt
- **Failure**: Return 4xx/5xx to trigger retry
- **Replay**: Failed webhooks can be replayed from Dashboard
- **Idempotency**: Handle duplicate deliveries gracefully (use upsert)

## Queue Long Operations

Return 200 immediately, process asynchronously:

```typescript
export async function POST(req: NextRequest) {
  const evt = await verifyWebhook(req)

  // Queue work instead of doing it inline
  await queue.enqueue('process-webhook', evt)

  return new Response('OK', { status: 200 })
}
```

## Local Development

Use ngrok to tunnel localhost:

```bash
ngrok http 3000
```

Add the ngrok URL as endpoint in Dashboard -> Webhooks.

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Verification fails | Wrong import | Use `@clerk/nextjs/webhooks` not svix directly |
| 401 on webhook route | Route protected by middleware | Add `/api/webhooks(.*)` to public routes |
| Duplicate DB entries | Only handling `user.created` | Use upsert or handle all three events |
| Timeouts | Handler too slow | Queue async work, return 200 fast |
| 404 on endpoint | Wrong file path | Must be `app/api/webhooks/route.ts` |

[Docs](https://clerk.com/docs/guides/development/webhooks/overview)
