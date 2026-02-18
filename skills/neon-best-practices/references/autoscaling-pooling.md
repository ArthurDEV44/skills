# Autoscaling, Scale-to-Zero & Connection Pooling

## Compute Units (CU)

Neon computes are measured in **Compute Units (CU)**. Each CU provides a specific amount of vCPU and RAM. Autoscaling adjusts CU allocation based on workload.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling_limit_min_cu` | Minimum compute size | 0.25 (Free) |
| `autoscaling_limit_max_cu` | Maximum compute size | Plan-dependent |
| `suspend_timeout_seconds` | Inactivity before auto-suspend | 300 (Free tier) |

### Configure via Neon API

```bash
# Create project with autoscaling config
curl --request POST \
  --url https://console.neon.tech/api/v2/projects \
  --header "Authorization: Bearer $NEON_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
  "project": {
    "default_endpoint_settings": {
      "autoscaling_limit_min_cu": 1,
      "autoscaling_limit_max_cu": 4,
      "suspend_timeout_seconds": 600
    },
    "pg_version": 16
  }
}'
```

```bash
# Update existing compute endpoint
curl --request PATCH \
  --url https://console.neon.tech/api/v2/projects/{project-id}/endpoints/{endpoint-id} \
  --header "Authorization: Bearer $NEON_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
  "endpoint": {
    "autoscaling_limit_min_cu": 0.5,
    "autoscaling_limit_max_cu": 4,
    "suspend_timeout_seconds": 300
  }
}'
```

### Configure via Terraform

```hcl
resource "neon_endpoint" "production" {
  project_id = neon_project.app.id
  branch_id  = neon_branch.main.id
  type       = "read_write"

  autoscaling_limit_min_cu = 1
  autoscaling_limit_max_cu = 4
  suspend_timeout_seconds  = 600

  pooler_enabled = true
}
```

## Scale-to-Zero

When a compute has no activity for `suspend_timeout_seconds`, Neon suspends it to save costs. The next query triggers a **cold start** that reactivates the compute.

### Cold Start Performance

- **US East (Ohio):** ~500ms (fastest — Neon Control Plane is hosted here)
- **Other regions:** Slightly higher, improving as Neon deploys regional control planes
- **Optimizations applied:** Compute pools (pre-started instances), config optimization, IP caching, concurrency improvements

### Configuration by Plan

| Plan | Auto-suspend | Configurable? |
|------|-------------|---------------|
| Free | 5 minutes | No |
| Launch | Configurable | Yes |
| Scale | Configurable or disabled | Yes |
| Business | Configurable or disabled | Yes |

### Recommendations

- **Development:** Keep auto-suspend enabled (saves costs)
- **Production (latency-sensitive):** Disable auto-suspend on paid plans by setting `suspend_timeout_seconds: 0`
- **Production (cost-sensitive):** Set a higher timeout (600-3600s) to reduce cold starts while saving on idle time

## Connection Pooling Deep Dive

### PgBouncer Configuration

Neon's built-in PgBouncer runs in **transaction mode** with these defaults:

```ini
[pgbouncer]
pool_mode = transaction
max_client_conn = 10000
default_pool_size = 0.9 * max_connections
max_prepared_statements = 1000
query_wait_timeout = 120
```

### Pooled vs Direct Connections

```
# Pooled (add -pooler to endpoint)
postgresql://user:pass@ep-cool-darkness-123456-pooler.region.aws.neon.tech/dbname?sslmode=require

# Direct (no -pooler)
postgresql://user:pass@ep-cool-darkness-123456.region.aws.neon.tech/dbname?sslmode=require
```

### Transaction Mode Caveats

In transaction mode, each SQL transaction can run on a different backend connection. This means:

1. **`SET` commands don't persist** — `SET search_path TO myschema` only lasts for one transaction

   **Workaround:** Use `ALTER ROLE your_role SET search_path TO schema1, schema2;`

2. **Prepared statements** — PgBouncer tracks up to 1,000 prepared statements per client

3. **`LISTEN/NOTIFY`** — Does not work reliably over pooled connections; use direct connections

4. **`pg_dump`** — Issues many `SET` commands; always use direct connections

5. **Advisory locks** — Session-level advisory locks are unreliable; use transaction-level ones (`pg_advisory_xact_lock`)

### Environment Variable Pattern

```env
# .env.local — Next.js
# Application queries (pooled)
DATABASE_URL="postgresql://...@ep-xxx-pooler.region.aws.neon.tech/dbname?sslmode=require"

# Migrations and admin (direct)
DATABASE_URL_UNPOOLED="postgresql://...@ep-xxx.region.aws.neon.tech/dbname?sslmode=require"
```

```typescript
// drizzle.config.ts — use direct connection
export default {
  dbCredentials: { url: process.env.DATABASE_URL_UNPOOLED! },
  // ...
} satisfies Config

// src/db/index.ts — use pooled connection
const sql = neon(process.env.DATABASE_URL!)
```

## neonConfig Advanced Options

```typescript
import { neonConfig } from '@neondatabase/serverless'

// Pipeline startup messages for faster connection (default: 'password')
neonConfig.pipelineConnect = 'password'

// Batch multiple writes into single WebSocket frames
neonConfig.coalesceWrites = true

// Use secure WebSocket (wss://) — default: true
neonConfig.useSecureWebSocket = true

// Route Pool.query() over HTTP fetch for lower latency
neonConfig.poolQueryViaFetch = true

// Custom fetch endpoint (useful for local dev proxy)
neonConfig.fetchEndpoint = (host, port, options) => {
  if (process.env.NODE_ENV === 'development') {
    return `http://localhost:3000/api/db`
  }
  return `https://${host}/sql`
}
```

### Per-Client Overrides

```typescript
import { Client } from '@neondatabase/serverless'

const client = new Client(process.env.DATABASE_URL)
client.neonConfig.pipelineConnect = false  // Disable for this client only
await client.connect()
```

## Monitoring & Troubleshooting

### Connection Issues

- **"too many connections"** — You're using direct connections in serverless. Switch to pooled (`-pooler`)
- **"SSL SYSCALL error"** — Compute was suspended mid-connection. Handle reconnection in your application or increase `suspend_timeout_seconds`
- **"relation does not exist"** — `SET search_path` was lost in transaction mode. Qualify schema names or use `ALTER ROLE`

### Performance Tips

1. **Use `neon()` HTTP driver** for stateless queries — lower latency than WebSocket Pool
2. **Keep connections short** — In serverless, create Pool per request, end it in `finally`
3. **Place compute in the same region as your app** — Minimize network round trips
4. **Use read replicas** for read-heavy workloads — Create read-only endpoints on the same branch
