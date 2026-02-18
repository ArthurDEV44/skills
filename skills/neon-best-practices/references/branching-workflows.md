# Database Branching Workflows

Neon's copy-on-write branching creates instant, full database copies (schema + data) regardless of size. Branches are ideal for development, testing, and preview deployments.

## Core Concepts

- **Branch** = a full copy of your database at a point in time (schema + data by default, schema-only optional)
- **Copy-on-write** = branches are instant and storage-efficient; unchanged data is shared
- **Parent branch** = the branch a new branch is created from (typically `main` or `production`)
- **Branch reset** = resets a branch to match its parent's current state

## Neon CLI Branch Management

```bash
# Install Neon CLI
npm install -g neonctl

# Authenticate
neon auth

# Create a branch from production
neon branches create --name dev/feature-auth --project-id <project-id>

# Create a schema-only branch (no data copied)
neon branches create --name dev/feature-auth --project-id <project-id> --type schema-only

# Get the connection string for a branch
neon connection-string dev/feature-auth --project-id <project-id>

# Get pooled connection string
neon connection-string dev/feature-auth --project-id <project-id> --pooled

# Reset a branch to its parent's latest state
neon branches reset dev/feature-auth --parent

# Delete a branch
neon branches delete dev/feature-auth --project-id <project-id>

# List all branches
neon branches list --project-id <project-id>
```

## Multi-Branch Environment Setup

```typescript
// src/db/index.ts — connect to the right branch based on environment
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'
import * as schema from './schema'

const getBranchUrl = () => {
  switch (process.env.NODE_ENV) {
    case 'development':
      return process.env.DEV_DATABASE_URL
    case 'test':
      return process.env.TEST_DATABASE_URL
    default:
      return process.env.DATABASE_URL
  }
}

const sql = neon(getBranchUrl()!)
export const db = drizzle(sql, { schema })
```

```env
# .env.local
DATABASE_URL="postgresql://...@ep-xxx-pooler.../dbname"           # production
DEV_DATABASE_URL="postgresql://...@ep-xxx-pooler.../dbname"       # dev branch
TEST_DATABASE_URL="postgresql://...@ep-xxx-pooler.../dbname"      # test branch
```

## GitHub Actions — Full PR Branch Workflow

This workflow creates a Neon branch per pull request, optionally runs migrations, posts a schema diff comment, and cleans up on PR close.

### Required Secrets & Variables

- `secrets.NEON_API_KEY` — Neon API key
- `vars.NEON_PROJECT_ID` — Neon project ID

### Create & Delete Branch on PR

```yaml
name: Neon Branch per PR

on:
  pull_request:
    types: [opened, reopened, synchronize, closed]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}

jobs:
  setup:
    name: Get Branch Name
    outputs:
      branch: ${{ steps.branch_name.outputs.current_branch }}
    runs-on: ubuntu-latest
    steps:
      - uses: tj-actions/branch-names@v8
        id: branch_name

  create_branch:
    name: Create Neon Branch
    needs: setup
    if: github.event.action != 'closed'
    runs-on: ubuntu-latest
    outputs:
      db_url: ${{ steps.create.outputs.db_url_with_pooler }}
    steps:
      - name: Create Neon Branch
        id: create
        uses: neondatabase/create-branch-action@v5
        with:
          project_id: ${{ vars.NEON_PROJECT_ID }}
          branch_name: preview/pr-${{ github.event.number }}-${{ needs.setup.outputs.branch }}
          api_key: ${{ secrets.NEON_API_KEY }}

      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm ci

      - name: Run Migrations
        run: npx drizzle-kit push
        env:
          DATABASE_URL_UNPOOLED: ${{ steps.create.outputs.db_url }}

  schema_diff:
    name: Schema Diff
    needs: [setup, create_branch]
    if: github.event.action != 'closed'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - name: Post Schema Diff
        uses: neondatabase/schema-diff-action@v1
        with:
          project_id: ${{ vars.NEON_PROJECT_ID }}
          compare_branch: preview/pr-${{ github.event.number }}-${{ needs.setup.outputs.branch }}
          api_key: ${{ secrets.NEON_API_KEY }}

  delete_branch:
    name: Delete Neon Branch
    needs: setup
    if: github.event.action == 'closed'
    runs-on: ubuntu-latest
    steps:
      - uses: neondatabase/delete-branch-action@v3
        with:
          project_id: ${{ vars.NEON_PROJECT_ID }}
          branch: preview/pr-${{ github.event.number }}-${{ needs.setup.outputs.branch }}
          api_key: ${{ secrets.NEON_API_KEY }}
```

### Reset Branch on Label

Add a label-triggered reset for refreshing preview branches:

```yaml
  reset_branch:
    name: Reset Neon Branch
    needs: setup
    if: |
      contains(github.event.pull_request.labels.*.name, 'Reset Neon Branch') &&
      github.event.action != 'closed'
    runs-on: ubuntu-latest
    steps:
      - uses: neondatabase/reset-branch-action@v1
        with:
          project_id: ${{ vars.NEON_PROJECT_ID }}
          parent: true
          branch: preview/pr-${{ github.event.number }}-${{ needs.setup.outputs.branch }}
          api_key: ${{ secrets.NEON_API_KEY }}
```

## Available Neon GitHub Actions

| Action | Purpose |
|--------|---------|
| `neondatabase/create-branch-action@v5` | Create a branch for preview/testing |
| `neondatabase/delete-branch-action@v3` | Clean up branch on PR close |
| `neondatabase/reset-branch-action@v1` | Reset branch to parent's latest state |
| `neondatabase/schema-diff-action@v1` | Post schema diff as PR comment |

## Vercel Preview Branching

The Neon-managed Vercel integration creates a branch per preview deployment automatically.

### Setup

1. Install the Neon integration in Vercel
2. Connect your project and toggle **Preview** branching on
3. Enable **Resource must be active before deployment**

### How It Works

1. Push to feature branch → Vercel starts preview deployment
2. Vercel webhook → Neon creates `preview/<git-branch>` branch
3. Connection string injected as env vars for that deployment only
4. (Optional) Migrations run in build step

### Run Migrations in Preview Builds

In **Vercel Dashboard → Settings → Build and Deployment Settings**, override the build command:

```bash
# With Drizzle
npx drizzle-kit push && npm run build

# With Prisma
npx prisma migrate deploy && npm run build
```

## Branching Best Practices

1. **Name branches consistently** — Use `preview/pr-{number}-{branch}` for PRs, `dev/{feature}` for development
2. **Reset instead of recreate** — Use `neon branches reset --parent` to refresh a branch without changing its connection string
3. **Schema-only branches for sensitive data** — Create schema-only branches when production data should not be copied
4. **Clean up stale branches** — Automate deletion on PR close via GitHub Actions
5. **Run migrations on preview branches** — Add migration commands to your build step so schema matches code
6. **Use schema-diff** — Post schema diff comments on PRs for easy review of database changes
