---
name: mcp-server-dev
description: "Build MCP (Model Context Protocol) servers in TypeScript with @modelcontextprotocol/sdk. Use when writing, reviewing, or refactoring MCP server code: (1) Creating MCP servers with McpServer, (2) Registering tools with registerTool, inputSchema, outputSchema, Zod validation, (3) Defining resources and resource templates, (4) Defining prompts with arguments, (5) Transports: StdioServerTransport, NodeStreamableHTTPServerTransport, SSE, (6) Tool annotations (readOnlyHint, destructiveHint, idempotentHint), (7) Error handling and isError responses, (8) Dynamic tool loading and tool list change notifications, (9) Middleware patterns for MCP tools, (10) Testing MCP servers with vitest, (11) Publishing and configuring for Claude Code, Cursor, Windsurf, (12) Any @modelcontextprotocol/sdk imports."
---

# MCP Server Development — TypeScript SDK

## What is MCP

The **Model Context Protocol** is an open standard for connecting AI assistants (Claude, Cursor, etc.) to external tools and data. An MCP server exposes **tools**, **resources**, and **prompts** that clients can discover and invoke.

## Quick Setup

```bash
bun init
bun add @modelcontextprotocol/sdk zod
```

```typescript
// src/index.ts
import { McpServer } from '@modelcontextprotocol/server'
import { StdioServerTransport } from '@modelcontextprotocol/node'
import * as z from 'zod/v4'

const server = new McpServer({
  name: 'my-mcp-server',
  version: '1.0.0',
})

// Register a tool
server.registerTool(
  'greet',
  {
    description: 'Greet a user by name',
    inputSchema: z.object({
      name: z.string().describe('Name of the person to greet'),
    }),
  },
  async ({ name }) => ({
    content: [{ type: 'text', text: `Hello, ${name}!` }],
  })
)

// Start server with stdio transport
const transport = new StdioServerTransport()
await server.connect(transport)
```

```json
// package.json essentials
{
  "type": "module",
  "bin": { "my-mcp-server": "./dist/index.js" },
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js"
  }
}
```

## Three Primitives

| Primitive | Purpose | Client Action | Example |
|-----------|---------|---------------|---------|
| **Tools** | Execute actions, return results | LLM decides when to call | API calls, calculations, file ops |
| **Resources** | Expose read-only data | User selects, attached to context | Files, DB records, API data |
| **Prompts** | Reusable prompt templates | User selects from menu | Code review template, debug prompt |

## Registering Tools

```typescript
import * as z from 'zod/v4'

// Basic tool
server.registerTool(
  'calculate-bmi',
  {
    title: 'BMI Calculator',
    description: 'Calculate Body Mass Index from weight and height',
    inputSchema: z.object({
      weightKg: z.number().describe('Weight in kilograms'),
      heightM: z.number().positive().describe('Height in meters'),
    }),
  },
  async ({ weightKg, heightM }) => {
    const bmi = weightKg / (heightM * heightM)
    return {
      content: [{ type: 'text', text: `BMI: ${bmi.toFixed(1)}` }],
    }
  }
)

// Tool with output schema (MCP 2025-06-18)
server.registerTool(
  'calculate-bmi',
  {
    description: 'Calculate BMI',
    inputSchema: z.object({
      weightKg: z.number(),
      heightM: z.number(),
    }),
    outputSchema: z.object({
      bmi: z.number(),
      category: z.string(),
    }),
    annotations: {
      title: 'BMI Calculator',
      readOnlyHint: true,
      idempotentHint: true,
    },
  },
  async ({ weightKg, heightM }) => {
    const bmi = weightKg / (heightM * heightM)
    const output = { bmi, category: bmi < 25 ? 'normal' : 'overweight' }
    return {
      content: [{ type: 'text', text: JSON.stringify(output) }],
      structuredContent: output,
    }
  }
)

// Tool with no parameters
server.registerTool(
  'ping',
  { description: 'Health check', inputSchema: z.object({}) },
  async () => ({ content: [{ type: 'text', text: 'pong' }] })
)
```

See [references/tools-resources-prompts.md](references/tools-resources-prompts.md) for resources, prompts, error responses, and content types.

## Tool Annotations (MCP 2025-06-18)

```typescript
annotations: {
  title: 'Human-Readable Title',       // Display name
  readOnlyHint: true,                   // No side effects
  destructiveHint: false,               // Does not destroy data
  idempotentHint: true,                 // Same input = same result
  longRunningHint: false,               // Completes quickly
}
```

Annotations help clients decide how to present and auto-approve tools. A `readOnlyHint: true` tool is safer for auto-approval than a `destructiveHint: true` one.

## Transports

```typescript
// 1. Stdio — Local process, Claude Code / CLI (most common)
import { StdioServerTransport } from '@modelcontextprotocol/node'
const transport = new StdioServerTransport()
await server.connect(transport)

// 2. Streamable HTTP — Remote server, session management
import { NodeStreamableHTTPServerTransport } from '@modelcontextprotocol/node'
import { randomUUID } from 'node:crypto'
import express from 'express'

const app = express()
app.use(express.json())

const transport = new NodeStreamableHTTPServerTransport({
  sessionIdGenerator: () => randomUUID(),
})
await server.connect(transport)

app.post('/mcp', async (req, res) => {
  await transport.handleRequest(req, res, req.body)
})
app.get('/mcp', async (req, res) => {
  await transport.handleRequest(req, res)
})
app.delete('/mcp', async (req, res) => {
  await transport.handleRequest(req, res)
})
app.listen(3000)

// 3. Stateless HTTP — JSON responses, no SSE
const transport = new NodeStreamableHTTPServerTransport({
  sessionIdGenerator: () => randomUUID(),
  enableJsonResponse: true,  // Disables SSE, returns plain JSON
})
```

See [references/transports.md](references/transports.md) for session management, SSE, reconnection, and multi-client patterns.

## Error Handling

```typescript
server.registerTool(
  'fetch-data',
  {
    description: 'Fetch data from API',
    inputSchema: z.object({ url: z.string().url() }),
  },
  async ({ url }) => {
    try {
      const res = await fetch(url)
      if (!res.ok) {
        return {
          content: [{ type: 'text', text: `HTTP ${res.status}: ${res.statusText}` }],
          isError: true,
        }
      }
      const data = await res.text()
      return { content: [{ type: 'text', text: data }] }
    } catch (err) {
      return {
        content: [{ type: 'text', text: `Error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      }
    }
  }
)
```

**Rules:**
- Return `isError: true` for application-level errors (the LLM sees the error and can retry)
- Throw exceptions only for protocol-level errors (invalid request, server bug)
- Always include a human-readable error message in `content`

## Client Configuration

### Claude Code (`~/.claude/settings.json`)
```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

## Project Structure

```
my-mcp-server/
  src/
    index.ts              # Entry: server creation + transport
    tools/
      registry.ts         # Tool registry (if many tools)
      my-tool.ts          # Individual tool definitions
      another-tool.ts
    resources/
      my-resource.ts      # Resource definitions
    prompts/
      my-prompt.ts        # Prompt definitions
    utils/
      validation.ts       # Shared Zod schemas
  bin/
    cli.js                # #!/usr/bin/env node entry
  tests/
    tools.test.ts         # Tool unit tests
  package.json
  tsconfig.json
```

See [references/advanced-patterns.md](references/advanced-patterns.md) for dynamic tool loading, middleware, tool registry, and the lazy MCP pattern.

See [references/testing-distribution.md](references/testing-distribution.md) for testing with vitest, CLI setup, npm publishing, and auto-setup scripts.

## Common Pitfalls

- **v2 requires `z.object()` wrappers.** Raw shapes like `{ name: z.string() }` no longer work — wrap with `z.object({})`.
- **Use `zod/v4` not `zod`.** The SDK v2 requires Zod v4 (`import * as z from 'zod/v4'`).
- **`"type": "module"` is required.** The SDK uses ESM. All imports need `.js` extensions in TypeScript.
- **Stdio transport = no `console.log`.** Stdout is the MCP protocol channel. Use `console.error` for debug output, or implement MCP logging.
- **`isError` is for the LLM, not the protocol.** Return `isError: true` with a helpful message so the model can understand and recover.
- **Tool names must be unique.** Registering a duplicate name will overwrite the previous tool.
- **Content array must not be empty.** Always return at least one content item, even for "no result" cases.
- **Schema descriptions matter.** LLMs use `description` fields on both tools and individual parameters to decide what to call and how.

## References

- [references/tools-resources-prompts.md](references/tools-resources-prompts.md) — Deep dive into tools, resources, resource templates, prompts, content types
- [references/transports.md](references/transports.md) — Stdio, Streamable HTTP, SSE, session management, Express integration
- [references/advanced-patterns.md](references/advanced-patterns.md) — Tool registry, dynamic loading, middleware, lazy MCP pattern, annotations
- [references/testing-distribution.md](references/testing-distribution.md) — Testing with vitest, CLI packaging, npm publishing, client auto-setup
