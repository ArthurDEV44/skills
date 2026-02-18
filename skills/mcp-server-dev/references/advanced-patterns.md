# Advanced Patterns

## Tool Registry

For servers with many tools, extract registration into a registry pattern:

```typescript
// src/tools/registry.ts
export interface ToolDefinition {
  name: string
  description: string
  inputSchema: Record<string, unknown>   // JSON Schema (from Zod)
  outputSchema?: Record<string, unknown>
  annotations?: {
    title?: string
    readOnlyHint?: boolean
    destructiveHint?: boolean
    idempotentHint?: boolean
    longRunningHint?: boolean
  }
  execute: (args: unknown) => Promise<{
    content: Array<{ type: 'text'; text: string }>
    isError?: boolean
    structuredContent?: Record<string, unknown>
  }>
}

export class ToolRegistry {
  private tools = new Map<string, ToolDefinition>()
  private changeCallbacks: Array<() => void> = []

  register(tool: ToolDefinition): this {
    this.tools.set(tool.name, tool)
    this.notifyChange()
    return this
  }

  unregister(name: string): boolean {
    const deleted = this.tools.delete(name)
    if (deleted) this.notifyChange()
    return deleted
  }

  get(name: string): ToolDefinition | undefined {
    return this.tools.get(name)
  }

  list(): ToolDefinition[] {
    return Array.from(this.tools.values())
  }

  async execute(name: string, args: unknown) {
    const tool = this.tools.get(name)
    if (!tool) {
      return {
        content: [{ type: 'text' as const, text: `Unknown tool: ${name}` }],
        isError: true,
      }
    }
    return tool.execute(args)
  }

  onChange(callback: () => void): void {
    this.changeCallbacks.push(callback)
  }

  private notifyChange(): void {
    this.changeCallbacks.forEach((cb) => cb())
  }
}
```

### Using with Low-Level Server

For fine-grained control, use the low-level `Server` API instead of `McpServer`:

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'

const server = new Server(
  { name: 'my-server', version: '1.0.0' },
  { capabilities: { tools: {} } }
)

const registry = new ToolRegistry()

// Register handlers
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: registry.list().map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
    outputSchema: t.outputSchema,
    annotations: t.annotations,
  })),
}))

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params
  return registry.execute(name, args)
})

// Notify clients when tool list changes
registry.onChange(() => {
  server.notification({ method: 'notifications/tools/list_changed' })
})
```

## Dynamic Tool Loading

Load tools on-demand to minimize initial token overhead:

```typescript
// src/tools/catalog.ts
export type ToolCategory = 'search' | 'analyze' | 'transform' | 'generate'

export interface ToolMetadata {
  name: string
  category: ToolCategory
  keywords: string[]
  description: string
  loader: () => Promise<ToolDefinition>   // Lazy import
}

export const TOOL_CATALOG: ToolMetadata[] = [
  {
    name: 'search-docs',
    category: 'search',
    keywords: ['search', 'find', 'query', 'documentation'],
    description: 'Search project documentation',
    loader: () => import('./search-docs.js').then((m) => m.searchDocsTool),
  },
  {
    name: 'analyze-deps',
    category: 'analyze',
    keywords: ['dependencies', 'packages', 'imports'],
    description: 'Analyze project dependencies',
    loader: () => import('./analyze-deps.js').then((m) => m.analyzeDeps),
  },
  // ...
]
```

### Dynamic Loader Class

```typescript
export class DynamicToolLoader {
  private loaded = new Map<string, ToolDefinition>()

  constructor(private catalog: ToolMetadata[]) {}

  getAvailable(): Array<{ name: string; category: string; description: string }> {
    return this.catalog.map(({ name, category, description }) => ({
      name, category, description,
    }))
  }

  search(query: string): ToolMetadata[] {
    const q = query.toLowerCase()
    return this.catalog.filter(
      (t) =>
        t.name.includes(q) ||
        t.description.toLowerCase().includes(q) ||
        t.keywords.some((k) => k.includes(q))
    )
  }

  async loadByNames(names: string[]): Promise<ToolDefinition[]> {
    const results: ToolDefinition[] = []
    for (const name of names) {
      if (this.loaded.has(name)) {
        results.push(this.loaded.get(name)!)
        continue
      }
      const meta = this.catalog.find((t) => t.name === name)
      if (!meta) continue
      const tool = await meta.loader()
      this.loaded.set(name, tool)
      results.push(tool)
    }
    return results
  }

  async loadByCategory(category: ToolCategory): Promise<ToolDefinition[]> {
    const names = this.catalog
      .filter((t) => t.category === category)
      .map((t) => t.name)
    return this.loadByNames(names)
  }

  isLoaded(name: string): boolean {
    return this.loaded.has(name)
  }
}
```

### Discovery Tool

Expose a meta-tool that lets the LLM discover and load tools:

```typescript
server.registerTool(
  'discover_tools',
  {
    description: 'Find and load available tools by query or category',
    inputSchema: z.object({
      query: z.string().optional().describe('Search by name or keyword'),
      category: z.enum(['search', 'analyze', 'transform', 'generate']).optional(),
      load: z.boolean().default(false).describe('Load matched tools into the session'),
    }),
    annotations: { readOnlyHint: true },
  },
  async ({ query, category, load }) => {
    let matches = loader.getAvailable()

    if (query) matches = loader.search(query).map(({ name, category, description }) => ({ name, category, description }))
    if (category) matches = matches.filter((t) => t.category === category)

    if (load) {
      const names = matches.map((m) => m.name)
      const tools = await loader.loadByNames(names)
      tools.forEach((t) => registry.register(t))
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ tools: matches, loaded: load }, null, 2),
      }],
    }
  }
)
```

## Lazy MCP Pattern

Minimize token usage by exposing only 2 meta-tools at startup:

```typescript
// browse_tools — List categories and tools
server.registerTool(
  'browse_tools',
  {
    description: 'List tool categories or tools within a category',
    inputSchema: z.object({
      category: z.enum(['search', 'analyze', 'transform', 'generate']).optional()
        .describe('Category to browse. Omit to see all categories.'),
    }),
  },
  async ({ category }) => {
    if (!category) {
      const categories = [...new Set(TOOL_CATALOG.map((t) => t.category))]
      return {
        content: [{ type: 'text', text: `Available categories: ${categories.join(', ')}` }],
      }
    }
    const tools = TOOL_CATALOG.filter((t) => t.category === category)
    const list = tools.map((t) => `- ${t.name}: ${t.description}`).join('\n')
    return { content: [{ type: 'text', text: list }] }
  }
)

// run_tool — Execute any tool by name (loads on first use)
server.registerTool(
  'run_tool',
  {
    description: 'Execute a tool by name. Use browse_tools to find tools first.',
    inputSchema: z.object({
      name: z.string().describe('Tool name to execute'),
      args: z.record(z.unknown()).default({}).describe('Tool arguments'),
    }),
  },
  async ({ name, args }) => {
    if (!registry.get(name)) {
      const tools = await loader.loadByNames([name])
      if (tools.length === 0) {
        return {
          content: [{ type: 'text', text: `Tool "${name}" not found. Use browse_tools to see available tools.` }],
          isError: true,
        }
      }
      tools.forEach((t) => registry.register(t))
    }
    return registry.execute(name, args)
  }
)
```

**Token savings:** A server with 20 tools costs ~1,100 tokens at startup. Lazy MCP reduces this to ~150 tokens (2 meta-tools), loading additional tools on-demand.

## Middleware Pattern

Wrap tool execution with before/after hooks:

```typescript
export interface Middleware {
  name: string
  priority: number   // Lower runs first

  beforeTool?(ctx: ToolContext): Promise<ToolContext | null>
  afterTool?(ctx: ToolContext, result: ToolResult): Promise<ToolResult>
  onError?(ctx: ToolContext, error: Error): Promise<ToolResult | null>
}

export interface ToolContext {
  toolName: string
  arguments: Record<string, unknown>
  startTime: number
  metadata: Record<string, unknown>
}

export class MiddlewareChain {
  private middlewares: Middleware[] = []

  use(mw: Middleware): this {
    this.middlewares.push(mw)
    this.middlewares.sort((a, b) => a.priority - b.priority)
    return this
  }

  async executeBefore(ctx: ToolContext): Promise<ToolContext | null> {
    let current: ToolContext | null = ctx
    for (const mw of this.middlewares) {
      if (!current || !mw.beforeTool) continue
      current = await mw.beforeTool(current)
    }
    return current
  }

  async executeAfter(ctx: ToolContext, result: ToolResult): Promise<ToolResult> {
    let current = result
    for (const mw of this.middlewares) {
      if (!mw.afterTool) continue
      current = await mw.afterTool(ctx, current)
    }
    return current
  }
}
```

### Example: Logging Middleware

```typescript
const loggingMiddleware: Middleware = {
  name: 'logging',
  priority: 0,

  async beforeTool(ctx) {
    console.error(`[${new Date().toISOString()}] Tool called: ${ctx.toolName}`)
    return ctx
  },

  async afterTool(ctx, result) {
    const duration = Date.now() - ctx.startTime
    console.error(`[${new Date().toISOString()}] ${ctx.toolName} completed in ${duration}ms`)
    return result
  },

  async onError(ctx, error) {
    console.error(`[${new Date().toISOString()}] ${ctx.toolName} failed:`, error.message)
    return null  // Let error propagate
  },
}
```

### Example: Token Counting Middleware

```typescript
import { encodingForModel } from 'js-tiktoken'

const tokenMiddleware: Middleware = {
  name: 'token-counter',
  priority: 10,

  async afterTool(ctx, result) {
    const enc = encodingForModel('gpt-4o')
    const inputTokens = enc.encode(JSON.stringify(ctx.arguments)).length
    const outputTokens = result.content.reduce(
      (sum, c) => sum + (c.type === 'text' ? enc.encode(c.text).length : 0),
      0
    )
    console.error(`Tokens — in: ${inputTokens}, out: ${outputTokens}`)
    return result
  },
}
```

## Loading Modes

Offer configurable loading strategies for different client capabilities:

```typescript
export type LoadingMode = 'lazy' | 'core' | 'all'

export async function createServer(mode: LoadingMode = 'core') {
  const server = new McpServer({ name: 'my-server', version: '1.0.0' })

  switch (mode) {
    case 'lazy':
      // Only meta-tools (browse_tools + run_tool) — ~150 tokens
      registerLazyTools(server)
      break

    case 'core':
      // Essential tools + discover_tools — ~250 tokens
      registerCoreTools(server)
      registerDiscoverTool(server)
      break

    case 'all':
      // All tools registered at startup — ~1,100+ tokens
      registerAllTools(server)
      break
  }

  return server
}
```

## Environment Variables

Access environment variables passed by the client:

```typescript
// Client config passes env vars:
// { "env": { "API_KEY": "sk-...", "BASE_URL": "https://api.example.com" } }

// Server reads them normally:
const apiKey = process.env.API_KEY
const baseUrl = process.env.BASE_URL
```

**Security:** Never hardcode secrets in the server. Always read from `process.env` and document required variables.
