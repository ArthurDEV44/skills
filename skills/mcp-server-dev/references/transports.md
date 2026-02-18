# Transports

MCP supports multiple transport mechanisms. Choose based on your deployment model.

## Transport Comparison

| Transport | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Stdio** | Local CLI tools, IDE plugins | Simple, secure, no network | Single client, local only |
| **Streamable HTTP** | Remote servers, multi-client | Sessions, SSE streaming, scalable | Requires HTTP server setup |
| **SSE (Legacy)** | Backward compatibility | Wide support | Deprecated in favor of Streamable HTTP |

## Stdio Transport

The most common transport for local MCP servers. Communication via stdin/stdout.

```typescript
import { McpServer } from '@modelcontextprotocol/server'
import { StdioServerTransport } from '@modelcontextprotocol/node'

const server = new McpServer({ name: 'my-server', version: '1.0.0' })

// Register tools, resources, prompts...

const transport = new StdioServerTransport()
await server.connect(transport)
```

**Important:** stdout is the protocol channel. Never use `console.log()` — it will corrupt the protocol stream. Use `console.error()` for debug output.

```typescript
// Debug logging in stdio mode
console.error('[DEBUG] Tool called:', toolName)
console.error('[DEBUG] Args:', JSON.stringify(args))
```

### MCP Protocol Logging (Better Alternative)

```typescript
// Use MCP's built-in logging instead of console.error
server.server.sendLoggingMessage({
  level: 'info',
  logger: 'my-server',
  data: { message: 'Tool executed', tool: 'greet', duration: 42 },
})
```

## Streamable HTTP Transport

For remote servers accessible over HTTP. Supports sessions, SSE streaming, and multiple concurrent clients.

### Stateful (with Sessions)

```typescript
import { McpServer } from '@modelcontextprotocol/server'
import { NodeStreamableHTTPServerTransport } from '@modelcontextprotocol/node'
import { randomUUID } from 'node:crypto'
import express from 'express'

const app = express()
app.use(express.json())

const server = new McpServer({ name: 'my-server', version: '1.0.0' })

// Register tools...

// Stateful transport — maintains session across requests
const transport = new NodeStreamableHTTPServerTransport({
  sessionIdGenerator: () => randomUUID(),
})

await server.connect(transport)

// Handle all MCP HTTP methods
app.post('/mcp', async (req, res) => {
  await transport.handleRequest(req, res, req.body)
})

// GET for SSE stream (client listens for server-initiated messages)
app.get('/mcp', async (req, res) => {
  await transport.handleRequest(req, res)
})

// DELETE for session cleanup
app.delete('/mcp', async (req, res) => {
  await transport.handleRequest(req, res)
})

app.listen(3000, () => {
  console.error('MCP server running on http://localhost:3000/mcp')
})
```

### Stateless (JSON Response Mode)

Disables SSE streaming. Each request/response is a complete JSON exchange.

```typescript
const transport = new NodeStreamableHTTPServerTransport({
  sessionIdGenerator: () => randomUUID(),
  enableJsonResponse: true,   // No SSE, plain JSON responses
})

// GET requests return 405 in JSON mode (no SSE stream to establish)
app.post('/mcp', async (req, res) => {
  await transport.handleRequest(req, res, req.body)
})
```

### Multi-Client Server

Handle multiple clients, each with their own session and server instance:

```typescript
import express from 'express'
import { McpServer } from '@modelcontextprotocol/server'
import { NodeStreamableHTTPServerTransport } from '@modelcontextprotocol/node'
import { randomUUID } from 'node:crypto'

const app = express()
app.use(express.json())

// Track active sessions
const sessions = new Map<string, {
  server: McpServer
  transport: NodeStreamableHTTPServerTransport
}>()

function createSession() {
  const server = new McpServer({ name: 'my-server', version: '1.0.0' })

  // Register tools for this session's server...
  registerAllTools(server)

  const transport = new NodeStreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  })

  return { server, transport }
}

app.post('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] as string | undefined

  if (sessionId && sessions.has(sessionId)) {
    // Existing session
    const { transport } = sessions.get(sessionId)!
    await transport.handleRequest(req, res, req.body)
  } else {
    // New session
    const session = createSession()
    await session.server.connect(session.transport)

    // Store session after connect (transport generates the ID)
    session.transport.on('session-created', (id) => {
      sessions.set(id, session)
    })

    await session.transport.handleRequest(req, res, req.body)
  }
})

app.delete('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] as string
  const session = sessions.get(sessionId)
  if (session) {
    await session.transport.handleRequest(req, res)
    sessions.delete(sessionId)
  } else {
    res.status(404).end()
  }
})

app.listen(3000)
```

## SSE Transport (Legacy)

Deprecated but still used by some clients. Uses separate endpoints for SSE stream and messages.

```typescript
import { SSEServerTransport } from '@modelcontextprotocol/node'

let transport: SSEServerTransport | null = null

// SSE endpoint — client connects here for server-to-client messages
app.get('/sse', async (req, res) => {
  transport = new SSEServerTransport('/messages', res)
  await server.connect(transport)
})

// Message endpoint — client posts JSON-RPC messages here
app.post('/messages', async (req, res) => {
  if (!transport) return res.status(503).end()
  await transport.handlePostMessage(req, res, req.body)
})
```

## Remote Server Client Configuration

For HTTP-based servers, client config uses `url` instead of `command`:

```json
{
  "mcpServers": {
    "my-remote-server": {
      "url": "http://localhost:3000/mcp",
      "headers": {
        "Authorization": "Bearer my-api-key"
      }
    }
  }
}
```

## Choosing a Transport

- **Building a CLI tool / IDE plugin?** Use **Stdio**. It's the simplest and most secure.
- **Building a remote API / SaaS?** Use **Streamable HTTP** with sessions.
- **Need request/response without streaming?** Use **Streamable HTTP with `enableJsonResponse: true`**.
- **Supporting legacy clients?** Add SSE transport alongside Streamable HTTP.
