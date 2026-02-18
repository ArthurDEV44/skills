# Testing & Distribution

## Testing MCP Tools

### Unit Testing with Vitest

Test tool handlers directly — they are plain async functions:

```typescript
// src/tools/my-tool.test.ts
import { describe, it, expect } from 'vitest'
import { myTool } from './my-tool.js'

describe('my-tool', () => {
  it('should return expected result', async () => {
    const result = await myTool.execute({ input: 'test' })

    expect(result.isError).toBeUndefined()
    expect(result.content).toHaveLength(1)
    expect(result.content[0]!.type).toBe('text')
    expect(result.content[0]!.text).toContain('expected output')
  })

  it('should handle errors gracefully', async () => {
    const result = await myTool.execute({ input: '' })

    expect(result.isError).toBe(true)
    expect(result.content[0]!.text).toContain('Error')
  })

  it('should have valid tool definition', () => {
    expect(myTool.name).toBe('my-tool')
    expect(myTool.description).toBeTruthy()
    expect(myTool.inputSchema).toBeDefined()
  })

  it('should return structured content when outputSchema defined', async () => {
    const result = await myTool.execute({ input: 'test' })

    if (myTool.outputSchema) {
      expect(result.structuredContent).toBeDefined()
      // Validate against schema
      const parsed = myTool.outputSchema.parse(result.structuredContent)
      expect(parsed).toEqual(result.structuredContent)
    }
  })
})
```

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 30_000,         // WASM modules can take time to init
    include: ['src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/*.d.ts'],
    },
  },
})
```

### Testing the Full Server

```typescript
// tests/server.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { createServer } from '../src/server.js'

describe('MCP Server Integration', () => {
  let client: Client
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    const server = await createServer({ mode: 'all' })
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair()

    client = new Client({ name: 'test-client', version: '1.0.0' })

    await server.connect(serverTransport)
    await client.connect(clientTransport)

    cleanup = async () => {
      await client.close()
      await server.close()
    }
  })

  afterAll(async () => {
    await cleanup()
  })

  it('should list all tools', async () => {
    const { tools } = await client.listTools()
    expect(tools.length).toBeGreaterThan(0)
    expect(tools[0]).toHaveProperty('name')
    expect(tools[0]).toHaveProperty('description')
    expect(tools[0]).toHaveProperty('inputSchema')
  })

  it('should call a tool successfully', async () => {
    const result = await client.callTool({
      name: 'greet',
      arguments: { name: 'World' },
    })

    expect(result.isError).toBeFalsy()
    expect(result.content[0]).toEqual({
      type: 'text',
      text: 'Hello, World!',
    })
  })

  it('should return error for unknown tool', async () => {
    const result = await client.callTool({
      name: 'nonexistent',
      arguments: {},
    })

    expect(result.isError).toBe(true)
  })
})
```

### Testing Resources and Prompts

```typescript
it('should list resources', async () => {
  const { resources } = await client.listResources()
  expect(resources.length).toBeGreaterThan(0)
})

it('should read a resource', async () => {
  const result = await client.readResource({
    uri: 'config://app/settings',
  })
  expect(result.contents[0]!.text).toBeTruthy()
})

it('should list prompts', async () => {
  const { prompts } = await client.listPrompts()
  expect(prompts.length).toBeGreaterThan(0)
})

it('should get a prompt', async () => {
  const result = await client.getPrompt({
    name: 'code-review',
    arguments: { code: 'const x = 1', language: 'typescript' },
  })
  expect(result.messages.length).toBeGreaterThan(0)
})
```

## CLI Entry Point

### bin/cli.js

```javascript
#!/usr/bin/env node

import { parseArgs } from 'node:util'

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    verbose: { type: 'boolean', default: false },
    mode: { type: 'string', default: 'core' },
    version: { type: 'boolean', short: 'v', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  allowPositionals: true,
  strict: false,
})

const command = positionals[0] ?? 'serve'

if (values.version) {
  const pkg = await import('../package.json', { with: { type: 'json' } })
  console.log(pkg.default.version)
  process.exit(0)
}

if (values.help) {
  console.log(`
Usage: my-mcp-server [command] [options]

Commands:
  serve     Start MCP server (default)
  setup     Configure for Claude Code / Cursor / Windsurf
  doctor    Check configuration and dependencies

Options:
  --mode <lazy|core|all>   Tool loading mode (default: core)
  --verbose                Enable debug logging
  -v, --version            Show version
  -h, --help               Show help
`)
  process.exit(0)
}

switch (command) {
  case 'serve': {
    const { runServer } = await import('../dist/server.js')
    await runServer({ verbose: values.verbose, mode: values.mode })
    break
  }
  case 'setup': {
    const { runSetup } = await import('../dist/cli/setup.js')
    await runSetup()
    break
  }
  case 'doctor': {
    const { runDoctor } = await import('../dist/cli/doctor.js')
    await runDoctor()
    break
  }
  default:
    console.error(`Unknown command: ${command}`)
    process.exit(1)
}
```

## Auto-Setup Script

Help users configure the MCP server for their client:

```typescript
// src/cli/setup.ts
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

interface ClientConfig {
  name: string
  configPath: string
  key: string
}

const CLIENTS: ClientConfig[] = [
  {
    name: 'Claude Code',
    configPath: join(homedir(), '.claude', 'settings.json'),
    key: 'mcpServers',
  },
  {
    name: 'Claude Desktop',
    configPath: join(homedir(), 'Library', 'Application Support', 'Claude', 'claude_desktop_config.json'),
    key: 'mcpServers',
  },
  {
    name: 'Cursor',
    configPath: join(process.cwd(), '.cursor', 'mcp.json'),
    key: 'mcpServers',
  },
]

export async function runSetup() {
  const serverEntry = {
    command: 'npx',
    args: ['-y', 'my-mcp-server'],
    env: {},
  }

  for (const client of CLIENTS) {
    if (!existsSync(client.configPath)) continue

    console.log(`Configuring ${client.name}...`)
    const config = JSON.parse(readFileSync(client.configPath, 'utf-8'))
    config[client.key] = config[client.key] ?? {}
    config[client.key]['my-server'] = serverEntry
    writeFileSync(client.configPath, JSON.stringify(config, null, 2))
    console.log(`  Updated ${client.configPath}`)
  }
}
```

## package.json for Publishing

```json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "description": "MCP server for ...",
  "type": "module",
  "bin": {
    "my-mcp-server": "./bin/cli.js"
  },
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "files": [
    "dist/",
    "bin/"
  ],
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "test:watch": "vitest",
    "start": "node bin/cli.js serve",
    "prepublishOnly": "bun run build && bun run test"
  },
  "keywords": ["mcp", "model-context-protocol", "ai", "tools"],
  "dependencies": {
    "@modelcontextprotocol/sdk": "^2.0.0",
    "zod": "^4.0.0"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "vitest": "^4.0.0",
    "@vitest/coverage-v8": "^4.0.0"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
```

## tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*.ts"],
  "exclude": ["src/**/*.test.ts", "node_modules", "dist"]
}
```

## Publishing Checklist

1. **Build:** `bun run build` — compiles TypeScript to `dist/`
2. **Test:** `bun run test` — all tests pass
3. **Bin entry:** `bin/cli.js` has `#!/usr/bin/env node` shebang
4. **Files field:** Only `dist/` and `bin/` are included
5. **Type module:** `"type": "module"` in package.json
6. **README:** Document required environment variables
7. **Publish:** `npm publish` (or `bun publish`)

## Debugging

### Inspector

Use the MCP Inspector for interactive testing:

```bash
npx @modelcontextprotocol/inspector node dist/index.js
```

Opens a web UI where you can:
- List tools, resources, prompts
- Call tools with arguments
- Read resources
- Execute prompts
- See raw JSON-RPC messages

### Stdio Debugging

```bash
# See raw protocol messages
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | node dist/index.js 2>/dev/null

# Pipe stderr to a log file for debugging
node dist/index.js 2>debug.log
```
