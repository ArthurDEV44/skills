# Tools, Resources & Prompts

## Tools — Deep Dive

### Tool Registration API (v2)

```typescript
server.registerTool(
  name: string,
  config: {
    title?: string,               // Human-readable display name
    description: string,          // What the tool does (LLM reads this)
    inputSchema: z.ZodObject,     // Zod v4 schema for input validation
    outputSchema?: z.ZodObject,   // Structured output schema (MCP 2025-06-18)
    annotations?: ToolAnnotations,
    _meta?: Record<string, unknown>,
  },
  handler: (args: Input, extra: RequestHandlerExtra) => Promise<CallToolResult>
)
```

### CallToolResult Structure

```typescript
interface CallToolResult {
  content: Array<TextContent | ImageContent | EmbeddedResource>
  isError?: boolean                       // true = tool error (LLM sees it)
  structuredContent?: Record<string, any> // Matches outputSchema if provided
  _meta?: Record<string, unknown>
}

// Content types
interface TextContent {
  type: 'text'
  text: string
}

interface ImageContent {
  type: 'image'
  data: string        // Base64-encoded
  mimeType: string    // 'image/png', 'image/jpeg', etc.
}

interface EmbeddedResource {
  type: 'resource'
  resource: {
    uri: string
    mimeType?: string
    text?: string       // Text resource content
    blob?: string       // Binary resource content (base64)
  }
}
```

### Returning Multiple Content Items

```typescript
server.registerTool(
  'analyze-image',
  {
    description: 'Analyze an image and return results',
    inputSchema: z.object({ imagePath: z.string() }),
  },
  async ({ imagePath }) => {
    const analysis = await analyzeImage(imagePath)
    const thumbnail = await generateThumbnail(imagePath)

    return {
      content: [
        { type: 'text', text: `Analysis: ${analysis.description}` },
        { type: 'image', data: thumbnail.base64, mimeType: 'image/png' },
        { type: 'text', text: `Tags: ${analysis.tags.join(', ')}` },
      ],
    }
  }
)
```

### Tool with Output Schema

When `outputSchema` is provided, clients can parse `structuredContent` for programmatic use:

```typescript
server.registerTool(
  'get-weather',
  {
    description: 'Get current weather for a city',
    inputSchema: z.object({
      city: z.string(),
      units: z.enum(['celsius', 'fahrenheit']).default('celsius'),
    }),
    outputSchema: z.object({
      temperature: z.number(),
      humidity: z.number(),
      description: z.string(),
      windSpeed: z.number(),
    }),
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  async ({ city, units }) => {
    const weather = await fetchWeather(city, units)
    const output = {
      temperature: weather.temp,
      humidity: weather.humidity,
      description: weather.description,
      windSpeed: weather.wind,
    }
    return {
      content: [{ type: 'text', text: JSON.stringify(output, null, 2) }],
      structuredContent: output,
    }
  }
)
```

### Input Validation Best Practices

```typescript
// Use descriptive Zod schemas — descriptions become part of the JSON Schema
const inputSchema = z.object({
  query: z.string()
    .min(1)
    .max(500)
    .describe('Search query string'),

  limit: z.number()
    .int()
    .min(1)
    .max(100)
    .default(10)
    .describe('Maximum number of results to return'),

  filters: z.object({
    category: z.enum(['docs', 'code', 'issues']).optional()
      .describe('Filter results by category'),
    dateAfter: z.string().datetime().optional()
      .describe('Only include results after this ISO date'),
  }).optional()
    .describe('Optional filters to narrow results'),
})
```

### Error Responses

```typescript
// Application error — LLM sees the error, can retry or adjust
return {
  content: [{ type: 'text', text: 'Rate limited. Try again in 30 seconds.' }],
  isError: true,
}

// Structured error with details
return {
  content: [{
    type: 'text',
    text: JSON.stringify({
      error: 'VALIDATION_FAILED',
      message: 'Email format is invalid',
      field: 'email',
    }),
  }],
  isError: true,
}

// Protocol error — throw (client handles, LLM does NOT see details)
throw new Error('Internal server error: database connection failed')
```

---

## Resources

Resources expose read-only data that clients can browse and attach to LLM context.

### Static Resources

```typescript
server.resource(
  'config',                              // Internal name
  'config://app/settings',               // URI (unique identifier)
  {
    title: 'Application Settings',
    description: 'Current application configuration',
    mimeType: 'application/json',
  },
  async () => ({
    contents: [{
      uri: 'config://app/settings',
      mimeType: 'application/json',
      text: JSON.stringify(getAppConfig(), null, 2),
    }],
  })
)
```

### Resource Templates (Dynamic URIs)

```typescript
server.resource(
  'user-profile',
  'users://{userId}/profile',            // URI template with parameter
  {
    title: 'User Profile',
    description: 'Get profile for a specific user',
    mimeType: 'application/json',
  },
  async (uri, { userId }) => {           // Parameters extracted from URI
    const user = await getUser(userId)
    return {
      contents: [{
        uri: uri.href,
        mimeType: 'application/json',
        text: JSON.stringify(user, null, 2),
      }],
    }
  }
)
```

### File Resources

```typescript
import { readFile } from 'node:fs/promises'

server.resource(
  'readme',
  'file:///project/README.md',
  { title: 'Project README', mimeType: 'text/markdown' },
  async () => ({
    contents: [{
      uri: 'file:///project/README.md',
      mimeType: 'text/markdown',
      text: await readFile('./README.md', 'utf-8'),
    }],
  })
)
```

### Binary Resources

```typescript
server.resource(
  'logo',
  'assets://logo.png',
  { title: 'Company Logo', mimeType: 'image/png' },
  async () => ({
    contents: [{
      uri: 'assets://logo.png',
      mimeType: 'image/png',
      blob: (await readFile('./assets/logo.png')).toString('base64'),
    }],
  })
)
```

### Resource List Change Notifications

When the set of available resources changes dynamically:

```typescript
// After adding/removing resources, notify clients
server.sendResourceListChanged()
```

---

## Prompts

Prompts are reusable LLM prompt templates with user-provided arguments.

### Basic Prompt

```typescript
server.prompt(
  'code-review',                          // Prompt name
  {
    title: 'Code Review',
    description: 'Review code for bugs, style, and best practices',
    argsSchema: z.object({
      code: z.string().describe('Code to review'),
      language: z.string().default('typescript').describe('Programming language'),
      focus: z.enum(['bugs', 'style', 'performance', 'all']).default('all')
        .describe('Review focus area'),
    }),
  },
  async ({ code, language, focus }) => ({
    messages: [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Review this ${language} code for ${focus}:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nProvide specific, actionable feedback.`,
        },
      },
    ],
  })
)
```

### Multi-Message Prompt with System Instructions

```typescript
server.prompt(
  'debug-error',
  {
    description: 'Systematic debugging of an error',
    argsSchema: z.object({
      error: z.string().describe('Error message or stack trace'),
      context: z.string().optional().describe('Additional context'),
    }),
  },
  async ({ error, context }) => ({
    messages: [
      {
        role: 'assistant',
        content: {
          type: 'text',
          text: 'I am a senior debugging assistant. I will analyze errors systematically.',
        },
      },
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Error:\n${error}${context ? `\n\nContext:\n${context}` : ''}\n\nAnalyze the root cause and suggest a fix.`,
        },
      },
    ],
  })
)
```

### Prompt with Embedded Resources

```typescript
server.prompt(
  'summarize-file',
  {
    description: 'Summarize a project file',
    argsSchema: z.object({
      filePath: z.string().describe('Path to the file'),
    }),
  },
  async ({ filePath }) => ({
    messages: [
      {
        role: 'user',
        content: {
          type: 'resource',
          resource: {
            uri: `file:///${filePath}`,
            text: await readFile(filePath, 'utf-8'),
            mimeType: 'text/plain',
          },
        },
      },
      {
        role: 'user',
        content: {
          type: 'text',
          text: 'Summarize this file in 3-5 bullet points.',
        },
      },
    ],
  })
)
```
