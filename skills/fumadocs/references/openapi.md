# OpenAPI Integration

## Installation

```bash
npm i fumadocs-openapi
```

## Generate MDX from OpenAPI Spec

Use the CLI to generate documentation pages from an OpenAPI specification:

```bash
npx fumadocs-openapi generate ./openapi.yaml -o ./content/docs/api
```

This generates one MDX file per operation, ready to render with Fumadocs.

### Configuration File

Create `openapi.config.ts` for advanced options:

```ts
// openapi.config.ts
import { defineConfig } from 'fumadocs-openapi/config';

export default defineConfig({
  input: ['./openapi.yaml'],
  output: './content/docs/api',
  // Group by tag
  groupBy: 'tag',
  // Custom frontmatter
  frontmatter: (operation) => ({
    title: operation.summary,
    description: operation.description,
  }),
});
```

Run with:

```bash
npx fumadocs-openapi generate
```

## API Playground

Fumadocs OpenAPI provides interactive API playground components:

```tsx
// app/docs/[[...slug]]/page.tsx
import { createOpenAPI } from 'fumadocs-openapi/server';
import { source } from '@/lib/source';

const openapi = createOpenAPI({
  // path to your OpenAPI schema
  schema: './openapi.yaml',
});

// Use in your page component to render API docs with playground
```

## Route Configuration

Add API docs to your page tree:

```json
// content/docs/meta.json
{
  "pages": [
    "index",
    "getting-started",
    "---API Reference---",
    "api/..."
  ]
}
```

## Rendering API Pages

Generated MDX files include special components for:
- Request/response schemas
- Parameter tables
- Authentication requirements
- Example requests (cURL, JavaScript, Python)
- Try-it-out playground
