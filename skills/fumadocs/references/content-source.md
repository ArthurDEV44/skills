# Content Source

## defineDocs

Defines document and meta collections in `source.config.ts`:

```ts
// source.config.ts
import { defineConfig, defineDocs } from 'fumadocs-mdx/config';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    async: true, // Enable lazy loading for large doc sets
  },
});

export default defineConfig();
```

`defineDocs` creates two collections: `docs` (MDX pages) and `meta` (meta.json files).

## Custom Frontmatter Schema

Extend the default frontmatter with Zod:

```ts
// source.config.ts
import { defineConfig, defineDocs, frontmatterSchema } from 'fumadocs-mdx/config';
import { z } from 'zod';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: frontmatterSchema.extend({
      author: z.string().optional(),
      date: z.string().date().or(z.date()).optional(),
      tags: z.array(z.string()).default([]),
    }),
  },
});
```

### Default Frontmatter Fields

Every MDX page supports these fields out of the box:

```yaml
---
title: Page Title
description: Page description for metadata
icon: BookOpen         # Lucide icon name for sidebar
full: false            # Full-width page (no TOC sidebar)
---
```

## defineCollections

Create additional content collections beyond docs:

```ts
// source.config.ts
import { defineDocs, defineCollections, frontmatterSchema } from 'fumadocs-mdx/config';
import { z } from 'zod';

export const docs = defineDocs({
  dir: 'content/docs',
});

export const blog = defineCollections({
  type: 'doc',
  dir: 'content/blog',
  schema: frontmatterSchema.extend({
    author: z.string(),
    date: z.string().date().or(z.date()),
  }),
});
```

## meta.json

Controls page ordering and navigation structure within a directory:

```json
// content/docs/meta.json
{
  "title": "Documentation",
  "pages": [
    "index",
    "getting-started",
    "---Guides---",
    "guides/installation",
    "guides/configuration"
  ]
}
```

- List pages by filename (without `.mdx` extension)
- Use `---Title---` for separator headings in sidebar
- Use `...` to include remaining pages alphabetically
- Nested folders can have their own `meta.json`

### Folder as Page

```json
// content/docs/guides/meta.json
{
  "title": "Guides",
  "description": "Step-by-step guides",
  "pages": ["installation", "configuration", "..."]
}
```

## Source Loader

The loader connects content collections to your app:

```ts
// lib/source.ts
import { docs, meta } from '@/.source';
import { createMDXSource } from 'fumadocs-mdx/runtime/next';
import { loader } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/docs',
  source: createMDXSource(docs, meta),
});
```

### Loader API

```ts
// Get a page by slug
const page = source.getPage(['getting-started']);
const page = source.getPage(params.slug); // from catch-all route

// Get the page tree (for sidebar navigation)
const tree = source.getPageTree();

// Generate static params for all pages
const params = source.generateParams();

// Get all pages
const pages = source.getPages();
```

## Multiple Sources

```ts
// lib/source.ts
import { docs, meta, blog as blogPosts } from '@/.source';
import { createMDXSource } from 'fumadocs-mdx/runtime/next';
import type { InferMetaType, InferPageType } from 'fumadocs-core/source';
import { loader } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/docs',
  source: createMDXSource(docs, meta),
});

export const blog = loader({
  baseUrl: '/blog',
  source: createMDXSource(blogPosts, []),
});

export type DocsPage = InferPageType<typeof source>;
export type DocsMeta = InferMetaType<typeof source>;
```

## MDX Options

Configure remark/rehype plugins globally:

```ts
// source.config.ts
import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export const docs = defineDocs({
  dir: 'content/docs',
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
```
