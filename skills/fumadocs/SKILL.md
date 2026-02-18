---
name: fumadocs
description: Fumadocs documentation framework for Next.js - project setup, content sources, MDX, layouts, UI components, search, OpenAPI, and i18n. Use when building documentation sites with Fumadocs, setting up fumadocs-core or fumadocs-ui, configuring source.config.ts, creating DocsLayout/DocsPage, adding search (Orama/Algolia), writing MDX docs, or integrating OpenAPI specs.
---

# Fumadocs

Best practices and patterns for building documentation sites with Fumadocs on Next.js.

## Project Setup

See [references/project-setup.md](./references/project-setup.md) for:
- Installation (fumadocs-core, fumadocs-ui, fumadocs-mdx)
- `source.config.ts` configuration
- `next.config.mjs` with fumadocs plugin
- App directory structure and routing
- Tailwind CSS and `fumadocs-ui/css` setup
- RootProvider and theme configuration

## Content Source

See [references/content-source.md](./references/content-source.md) for:
- `defineDocs` and `defineCollections` in `source.config.ts`
- Frontmatter schema with Zod validation
- `meta.json` for page ordering and navigation
- `loader()` and `createMDXSource()` in `lib/source.ts`
- Page tree generation and `getPageTree()`
- Multiple content collections (docs, blog, etc.)
- Lazy loading with `async: true`

## Layouts and Pages

See [references/layouts-pages.md](./references/layouts-pages.md) for:
- `DocsLayout` with sidebar, navbar, tabs
- `DocsPage`, `DocsBody`, `DocsTitle`, `DocsDescription`
- Catch-all route `[[...slug]]/page.tsx`
- `generateStaticParams` and `generateMetadata`
- Table of contents (`toc`) configuration
- Breadcrumbs and footer navigation
- Home layout vs docs layout

## UI Components

See [references/ui-components.md](./references/ui-components.md) for:
- MDX component overrides (`getMDXComponents`)
- `Callout` (info, warn, error)
- `Tabs` / `Tab` for tabbed content
- `Card` / `Cards` for link cards
- `Accordion` / `Accordions`
- `Steps` / `Step` for sequential guides
- `CodeBlock` / `Pre` with syntax highlighting
- `ImageZoom`, `TypeTable`, `Files`
- `Banner` for announcements

## Search

See [references/search.md](./references/search.md) for:
- Built-in Orama search (default)
- `createFromSource` server API
- Search API route handler
- Algolia integration
- `SearchDialog` customization
- i18n locale-aware search

## OpenAPI Integration

See [references/openapi.md](./references/openapi.md) for:
- `fumadocs-openapi` package
- Generating MDX from OpenAPI specs
- API playground components
- Route configuration for API docs

## Internationalization

See [references/i18n.md](./references/i18n.md) for:
- Folder-based i18n (`[lang]` segment)
- `i18n.ts` configuration
- Middleware for locale detection
- Localized content directories
- `source.getPageTree()` with locale
- Search with locale maps
