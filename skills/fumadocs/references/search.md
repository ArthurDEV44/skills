# Search

## Built-in Orama Search (Default)

Fumadocs includes Orama as the default search engine. It runs entirely client-side with a server-generated index.

### Search API Route

```ts
// app/api/search/route.ts
import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

export const { GET } = createFromSource(source);
```

This generates a search index from all pages and serves it via a GET endpoint. The `SearchDialog` component in `RootProvider` uses this endpoint automatically.

### Custom Search Options

```ts
// app/api/search/route.ts
import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

export const { GET } = createFromSource(source, {
  // Index specific fields
  indexes: source.getPages().map((page) => ({
    title: page.data.title,
    description: page.data.description,
    url: page.url,
    id: page.url,
    structuredData: page.data.structuredData,
  })),
});
```

### Static Search Index

For static exports, generate the index at build time:

```ts
// app/api/search/route.ts
import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

export const { staticGET: GET } = createFromSource(source);
```

## Algolia Search

### Installation

```bash
npm i fumadocs-core algoliasearch
```

### Sync Index Script

```ts
// scripts/sync-algolia.mts
import algolia from 'algoliasearch';
import { sync } from 'fumadocs-core/search/algolia';
import { source } from '@/lib/source';

const client = algolia('APP_ID', 'ADMIN_API_KEY');

sync(client, {
  document: source.getPages().map((page) => ({
    _id: page.url,
    title: page.data.title,
    description: page.data.description,
    url: page.url,
    structured: page.data.structuredData,
  })),
});
```

### Algolia Search Dialog

```tsx
// app/layout.tsx
import { RootProvider } from 'fumadocs-ui/provider';

<RootProvider
  search={{
    type: 'algolia',
    appId: 'APP_ID',
    apiKey: 'SEARCH_API_KEY',
    indexName: 'docs',
  }}
>
```

## Custom Search Dialog

```tsx
import { SearchDialog } from 'fumadocs-ui/components/dialog/search';

// Override in RootProvider:
<RootProvider
  search={{
    SearchDialog: MyCustomSearchDialog,
  }}
>
```

## Search with i18n

```ts
// app/api/search/route.ts
import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

const server = createFromSource(source, {
  localeMap: {
    ru: { language: 'russian' },
    en: { language: 'english' },
    zh: { language: 'chinese' },
  },
});

export const { GET } = server;
```

## Keyboard Shortcut

Search dialog opens with `Ctrl+K` / `Cmd+K` by default. This is configured automatically by `RootProvider`.
