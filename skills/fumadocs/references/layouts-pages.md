# Layouts and Pages

## DocsLayout

The main layout wrapping all documentation pages. Provides sidebar, navbar, and search.

```tsx
// app/docs/layout.tsx
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { source } from '@/lib/source';
import type { ReactNode } from 'react';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={source.getPageTree()}
      nav={{
        title: 'My Docs',
        url: '/docs',
      }}
      sidebar={{
        defaultOpenLevel: 1,
        collapsible: true,
      }}
    >
      {children}
    </DocsLayout>
  );
}
```

### DocsLayout Props

| Prop | Type | Description |
|------|------|-------------|
| `tree` | `PageTree` | Page tree from `source.getPageTree()` |
| `nav` | `object` | Navbar config: `title`, `url`, `links`, `githubUrl` |
| `sidebar` | `object` | Sidebar config: `defaultOpenLevel`, `collapsible`, `banner`, `footer` |
| `tabs` | `TabOptions[]` | Tab navigation between doc sections |
| `i18n` | `boolean` | Enable i18n language switcher |

### Navbar Links

```tsx
<DocsLayout
  tree={source.getPageTree()}
  nav={{
    title: 'My Docs',
    url: '/docs',
    links: [
      { text: 'Blog', url: '/blog' },
      { text: 'GitHub', url: 'https://github.com/org/repo', external: true },
    ],
    githubUrl: 'https://github.com/org/repo',
  }}
>
```

### Sidebar Tabs

Split docs into multiple tabbed sections:

```tsx
<DocsLayout
  tree={source.getPageTree()}
  tabs={[
    { title: 'Guides', url: '/docs/guides' },
    { title: 'API', url: '/docs/api' },
    { title: 'Examples', url: '/docs/examples' },
  ]}
>
```

## DocsPage

Renders individual documentation pages with TOC and navigation.

```tsx
// app/docs/[[...slug]]/page.tsx
import {
  DocsPage,
  DocsBody,
  DocsTitle,
  DocsDescription,
} from 'fumadocs-ui/layouts/docs/page';
import { source } from '@/lib/source';
import { notFound } from 'next/navigation';

export default async function Page(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const { body: MDXContent, toc } = await page.data.load();

  return (
    <DocsPage
      toc={toc}
      tableOfContent={{ style: 'clerk' }}
      breadcrumb={{ enabled: true }}
      footer={{ enabled: true }}
    >
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <DocsBody>
        <MDXContent />
      </DocsBody>
    </DocsPage>
  );
}
```

### DocsPage Props

| Prop | Type | Description |
|------|------|-------------|
| `toc` | `TOCItem[]` | Table of contents items from `page.data.load()` |
| `tableOfContent` | `object` | TOC config: `style` (`'clerk'` or `'normal'`), `enabled` |
| `breadcrumb` | `object` | Breadcrumb config: `enabled`, `includeRoot` |
| `footer` | `object` | Prev/next navigation: `enabled` |
| `full` | `boolean` | Full-width layout (no TOC sidebar) |

## Static Generation

Always include `generateStaticParams` and `generateMetadata`:

```tsx
// app/docs/[[...slug]]/page.tsx

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
  };
}
```

## Home Layout

For non-docs pages (landing page, blog), use `HomeLayout`:

```tsx
// app/(home)/layout.tsx
import { HomeLayout } from 'fumadocs-ui/layouts/home';
import type { ReactNode } from 'react';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <HomeLayout
      nav={{
        title: 'My Project',
        links: [
          { text: 'Docs', url: '/docs' },
          { text: 'Blog', url: '/blog' },
        ],
        githubUrl: 'https://github.com/org/repo',
      }}
    >
      {children}
    </HomeLayout>
  );
}
```

## Custom Page Tree

For advanced navigation control, build a custom tree:

```tsx
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import type { PageTree } from 'fumadocs-core/server';

const tree: PageTree.Root = {
  name: 'Docs',
  children: [
    {
      type: 'page',
      name: 'Getting Started',
      url: '/docs/getting-started',
    },
    {
      type: 'separator',
      name: 'Guides',
    },
    {
      type: 'folder',
      name: 'Advanced',
      children: [
        {
          type: 'page',
          name: 'Configuration',
          url: '/docs/advanced/configuration',
        },
      ],
    },
  ],
};
```
