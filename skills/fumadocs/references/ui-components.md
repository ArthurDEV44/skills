# UI Components

## MDX Components Setup

Override default MDX components to use Fumadocs UI:

```tsx
// components/mdx-components.tsx
import defaultComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Callout } from 'fumadocs-ui/components/callout';
import { Card, Cards } from 'fumadocs-ui/components/card';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { Accordion, Accordions } from 'fumadocs-ui/components/accordion';
import { Steps, Step } from 'fumadocs-ui/components/steps';
import { ImageZoom } from 'fumadocs-ui/components/image-zoom';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    pre: ({ ref: _ref, ...props }) => (
      <CodeBlock {...props}>
        <Pre>{props.children}</Pre>
      </CodeBlock>
    ),
    Callout,
    Card,
    Cards,
    Tab,
    Tabs,
    Accordion,
    Accordions,
    Steps,
    Step,
    ImageZoom,
    ...components,
  };
}
```

Register in the page component:

```tsx
// app/docs/[[...slug]]/page.tsx
import { getMDXComponents } from '@/components/mdx-components';

// Inside the page component:
<DocsBody>
  <MDXContent components={getMDXComponents()} />
</DocsBody>
```

## Callout

Highlighted message boxes for warnings, info, and errors.

```mdx
<Callout type="info" title="Note">
  This is an informational callout.
</Callout>

<Callout type="warn" title="Warning">
  Be careful with this operation.
</Callout>

<Callout type="error" title="Danger">
  This action is irreversible.
</Callout>
```

Types: `info` (default), `warn`, `error`.

## Tabs

Tabbed content blocks with persistent selection:

```mdx
<Tabs items={['npm', 'pnpm', 'yarn']}>
  <Tab value="npm">
    ```bash
    npm install fumadocs-ui
    ```
  </Tab>
  <Tab value="pnpm">
    ```bash
    pnpm add fumadocs-ui
    ```
  </Tab>
  <Tab value="yarn">
    ```bash
    yarn add fumadocs-ui
    ```
  </Tab>
</Tabs>
```

Tab selection persists across page navigation when using the same `items`.

## Cards

Link cards for navigation:

```mdx
<Cards>
  <Card title="Getting Started" href="/docs/getting-started">
    Learn the basics of the framework.
  </Card>
  <Card title="API Reference" href="/docs/api">
    Complete API documentation.
  </Card>
</Cards>
```

## Accordion

Collapsible content sections:

```mdx
<Accordions>
  <Accordion title="What is Fumadocs?">
    Fumadocs is a documentation framework for Next.js.
  </Accordion>
  <Accordion title="How do I install it?">
    Run `npm install fumadocs-ui fumadocs-core fumadocs-mdx`.
  </Accordion>
</Accordions>
```

## Steps

Sequential step-by-step guides:

```mdx
<Steps>
  <Step>
    ### Install Dependencies

    ```bash
    npm install fumadocs-ui
    ```
  </Step>
  <Step>
    ### Configure Source

    Create `source.config.ts` in your project root.
  </Step>
  <Step>
    ### Create Pages

    Add your MDX content files.
  </Step>
</Steps>
```

## CodeBlock

Enhanced code blocks with copy button and title:

```tsx
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';

// Override the default pre element
<CodeBlock title="source.config.ts" allowCopy>
  <Pre>{children}</Pre>
</CodeBlock>
```

In MDX, code blocks automatically use Shiki syntax highlighting. Add a title with:

````mdx
```ts title="source.config.ts"
import { defineConfig } from 'fumadocs-mdx/config';
```
````

## ImageZoom

Zoomable images on click:

```mdx
<ImageZoom src="/screenshot.png" alt="Screenshot" width={800} height={400} />
```

Or use with `next/image`:

```tsx
import { ImageZoom } from 'fumadocs-ui/components/image-zoom';
import Image from 'next/image';

<ImageZoom>
  <Image src="/screenshot.png" alt="Screenshot" width={800} height={400} />
</ImageZoom>
```

## TypeTable

Display TypeScript type definitions as a table:

```tsx
import { TypeTable } from 'fumadocs-ui/components/type-table';

<TypeTable
  type={{
    name: { description: 'The name of the item', type: 'string', default: 'undefined' },
    count: { description: 'Number of items', type: 'number', default: '0' },
    enabled: { description: 'Whether the feature is enabled', type: 'boolean', default: 'true' },
  }}
/>
```

## Files (File Tree)

Display a file/folder tree:

```tsx
import { Files, Folder, File } from 'fumadocs-ui/components/files';

<Files>
  <Folder name="app" defaultOpen>
    <File name="layout.tsx" />
    <Folder name="docs">
      <File name="layout.tsx" />
      <Folder name="[[...slug]]">
        <File name="page.tsx" />
      </Folder>
    </Folder>
  </Folder>
  <File name="source.config.ts" />
</Files>
```

## Banner

Announcement banner at the top of the page:

```tsx
import { Banner } from 'fumadocs-ui/components/banner';

// In your layout:
<Banner>
  Fumadocs v14 is now available! <a href="/blog/v14">Read the announcement</a>
</Banner>
```

## Inline TOC

Render table of contents inline within the page body:

```tsx
import { InlineTOC } from 'fumadocs-ui/components/inline-toc';

<InlineTOC items={toc} />
```
