# Technical SEO Reference — 2025-2026

## Robots.txt — Complete Template

```
# ==============================================
# ROBOTS.TXT — SEO-Optimized Template (2026)
# ==============================================

# ---- Search Engine Crawlers (ALLOW) ----
User-agent: Googlebot
Allow: /
Disallow: /admin/
Disallow: /api/
Disallow: /private/
Disallow: /*?sort=
Disallow: /*?filter=
Disallow: /*?page=
Disallow: /search?
Crawl-delay: 0

User-agent: Bingbot
Allow: /
Disallow: /admin/
Disallow: /api/
Disallow: /private/
Crawl-delay: 1

User-agent: Yandex
Allow: /
Disallow: /admin/
Crawl-delay: 2

# ---- AI Search Crawlers (ALLOW for citation) ----
# These crawlers serve AI search products. Allow them
# if you want to be cited in AI-generated answers.

User-agent: PerplexityBot
Allow: /

User-agent: Googlebot-Extended
Allow: /
# Google AI Overviews / Gemini features

# ---- AI Training Crawlers (BLOCK or ALLOW — your choice) ----
# These crawlers scrape content for model training.
# Block them if you don't want content used for training.
# Allow them if you want maximum AI ecosystem presence.

User-agent: GPTBot
Disallow: /
# OpenAI training crawler

User-agent: ChatGPT-User
Allow: /
# ChatGPT browsing for live search (different from training)

User-agent: ClaudeBot
Disallow: /
# Anthropic training crawler

User-agent: CCBot
Disallow: /
# Common Crawl (used by many AI companies)

User-agent: anthropic-ai
Disallow: /

User-agent: Bytespider
Disallow: /
# ByteDance/TikTok crawler

User-agent: Amazonbot
Allow: /
# Amazon Alexa

# ---- Generic fallback ----
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /api/
Disallow: /private/
Disallow: /_next/
Disallow: /node_modules/

# ---- Sitemaps ----
Sitemap: https://www.example.com/sitemap.xml
Sitemap: https://www.example.com/sitemap-news.xml
```

**Key decisions to explain to the user:**
- `GPTBot` (training) vs `ChatGPT-User` (search) are DIFFERENT crawlers — block one, allow the other
- `Googlebot-Extended` controls AI features, NOT regular search indexing
- Blocking all AI crawlers means losing AI citation opportunities
- `Crawl-delay` is respected by Bing/Yandex but ignored by Google

## XML Sitemap Templates

### Standard Sitemap

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://www.example.com/</loc>
    <lastmod>2026-03-14</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://www.example.com/blog/article-slug</loc>
    <lastmod>2026-03-10</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
    <image:image>
      <image:loc>https://www.example.com/images/article-hero.webp</image:loc>
      <image:title>Descriptive image title</image:title>
    </image:image>
  </url>
</urlset>
```

### Sitemap Index (for large sites)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://www.example.com/sitemap-pages.xml</loc>
    <lastmod>2026-03-14</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://www.example.com/sitemap-blog.xml</loc>
    <lastmod>2026-03-14</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://www.example.com/sitemap-products.xml</loc>
    <lastmod>2026-03-14</lastmod>
  </sitemap>
</sitemapindex>
```

### Next.js Dynamic Sitemap (App Router)

```typescript
// app/sitemap.ts
import { MetadataRoute } from 'next';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  // Fetch all dynamic routes
  const posts = await getAllPosts();
  const products = await getAllProducts();

  const staticRoutes: MetadataRoute.Sitemap = [
    {
      url: 'https://www.example.com',
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 1,
    },
    {
      url: 'https://www.example.com/about',
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
  ];

  const blogRoutes: MetadataRoute.Sitemap = posts.map((post) => ({
    url: `https://www.example.com/blog/${post.slug}`,
    lastModified: post.updatedAt,
    changeFrequency: 'weekly' as const,
    priority: 0.7,
  }));

  const productRoutes: MetadataRoute.Sitemap = products.map((product) => ({
    url: `https://www.example.com/products/${product.slug}`,
    lastModified: product.updatedAt,
    changeFrequency: 'daily' as const,
    priority: 0.9,
  }));

  return [...staticRoutes, ...blogRoutes, ...productRoutes];
}
```

### Next.js Robots.txt (App Router)

```typescript
// app/robots.ts
import { MetadataRoute } from 'next';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: 'Googlebot',
        allow: '/',
        disallow: ['/admin/', '/api/', '/private/'],
      },
      {
        userAgent: 'GPTBot',
        disallow: '/',
      },
      {
        userAgent: 'ChatGPT-User',
        allow: '/',
      },
      {
        userAgent: '*',
        allow: '/',
        disallow: ['/admin/', '/api/'],
      },
    ],
    sitemap: 'https://www.example.com/sitemap.xml',
  };
}
```

## Canonical URL Patterns

### Why Canonicals Matter
- Consolidate ranking signals from duplicate/similar pages
- Prevent dilution from URL parameters, www/non-www, trailing slashes
- Signal to Google which version of a page to index

### Implementation

```html
<!-- Self-referencing canonical (every page should have one) -->
<link rel="canonical" href="https://www.example.com/current-page/" />
```

### Next.js App Router

```typescript
// app/blog/[slug]/page.tsx
import { Metadata } from 'next';

export async function generateMetadata({ params }): Promise<Metadata> {
  return {
    alternates: {
      canonical: `https://www.example.com/blog/${params.slug}`,
      languages: {
        'en': `https://www.example.com/blog/${params.slug}`,
        'fr': `https://fr.example.com/blog/${params.slug}`,
        'es': `https://es.example.com/blog/${params.slug}`,
      },
    },
  };
}
```

### Hreflang for Multi-Language

```html
<link rel="alternate" hreflang="en" href="https://www.example.com/page/" />
<link rel="alternate" hreflang="fr" href="https://fr.example.com/page/" />
<link rel="alternate" hreflang="es" href="https://es.example.com/pagina/" />
<link rel="alternate" hreflang="x-default" href="https://www.example.com/page/" />
```

## IndexNow Protocol

### What It Is
IndexNow lets you notify search engines (Bing, Yandex, and others) instantly when content
is published or updated, bypassing the standard crawl queue.

**Note:** Google does NOT support IndexNow natively. For Google, use the Google Indexing API
(limited to job postings and live streams) or submit via Search Console.

### Implementation

**Step 1:** Generate an API key (8-128 hex characters)

```bash
openssl rand -hex 32
```

**Step 2:** Host key file at your domain root

```
# https://www.example.com/YOUR-KEY.txt
# Contents: just the key value
a1b2c3d4e5f6...
```

**Step 3:** Submit URLs

```typescript
// lib/indexnow.ts
const INDEXNOW_KEY = process.env.INDEXNOW_KEY!;
const SITE_HOST = 'www.example.com';

export async function submitToIndexNow(urls: string[]) {
  const response = await fetch('https://api.indexnow.org/indexnow', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      host: SITE_HOST,
      key: INDEXNOW_KEY,
      keyLocation: `https://${SITE_HOST}/${INDEXNOW_KEY}.txt`,
      urlList: urls,
    }),
  });

  if (response.status === 200 || response.status === 202) {
    console.log(`IndexNow: ${urls.length} URLs submitted successfully`);
  } else {
    console.error(`IndexNow error: ${response.status}`);
  }
}

// Usage: call after publishing/updating content
await submitToIndexNow([
  'https://www.example.com/blog/new-article',
  'https://www.example.com/blog/updated-article',
]);
```

**Step 4:** Automate with CI/CD (GitHub Actions)

```yaml
# .github/workflows/indexnow.yml
name: Submit to IndexNow
on:
  push:
    branches: [main]
    paths:
      - 'content/**'
      - 'src/pages/**'

jobs:
  indexnow:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2
      - name: Get changed content files
        id: changed
        run: |
          echo "files=$(git diff --name-only HEAD~1 HEAD -- content/ src/pages/ | tr '\n' ',')" >> $GITHUB_OUTPUT
      - name: Submit to IndexNow
        if: steps.changed.outputs.files != ''
        run: |
          # Convert file paths to URLs and submit
          node scripts/submit-indexnow.js "${{ steps.changed.outputs.files }}"
        env:
          INDEXNOW_KEY: ${{ secrets.INDEXNOW_KEY }}
```

## Crawl Budget Optimization

For sites with 10,000+ pages:

1. **Block faceted navigation** in robots.txt: `Disallow: /*?sort=`, `Disallow: /*?filter=`
2. **Paginate wisely**: use `rel="next"` / `rel="prev"` (though Google says they don't use them, Bing does)
3. **Set `noindex` on thin pages**: tag/category pages with < 3 items, search result pages
4. **Consolidate with canonicals**: URL parameter variants → canonical to clean URL
5. **Monitor crawl stats**: Google Search Console → Settings → Crawl Stats
6. **Flat architecture**: every important page within 3 clicks of homepage
7. **Internal linking**: link from high-authority pages to pages you want crawled more frequently

## JavaScript Rendering Strategy Decision Tree

```
Is the content publicly available and needs SEO?
├── YES → Does it change per-user or per-request?
│   ├── YES → SSR (Server-Side Rendering)
│   │   └── Next.js: default with App Router (server components)
│   │   └── Nuxt: `ssr: true` in nuxt.config
│   │   └── SvelteKit: default behavior
│   └── NO → SSG (Static Site Generation) or ISR
│       ├── Content changes < 1x/day → SSG
│       │   └── Next.js: `export const dynamic = 'force-static'`
│       │   └── Astro: default behavior
│       └── Content changes > 1x/day → ISR
│           └── Next.js: `export const revalidate = 3600` (seconds)
│           └── Nuxt: `routeRules: { '/blog/**': { isr: 3600 } }`
└── NO → CSR (Client-Side Rendering) is fine
    └── Dashboards, admin panels, authenticated-only content
```

**Critical rules:**
- NEVER render `<title>`, `<meta>`, `<h1>`, or primary content client-side only
- ALWAYS ensure `<a href>` links exist in initial HTML for crawlers
- TEST with `curl -A Googlebot URL` to see what crawlers receive
- CHECK production builds for hydration mismatches

## URL Structure Best Practices

| Pattern | Example | SEO Impact |
|---------|---------|------------|
| Clean, descriptive | `/blog/seo-guide-2026` | Excellent |
| Category prefix | `/guides/seo/technical` | Good (adds hierarchy) |
| Date-based | `/2026/03/seo-guide` | OK for news, bad for evergreen |
| Query parameters | `/search?q=seo` | Poor (usually noindex) |
| Hash fragments | `/page#section` | Not crawled by Google |
| Dynamic IDs | `/post/12345` | Poor (no keyword signal) |

Rules:
- Use hyphens, not underscores (`seo-guide` not `seo_guide`)
- Lowercase only
- Keep under 75 characters
- No stop words unless needed for readability
- Consistent trailing slash policy (pick one and redirect the other)
