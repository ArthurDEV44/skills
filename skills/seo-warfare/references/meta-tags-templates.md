# Meta Tags Templates — Ready-to-Use (2026)

## Complete HTML Head Template

```html
<!DOCTYPE html>
<html lang="{{lang_code}}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- ============ CORE SEO ============ -->
  <title>{{primary_keyword}} - {{secondary_keyword}} | {{brand_name}}</title>
  <meta name="description" content="{{benefit_statement_with_keyword_120_155_chars}}. {{cta}}." />
  <link rel="canonical" href="https://www.{{domain}}/{{page_path}}/" />

  <!-- ============ ROBOTS ============ -->
  <meta name="robots" content="index, follow, max-snippet:-1, max-image-preview:large, max-video-preview:-1" />
  <!-- max-snippet:-1 = no limit on snippet length (good for featured snippets) -->
  <!-- max-image-preview:large = allow large image previews in search -->

  <!-- ============ OPEN GRAPH (Facebook, LinkedIn, AI parsing) ============ -->
  <meta property="og:title" content="{{primary_keyword}} - {{brand_name}}" />
  <meta property="og:description" content="{{benefit_statement_120_chars}}" />
  <meta property="og:image" content="https://www.{{domain}}/og/{{page_slug}}.jpg" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta property="og:url" content="https://www.{{domain}}/{{page_path}}/" />
  <meta property="og:type" content="{{og_type}}" />
  <meta property="og:site_name" content="{{site_name}}" />
  <meta property="og:locale" content="{{locale}}" />

  <!-- ============ TWITTER CARD ============ -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:site" content="@{{twitter_handle}}" />
  <meta name="twitter:creator" content="@{{author_twitter}}" />
  <meta name="twitter:title" content="{{primary_keyword}} - {{brand_name}}" />
  <meta name="twitter:description" content="{{benefit_statement_120_chars}}" />
  <meta name="twitter:image" content="https://www.{{domain}}/og/{{page_slug}}.jpg" />

  <!-- ============ MULTI-LANGUAGE (if applicable) ============ -->
  <link rel="alternate" hreflang="en" href="https://www.{{domain}}/{{page_path}}/" />
  <link rel="alternate" hreflang="fr" href="https://fr.{{domain}}/{{page_path_fr}}/" />
  <link rel="alternate" hreflang="es" href="https://es.{{domain}}/{{page_path_es}}/" />
  <link rel="alternate" hreflang="x-default" href="https://www.{{domain}}/{{page_path}}/" />

  <!-- ============ PERFORMANCE ============ -->
  <!-- Preload LCP image -->
  <link rel="preload" as="image" href="/images/{{hero_image}}.webp" type="image/webp" fetchpriority="high" />
  <!-- Preload critical font -->
  <link rel="preload" as="font" href="/fonts/{{font_file}}.woff2" type="font/woff2" crossorigin="anonymous" />
  <!-- Preconnect to CDN and third-party origins -->
  <link rel="preconnect" href="https://cdn.{{domain}}" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <!-- DNS prefetch for analytics/third-party -->
  <link rel="dns-prefetch" href="https://www.googletagmanager.com" />

  <!-- ============ FAVICONS ============ -->
  <link rel="icon" href="/favicon.ico" sizes="32x32" />
  <link rel="icon" href="/icon.svg" type="image/svg+xml" />
  <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
  <link rel="manifest" href="/manifest.webmanifest" />
  <meta name="theme-color" content="{{theme_color}}" />

  <!-- ============ STRUCTURED DATA ============ -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Organization",
        "name": "{{company_name}}",
        "url": "https://www.{{domain}}",
        "logo": "https://www.{{domain}}/logo.png",
        "sameAs": ["{{social_url_1}}", "{{social_url_2}}"]
      },
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://www.{{domain}}" },
          { "@type": "ListItem", "position": 2, "name": "{{category}}", "item": "https://www.{{domain}}/{{category_slug}}" },
          { "@type": "ListItem", "position": 3, "name": "{{page_title}}" }
        ]
      }
    ]
  }
  </script>
</head>
```

## Page-Type Specific Templates

### Homepage

```html
<title>{{Brand Name}} — {{Value Proposition in 5-8 Words}}</title>
<meta name="description" content="{{Brand}} {{does_what}} for {{audience}}. {{Key_differentiator}}. {{CTA}}." />
<meta property="og:type" content="website" />
```

Example:
```html
<title>Acme Analytics — Real-Time Business Intelligence for SaaS</title>
<meta name="description" content="Acme Analytics delivers real-time dashboards and AI-powered insights for SaaS teams. Track MRR, churn, and growth metrics in one platform. Start free today." />
```

### Blog Post / Article

```html
<title>{{Article Title}} | {{Brand}}</title>
<meta name="description" content="{{Summary_of_key_insight_120_155_chars}}. Updated {{month}} {{year}}." />
<meta property="og:type" content="article" />
<meta property="article:published_time" content="{{ISO_8601}}" />
<meta property="article:modified_time" content="{{ISO_8601}}" />
<meta property="article:author" content="https://www.{{domain}}/about/{{author}}" />
<meta property="article:section" content="{{category}}" />
<meta property="article:tag" content="{{tag1}}" />
<meta property="article:tag" content="{{tag2}}" />
```

### Product Page

```html
<title>{{Product Name}} — {{Key Feature}} | {{Brand}}</title>
<meta name="description" content="{{Product}} {{key_benefit}}. {{Price_point}}. {{Social_proof}}. {{CTA}}." />
<meta property="og:type" content="product" />
<meta property="product:price:amount" content="{{price}}" />
<meta property="product:price:currency" content="{{currency}}" />
```

### Category / Collection Page

```html
<title>{{Category Name}} — {{Qualifying Phrase}} | {{Brand}}</title>
<meta name="description" content="Browse {{count}}+ {{category_items}}. {{Filter_options}}. {{CTA}}." />
<meta property="og:type" content="website" />
```

### Landing Page

```html
<title>{{Primary Keyword}} — {{Outcome Promise}} | {{Brand}}</title>
<meta name="description" content="{{How_you_help_them_achieve_outcome}}. {{Social_proof_stat}}. {{CTA}}." />
<meta property="og:type" content="website" />
<!-- Landing pages: consider noindex if thin/temporary -->
```

### About Page

```html
<title>About {{Brand}} — {{Mission Statement Short}} | {{Brand}}</title>
<meta name="description" content="{{Brand}} was founded in {{year}} to {{mission}}. {{Team_size}}, {{notable_clients_or_stats}}." />
```

### Local Business Page

```html
<title>{{Business Name}} — {{Service}} in {{City}}, {{State}}</title>
<meta name="description" content="{{Business}} offers {{services}} in {{city}}. {{Differentiator}}. Call {{phone}} or visit us at {{address}}." />
```

## Next.js App Router Metadata

```typescript
// app/layout.tsx — Global defaults
import { Metadata } from 'next';

export const metadata: Metadata = {
  metadataBase: new URL('https://www.example.com'),
  title: {
    template: '%s | Brand Name',
    default: 'Brand Name — Value Proposition',
  },
  description: 'Default site description.',
  openGraph: {
    type: 'website',
    siteName: 'Brand Name',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    site: '@handle',
  },
  robots: {
    index: true,
    follow: true,
    'max-snippet': -1,
    'max-image-preview': 'large',
    'max-video-preview': -1,
  },
  verification: {
    google: 'your-google-verification-code',
  },
};
```

```typescript
// app/blog/[slug]/page.tsx — Per-page metadata
import { Metadata } from 'next';

export async function generateMetadata({ params }): Promise<Metadata> {
  const post = await getPost(params.slug);

  return {
    title: post.title,
    description: post.excerpt,
    openGraph: {
      type: 'article',
      title: post.title,
      description: post.excerpt,
      images: [{ url: post.ogImage, width: 1200, height: 630, alt: post.title }],
      publishedTime: post.publishedAt,
      modifiedTime: post.updatedAt,
      authors: [post.author.name],
      section: post.category,
      tags: post.tags,
    },
    twitter: {
      title: post.title,
      description: post.excerpt,
      images: [post.ogImage],
    },
    alternates: {
      canonical: `/blog/${params.slug}`,
    },
  };
}
```

## Title Tag Rules

| Rule | Example |
|------|---------|
| Length: 50-60 chars | "Technical SEO Guide 2026 \| Acme" (35 chars) |
| Keyword near front | "SEO Audit Checklist — 47 Points \| Acme" |
| Brand at end | "... \| Brand Name" or "... — Brand Name" |
| No keyword stuffing | BAD: "SEO SEO Guide SEO Tips SEO 2026" |
| Unique per page | Every page has a distinct title |
| Match search intent | Informational: "How to...", Commercial: "Best...", "Top..." |

## Meta Description Rules

| Rule | Example |
|------|---------|
| Length: 120-155 chars | "Learn the 47-point technical SEO checklist..." |
| Include primary keyword | Natural placement, not forced |
| Include CTA | "Get started free", "Download now", "Learn more" |
| Unique per page | Every page has a distinct description |
| Not a ranking factor | But directly impacts CTR — write for humans |
| Include numbers/dates | "Updated March 2026", "47 actionable tips" |

## Open Graph Image Specifications

| Platform | Recommended Size | Minimum | Aspect Ratio |
|----------|-----------------|---------|--------------|
| Facebook | 1200 x 630 | 600 x 315 | 1.91:1 |
| Twitter | 1200 x 630 | 300 x 157 | 1.91:1 |
| LinkedIn | 1200 x 627 | 1200 x 627 | 1.91:1 |
| WhatsApp | 1200 x 630 | 300 x 200 | 1.91:1 |

**Rules:**
- Use 1200 x 630 as the universal size
- Include brand logo/name
- Use readable text (2-5 words max)
- JPEG or PNG, < 5MB
- Test with Facebook Sharing Debugger: https://developers.facebook.com/tools/debug/
