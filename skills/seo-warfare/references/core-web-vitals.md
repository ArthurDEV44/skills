# Core Web Vitals Optimization — 2025-2026

## Thresholds

| Metric | Good | Needs Improvement | Poor |
|--------|------|-------------------|------|
| **LCP** (Largest Contentful Paint) | <= 2.5s | 2.5-4.0s | > 4.0s |
| **CLS** (Cumulative Layout Shift) | <= 0.1 | 0.1-0.25 | > 0.25 |
| **INP** (Interaction to Next Paint) | <= 200ms | 200-500ms | > 500ms |

INP replaced FID in March 2024. 43% of sites still fail INP in 2026.

## Performance Budgets

| Category | Target |
|----------|--------|
| JavaScript (compressed) | < 300 KB |
| CSS (compressed) | < 80 KB |
| Hero image | < 200 KB |
| Total page weight | < 1.5 MB |
| Third-party scripts | < 5 scripts |
| LCP target | < 2.0s (buffer below 2.5s) |
| INP target | < 150ms (buffer below 200ms) |
| CLS target | < 0.05 (buffer below 0.1) |

## LCP Optimization

### 1. Preload Hero Image (highest impact: 200-800ms reduction)

```html
<head>
  <!-- Preload BEFORE other resources -->
  <link rel="preload" as="image" href="/hero.webp" type="image/webp" fetchpriority="high" />
  <link rel="preconnect" href="https://cdn.example.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
</head>
```

### 2. Responsive Hero Image

```html
<picture>
  <source
    srcset="/hero-800.avif 800w, /hero-1200.avif 1200w, /hero-1600.avif 1600w"
    type="image/avif"
    sizes="(max-width: 800px) 100vw, (max-width: 1200px) 80vw, 1200px"
  />
  <source
    srcset="/hero-800.webp 800w, /hero-1200.webp 1200w, /hero-1600.webp 1600w"
    type="image/webp"
    sizes="(max-width: 800px) 100vw, (max-width: 1200px) 80vw, 1200px"
  />
  <img
    src="/hero-1200.jpg"
    alt="Descriptive alt text"
    width="1200"
    height="630"
    fetchpriority="high"
    decoding="async"
  />
</picture>
```

**Rules:**
- NEVER `loading="lazy"` on LCP image
- ALWAYS `fetchpriority="high"` on LCP image
- Use `<img>` not CSS `background-image` for LCP element
- Include `width` and `height` attributes

### 3. Next.js Hero Image

```tsx
import Image from 'next/image';

export function Hero() {
  return (
    <Image
      src="/hero.jpg"
      alt="Hero description"
      width={1200}
      height={630}
      priority          // Sets fetchpriority="high" + preloads
      sizes="100vw"
      quality={85}
    />
  );
}
```

### 4. Critical CSS Inlining

```html
<head>
  <style>
    /* Only above-the-fold styles */
    body { margin: 0; font-family: 'Inter', system-ui, sans-serif; }
    .hero { /* hero styles */ }
    .nav { /* nav styles */ }
  </style>

  <!-- Non-critical CSS loads async -->
  <link rel="preload" as="style" href="/styles/main.css"
        onload="this.onload=null;this.rel='stylesheet'" />
  <noscript><link rel="stylesheet" href="/styles/main.css" /></noscript>
</head>
```

### 5. Font Optimization

```css
/* Variable font with swap for instant text rendering */
@font-face {
  font-family: 'Inter';
  src: url('/fonts/inter-var.woff2') format('woff2');
  font-display: swap;
  font-weight: 100 900;
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+2000-206F;
}

/* Metric-matched fallback to prevent CLS during font swap */
@font-face {
  font-family: 'Inter Fallback';
  src: local('Arial');
  size-adjust: 107.64%;
  ascent-override: 90%;
  descent-override: 22.43%;
  line-gap-override: 0%;
}

body {
  font-family: 'Inter', 'Inter Fallback', system-ui, sans-serif;
}
```

```html
<link rel="preload" as="font" href="/fonts/inter-var.woff2"
      type="font/woff2" crossorigin="anonymous" />
```

**Next.js automatic font optimization:**

```tsx
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
```

### 6. Server Response Time (TTFB)

- Target TTFB < 200ms
- Use CDN (Cloudflare, Vercel Edge, Fastly)
- Enable HTTP/2 or HTTP/3
- Database query optimization
- Enable Brotli or gzip compression
- Use edge caching for static assets

## CLS Optimization

### 1. Explicit Dimensions on All Media

```html
<!-- ALWAYS include width and height -->
<img src="/photo.jpg" alt="Product" width="800" height="600" loading="lazy" />
<video width="1280" height="720" poster="/poster.jpg"></video>
<iframe width="560" height="315" src="..." title="..."></iframe>
```

### 2. CSS aspect-ratio

```css
.responsive-image {
  aspect-ratio: 16 / 9;
  width: 100%;
  height: auto;
  object-fit: cover;
}

.square-avatar {
  aspect-ratio: 1;
  width: 48px;
  height: auto;
  border-radius: 50%;
}
```

### 3. Reserve Space for Dynamic Content

```css
/* Ad slots */
.ad-leaderboard { min-height: 90px; contain: layout; }
.ad-sidebar { min-height: 250px; contain: layout; }

/* Cookie banners - MUST be fixed position */
.cookie-banner {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 50;
}

/* Skeleton loading states */
.skeleton {
  min-height: 200px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
```

### 4. Font Swap CLS Prevention

Use the metric-matched fallback font pattern shown in LCP section above.
The `size-adjust`, `ascent-override`, `descent-override`, and `line-gap-override`
properties ensure the fallback font occupies the exact same space as the web font,
preventing layout shift during font swap.

## INP Optimization

### 1. Break Long Tasks with scheduler.yield()

```javascript
// Polyfill for browsers without scheduler API
if (!('scheduler' in globalThis)) {
  globalThis.scheduler = {
    yield: () => new Promise(resolve => setTimeout(resolve, 0)),
  };
}

async function processLargeList(items, processItem) {
  const CHUNK = 500;
  const results = [];

  for (let i = 0; i < items.length; i += CHUNK) {
    const chunk = items.slice(i, i + CHUNK);
    for (const item of chunk) {
      results.push(processItem(item));
    }
    if (i + CHUNK < items.length) {
      await scheduler.yield(); // Yield to browser between chunks
    }
  }

  return results;
}
```

### 2. Batch DOM Reads Before Writes

```javascript
// BAD: Interleaved reads/writes force multiple reflows
function resizeCards(cards) {
  cards.forEach(card => {
    const height = card.offsetHeight;       // READ -> reflow
    card.style.width = height * 1.5 + 'px'; // WRITE -> invalidate
  });
}

// GOOD: All reads first, then all writes
function resizeCardsBatched(cards) {
  const heights = cards.map(c => c.offsetHeight); // All READs (one reflow)
  cards.forEach((card, i) => {
    card.style.width = heights[i] * 1.5 + 'px';   // All WRITEs (one reflow)
  });
}
```

### 3. Web Workers for Heavy Processing

```javascript
// worker.js
self.addEventListener('message', ({ data }) => {
  if (data.type === 'sort') {
    const sorted = data.items.sort((a, b) =>
      data.dir === 'asc'
        ? (a[data.field] > b[data.field] ? 1 : -1)
        : (a[data.field] < b[data.field] ? 1 : -1)
    );
    self.postMessage({ type: 'sorted', items: sorted });
  }
});

// main.js
const worker = new Worker('/worker.js');

document.getElementById('sort-btn').addEventListener('click', () => {
  worker.postMessage({
    type: 'sort',
    items: currentProducts,
    field: 'price',
    dir: 'asc',
  });
});

worker.addEventListener('message', ({ data }) => {
  if (data.type === 'sorted') renderProducts(data.items);
});
```

### 4. React-Specific INP Optimization

```tsx
import { useTransition, useMemo, useCallback } from 'react';

function ProductList({ products, filter }) {
  const [isPending, startTransition] = useTransition();

  const handleFilter = useCallback((newFilter: string) => {
    // Mark as transition — React will yield to browser
    startTransition(() => {
      setFilter(newFilter);
    });
  }, []);

  const filteredProducts = useMemo(
    () => products.filter(p => p.category === filter),
    [products, filter]
  );

  return (
    <div style={{ opacity: isPending ? 0.7 : 1 }}>
      {filteredProducts.map(p => <ProductCard key={p.id} product={p} />)}
    </div>
  );
}
```

### 5. Defer Third-Party Scripts

```html
<!-- BAD: blocking -->
<script src="https://analytics.example.com/script.js"></script>

<!-- GOOD: defer (executes after HTML parsed) -->
<script defer src="https://analytics.example.com/script.js"></script>

<!-- GOOD: async (executes when available, non-blocking) -->
<script async src="https://analytics.example.com/script.js"></script>

<!-- BEST: load after user interaction (IdleCallback) -->
<script>
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      const s = document.createElement('script');
      s.src = 'https://analytics.example.com/script.js';
      document.head.appendChild(s);
    });
  }
</script>
```

## Production Measurement

### web-vitals Library

```javascript
import { onLCP, onCLS, onINP } from 'web-vitals/attribution';

function sendToAnalytics(metric) {
  const body = {
    name: metric.name,
    value: metric.value,
    rating: metric.rating,       // 'good', 'needs-improvement', 'poor'
    delta: metric.delta,
    id: metric.id,
    url: location.pathname,
    // Attribution data (tells you WHAT caused the issue)
    ...(metric.attribution && {
      element: metric.attribution.element,
      largestShiftTarget: metric.attribution.largestShiftTarget,
      interactionTarget: metric.attribution.interactionTarget,
      inputDelay: metric.attribution.inputDelay,
      processingDuration: metric.attribution.processingDuration,
      presentationDelay: metric.attribution.presentationDelay,
    }),
  };

  navigator.sendBeacon('/api/analytics/vitals', JSON.stringify(body));
}

onLCP(sendToAnalytics);
onCLS(sendToAnalytics);
onINP(sendToAnalytics);
```

## Lighthouse CI Configuration

```json
{
  "ci": {
    "collect": {
      "numberOfRuns": 3,
      "url": ["https://www.example.com/", "https://www.example.com/blog/"]
    },
    "assert": {
      "assertions": {
        "categories:performance": ["error", { "minScore": 0.9 }],
        "categories:accessibility": ["error", { "minScore": 0.9 }],
        "categories:best-practices": ["error", { "minScore": 0.9 }],
        "categories:seo": ["error", { "minScore": 0.95 }],
        "largest-contentful-paint": ["error", { "maxNumericValue": 2500 }],
        "cumulative-layout-shift": ["error", { "maxNumericValue": 0.1 }],
        "total-byte-weight": ["warning", { "maxNumericValue": 1500000 }],
        "dom-size": ["warning", { "maxNumericValue": 1500 }]
      }
    }
  }
}
```

## INP Quick Diagnostic

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| INP > 500ms on all interactions | Main thread blocked by JS | Break tasks with `scheduler.yield()` |
| INP spikes on specific button | Heavy event handler | Move to Web Worker |
| INP bad only on mobile | Too much JS for device | Code-split, reduce bundle |
| INP degrades over time | Memory leak or DOM bloat | Profile with DevTools Memory |
| INP > 200ms on scroll handlers | Layout thrashing | Batch DOM reads/writes |
