# Optimization Patterns — Performance, Lighthouse, Bundle, Backend

## Table of Contents

- [Lighthouse / Core Web Vitals (2026)](#lighthouse--core-web-vitals-2026)
- [Bundle Size Optimization](#bundle-size-optimization)
- [React / Frontend Performance](#react--frontend-performance)
- [Backend Performance](#backend-performance)
- [Database Query Optimization](#database-query-optimization)
- [Memory Management](#memory-management)
- [Rust-Specific Optimization](#rust-specific-optimization)

---

## Lighthouse / Core Web Vitals (2026)

### Current thresholds (March 2024+ — INP replaced FID):

| Metric | Good | Needs Improvement | Poor |
|--------|------|-------------------|------|
| LCP (Largest Contentful Paint) | < 2.5s | 2.5s - 4.0s | > 4.0s |
| INP (Interaction to Next Paint) | < 200ms | 200ms - 500ms | > 500ms |
| CLS (Cumulative Layout Shift) | < 0.1 | 0.1 - 0.25 | > 0.25 |

### Fix priority order: TTFB -> LCP -> INP -> CLS

### TTFB Optimization (target < 800ms):

```
1. Enable server-side caching (Redis, CDN edge caching)
2. Use CDN for static assets with immutable cache headers
3. Enable HTTP/2 or HTTP/3
4. Optimize server-side rendering time
5. Use stale-while-revalidate cache strategy
```

### LCP Optimization (target < 2.5s):

```html
<!-- Preload hero image -->
<link rel="preload" as="image" href="/hero.webp" fetchpriority="high">

<!-- Inline critical CSS -->
<style>/* above-the-fold styles inlined */</style>

<!-- Defer non-critical CSS -->
<link rel="stylesheet" href="/non-critical.css" media="print" onload="this.media='all'">

<!-- Preconnect to required origins -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://cdn.example.com" crossorigin>
```

**Key actions:**
- Serve images in WebP/AVIF format with fallback
- Set `fetchpriority="high"` on the LCP element
- Use `loading="lazy"` ONLY on below-fold images (never on LCP image)
- Inline critical CSS, defer the rest
- Eliminate render-blocking JavaScript (`defer` or `async` attributes)
- Server-side render (SSR) the above-the-fold content

### INP Optimization (target < 200ms):

```javascript
// Break long tasks with scheduler.yield() (Chrome 115+)
async function processItems(items) {
  for (const item of items) {
    processItem(item);
    // Yield to the main thread every iteration
    if (navigator.scheduling?.isInputPending?.()) {
      await scheduler.yield();
    }
  }
}

// Or use setTimeout as fallback
function yieldToMain() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

// Debounce expensive event handlers
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// Offload heavy computation to Web Workers
const worker = new Worker('/heavy-computation.js');
worker.postMessage(data);
worker.onmessage = (e) => updateUI(e.data);
```

**Key actions:**
- Break any JavaScript task >50ms into smaller chunks
- Use `requestIdleCallback` for non-urgent work
- Debounce/throttle input handlers (scroll, resize, keypress)
- Move heavy computation off the main thread (Web Workers)
- Avoid forced synchronous layouts (read-then-write patterns)
- Use CSS `content-visibility: auto` for off-screen content

### CLS Optimization (target < 0.1):

```html
<!-- Always set explicit dimensions on media -->
<img src="photo.webp" width="800" height="600" alt="...">
<video width="1280" height="720"></video>
<iframe width="560" height="315"></iframe>

<!-- Reserve space for dynamic content -->
<div style="min-height: 250px;"><!-- ad slot --></div>
```

```css
/* Prevent font-swap layout shift */
@font-face {
  font-family: 'CustomFont';
  src: url('/font.woff2') format('woff2');
  font-display: swap;
  /* Size-adjust fallback to match metrics */
  size-adjust: 105%;
  ascent-override: 95%;
}
```

**Key actions:**
- Set explicit `width` and `height` on ALL images, videos, iframes
- Use `aspect-ratio` CSS property for responsive containers
- Use `font-display: swap` with size-adjusted fallback fonts
- Reserve space for dynamically injected content (ads, embeds, lazy content)
- Avoid inserting content above existing content after page load
- Use CSS `transform` for animations instead of properties that trigger layout

---

## Bundle Size Optimization

### Tree-shaking prerequisites:

1. Use ESM syntax (`import`/`export`) — CommonJS cannot be tree-shaken
2. Mark packages `"sideEffects": false` in `package.json`
3. Use production mode (`NODE_ENV=production`)
4. Avoid barrel files (`index.ts` that re-exports everything) — they defeat tree-shaking

### Code splitting strategies:

```javascript
// Route-based splitting (React)
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));
const AdminPanel = lazy(() => import('./pages/AdminPanel'));

function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/admin" element={<AdminPanel />} />
      </Routes>
    </Suspense>
  );
}

// Feature-based splitting
const HeavyChart = lazy(() => import('./components/HeavyChart'));
const PdfExporter = lazy(() => import('./components/PdfExporter'));
```

### Common heavy-dependency replacements:

| Heavy Library | Size | Lighter Alternative | Size |
|--------------|------|-------------------|------|
| `moment` | 290KB | `date-fns` (individual) | 2-10KB |
| `lodash` (full) | 70KB | `lodash-es` (individual) | 1-5KB |
| `axios` | 14KB | `fetch` (native) | 0KB |
| `uuid` | 12KB | `crypto.randomUUID()` | 0KB |
| `classnames` | 1KB | Template literals | 0KB |
| `numeral` | 35KB | `Intl.NumberFormat` | 0KB |

### Import optimization:

```javascript
// BAD — imports entire library
import _ from 'lodash';
import { format } from 'date-fns';
import * as Icons from '@heroicons/react/24/outline';

// GOOD — import only what you need
import debounce from 'lodash/debounce';
import { format } from 'date-fns/format';
import { HomeIcon } from '@heroicons/react/24/outline/HomeIcon';
```

### Analysis commands:

```bash
# Vite
npx vite-bundle-visualizer

# Webpack
npx webpack-bundle-analyzer stats.json

# Generic
npx source-map-explorer dist/**/*.js

# Check individual package sizes
npx bundlephobia <package-name>
```

---

## React / Frontend Performance

### Memoization (use sparingly — only for measured bottlenecks):

```javascript
// Expensive computation
const sortedItems = useMemo(() => {
  return items.sort((a, b) => complexCompare(a, b));
}, [items]);

// Stable callback passed to memoized children
const handleClick = useCallback((id) => {
  dispatch({ type: 'SELECT', id });
}, [dispatch]);

// Memoize child component only when re-renders are measured as expensive
const ExpensiveList = memo(function ExpensiveList({ items }) {
  return items.map(item => <ExpensiveItem key={item.id} item={item} />);
});
```

### Virtualization for long lists:

```javascript
// Use @tanstack/react-virtual for lists >100 items
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualList({ items }) {
  const parentRef = useRef(null);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
  });
  // ... render only visible items
}
```

### State management:

- Colocate state — keep state as close to where it's used as possible
- Split context — avoid one giant context that re-renders everything
- Use `useSyncExternalStore` for external state libraries
- Avoid derived state in `useState` — compute during render or use `useMemo`

---

## Backend Performance

### API response optimization:

```
1. Paginate all list endpoints (cursor-based for large datasets)
2. Use field selection (GraphQL) or sparse fieldsets (REST)
3. Compress responses (gzip/brotli middleware)
4. Set appropriate Cache-Control headers
5. Use ETags for conditional requests
6. Implement connection pooling for database connections
```

### Async patterns:

```javascript
// BAD — sequential when parallelizable
const user = await getUser(id);
const orders = await getOrders(id);
const reviews = await getReviews(id);

// GOOD — parallel independent queries
const [user, orders, reviews] = await Promise.all([
  getUser(id),
  getOrders(id),
  getReviews(id),
]);

// GOOD — parallel with error isolation
const results = await Promise.allSettled([
  getUser(id),
  getOrders(id),
  getReviews(id),
]);
```

### Caching strategies:

| Pattern | Use Case | TTL |
|---------|----------|-----|
| In-memory (Map/LRU) | Hot config, feature flags | 5-60 min |
| Redis/Valkey | Session data, API responses | 1-60 min |
| CDN edge cache | Static assets, public API responses | 1h-1y |
| Stale-while-revalidate | Content that can be briefly stale | Background refresh |
| ETag / If-None-Match | Resources that change unpredictably | Until modified |

---

## Database Query Optimization

### N+1 Query Detection and Fix:

```javascript
// BAD — N+1 (1 query + N queries)
const posts = await db.query('SELECT * FROM posts LIMIT 20');
for (const post of posts) {
  post.author = await db.query('SELECT * FROM users WHERE id = ?', [post.author_id]);
}

// GOOD — JOIN (1 query)
const posts = await db.query(`
  SELECT p.*, u.name as author_name, u.avatar as author_avatar
  FROM posts p
  JOIN users u ON u.id = p.author_id
  LIMIT 20
`);

// GOOD — Batch loading (2 queries)
const posts = await db.query('SELECT * FROM posts LIMIT 20');
const authorIds = [...new Set(posts.map(p => p.author_id))];
const authors = await db.query('SELECT * FROM users WHERE id IN (?)', [authorIds]);
const authorMap = new Map(authors.map(a => [a.id, a]));
posts.forEach(p => p.author = authorMap.get(p.author_id));
```

### Pagination:

```sql
-- BAD — OFFSET gets slower as page number increases
SELECT * FROM posts ORDER BY created_at DESC LIMIT 20 OFFSET 10000;

-- GOOD — Cursor/keyset pagination (constant performance)
SELECT * FROM posts
WHERE created_at < '2026-03-01T00:00:00Z'
ORDER BY created_at DESC
LIMIT 20;
```

### Index strategy:

```
1. Index all columns used in WHERE clauses
2. Index all columns used in JOIN conditions
3. Index columns used in ORDER BY (to avoid filesort)
4. Use composite indexes for multi-column queries (leftmost prefix rule)
5. Avoid indexing low-cardinality columns (boolean, enum with <5 values)
6. Monitor slow query logs to find missing indexes
```

---

## Memory Management

### JavaScript memory leak patterns and fixes:

```javascript
// LEAK — Event listeners not cleaned up
useEffect(() => {
  window.addEventListener('resize', handleResize);
  // Missing cleanup!
}, []);

// FIX — Always return cleanup function
useEffect(() => {
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);

// LEAK — Interval not cleared
useEffect(() => {
  const id = setInterval(pollData, 5000);
  // Missing cleanup!
}, []);

// FIX
useEffect(() => {
  const id = setInterval(pollData, 5000);
  return () => clearInterval(id);
}, []);

// LEAK — AbortController not used for fetch
useEffect(() => {
  fetch('/api/data').then(r => r.json()).then(setData);
}, []);

// FIX — Cancel on unmount
useEffect(() => {
  const controller = new AbortController();
  fetch('/api/data', { signal: controller.signal })
    .then(r => r.json())
    .then(setData)
    .catch(e => { if (e.name !== 'AbortError') throw e; });
  return () => controller.abort();
}, []);
```

### Node.js / server-side:

```
1. Use WeakRef and WeakMap for cache-like structures
2. Set max size on all in-memory caches (LRU eviction)
3. Close database connections and file handles in finally blocks
4. Monitor heap usage: process.memoryUsage().heapUsed
5. Use --max-old-space-size to set explicit memory limits
6. Profile with --inspect and Chrome DevTools Memory tab
```

---

## Rust-Specific Optimization

### Common performance improvements:

```rust
// AVOID — unnecessary clones
let name = user.name.clone();  // if user lives long enough, borrow instead
process(&user.name);           // pass reference

// AVOID — allocation in loops
for item in items {
    let mut buf = String::new();  // allocates every iteration
    // ...
}
// BETTER — allocate once, reuse
let mut buf = String::new();
for item in items {
    buf.clear();  // reuse allocation
    // ...
}

// AVOID — collecting when iterator suffices
let filtered: Vec<_> = items.iter().filter(|x| x.active).collect();
for item in filtered { /* ... */ }
// BETTER — chain iterators
for item in items.iter().filter(|x| x.active) { /* ... */ }

// Use &str for function parameters when ownership not needed
fn greet(name: &str) { /* ... */ }  // not fn greet(name: String)

// Use Cow<str> when sometimes owned, sometimes borrowed
use std::borrow::Cow;
fn process(input: Cow<'_, str>) { /* ... */ }
```

### Async Rust:

```rust
// AVOID — blocking in async context
async fn handle_request() {
    let data = std::fs::read_to_string("file.txt").unwrap(); // BLOCKS
}

// BETTER — use async I/O or spawn_blocking
async fn handle_request() {
    let data = tokio::fs::read_to_string("file.txt").await.unwrap();
    // OR for CPU-heavy work:
    let result = tokio::task::spawn_blocking(|| heavy_computation()).await.unwrap();
}
```
