# Responsive Design

## Mobile-First Breakpoints

All breakpoints apply **upward** (min-width). Design for mobile first, add styles at larger sizes.

### Default Breakpoints

| Prefix | Min-width | CSS |
|--------|-----------|-----|
| `sm:` | 640px | `@media (width >= 640px)` |
| `md:` | 768px | `@media (width >= 768px)` |
| `lg:` | 1024px | `@media (width >= 1024px)` |
| `xl:` | 1280px | `@media (width >= 1280px)` |
| `2xl:` | 1536px | `@media (width >= 1536px)` |

### Custom Breakpoints

```css
@theme {
  --breakpoint-xs: 475px;
  --breakpoint-3xl: 1920px;
}
```

### Responsive Grid Pattern

```html
<div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
  <div class="p-4 md:p-6 lg:p-8">
    <h2 class="text-lg md:text-xl lg:text-2xl font-bold">Title</h2>
    <p class="text-sm md:text-base hidden sm:block">Description</p>
  </div>
</div>
```

### Breakpoint Ranges

Target a specific range with `max-*:` or combined prefixes:

```html
<!-- Only apply between md and xl -->
<div class="md:max-xl:flex">Flex between md and xl only</div>

<!-- Max-width variant (applies below breakpoint) -->
<div class="max-md:hidden">Hidden on mobile</div>
```

## Container Queries

Style elements based on **parent container size**, not viewport.

### Basic Container Query

```html
<div class="@container">
  <div class="flex flex-col @sm:flex-row @lg:grid @lg:grid-cols-3 gap-4">
    <div class="@sm:w-1/3 @lg:w-auto">Sidebar</div>
    <div class="@sm:w-2/3 @lg:col-span-2">Content</div>
  </div>
</div>
```

### Named Containers

```html
<div class="@container/sidebar">
  <div class="@sm/sidebar:flex @lg/sidebar:grid">
    Responds to sidebar container size
  </div>
</div>
```

### Container Query Breakpoints

| Prefix | Min-width |
|--------|-----------|
| `@xs:` | 320px |
| `@sm:` | 384px |
| `@md:` | 448px |
| `@lg:` | 512px |
| `@xl:` | 576px |
| `@2xl:` | 672px |

### Arbitrary Container Values

```html
<div class="@container">
  <div class="@min-[475px]:flex-row flex flex-col">
    Custom container breakpoint
  </div>
</div>
```

## Responsive Typography

```html
<h1 class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight">
  Responsive Heading
</h1>
```

With custom font sizes in `@theme`:

```css
@theme {
  --text-display: 3.5rem;
  --text-title: 2.25rem;
  --text-body: 1rem;
}
```

## Responsive Spacing

```html
<section class="px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-16">
  <div class="max-w-7xl mx-auto">Content</div>
</section>
```

## Responsive Visibility

```html
<!-- Show only on mobile -->
<nav class="sm:hidden">Mobile nav</nav>

<!-- Show only on desktop -->
<nav class="hidden sm:flex">Desktop nav</nav>

<!-- Show between md and lg -->
<div class="hidden md:block lg:hidden">Tablet only</div>
```

## Best Practices

1. **Design mobile-first** - Start with no prefix, add `sm:`, `md:`, etc.
2. **Use container queries for components** - They respond to parent, not viewport
3. **Avoid too many breakpoints** - 2-3 breakpoints per element is usually enough
4. **Use `max-*:` sparingly** - Mobile-first is cleaner than max-width overrides
5. **Test responsive with real devices** - Viewport width alone doesn't capture everything

[Docs](https://tailwindcss.com/docs/responsive-design)
