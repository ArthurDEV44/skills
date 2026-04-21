# Web Performance

Performance is not optimization. It is respect — for the user's time, device, and bandwidth.

## Font Loading Strategy

Web fonts are the most common source of invisible text and layout shift. Load them deliberately.

### Google Fonts Preconnect Pattern
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- Max 3 weights. font-display=swap is mandatory. -->
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@300;500;800&display=swap" rel="stylesheet">
```

### Preventing CLS from Web Fonts
Use `size-adjust` on your fallback to match metrics with the web font:
```css
@font-face {
  font-family: "Instrument Sans Fallback";
  src: local("Arial");
  size-adjust: 105%;
  ascent-override: 95%;
  descent-override: 22%;
  line-gap-override: 0%;
}
body { font-family: "Instrument Sans", "Instrument Sans Fallback", sans-serif; }
```

For non-Latin scripts, subset font files with `unicode-range` to skip unused glyphs.

## Largest Contentful Paint (LCP)

The hero image or headline is almost always the LCP element. Treat it as priority cargo.

```html
<!-- Hero image: eager load + high fetch priority -->
<img src="hero.webp" alt="..." loading="eager" fetchpriority="high"
     width="1200" height="800">

<!-- Preload critical hero image in <head> -->
<link rel="preload" as="image" href="hero.webp" type="image/webp">
```

Avoid render-blocking CSS by splitting critical from deferred:
```html
<head>
  <style>/* inlined critical styles — above-the-fold only */</style>
  <link rel="preload" href="full.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="full.css"></noscript>
</head>
```

## Cumulative Layout Shift (CLS)

Every element that moves after initial paint is a CLS violation. Lock dimensions early.

```html
<img src="photo.webp" width="800" height="600" alt="...">
<video width="1280" height="720" poster="thumb.webp"></video>
<iframe style="aspect-ratio: 16/9; width: 100%;" src="..."></iframe>
```

```css
.ad-slot { min-height: 250px; }
.skeleton { aspect-ratio: 3 / 2; background: var(--color-bg-muted); }
```

The `size-adjust` fallback font technique (see Font Loading above) is the single most
effective CLS fix for text-heavy pages.

## Image Optimization

```html
<!-- Lazy load everything below the fold -->
<img src="photo.webp" alt="..." loading="lazy" decoding="async"
     width="600" height="400">

<!-- Modern format with fallback -->
<picture>
  <source srcset="photo.avif" type="image/avif">
  <source srcset="photo.webp" type="image/webp">
  <img src="photo.jpg" alt="..." width="800" height="600" loading="lazy">
</picture>

<!-- Responsive images — let the browser choose -->
<img srcset="photo-400.webp 400w, photo-800.webp 800w, photo-1200.webp 1200w"
     sizes="(max-width: 600px) 100vw, (max-width: 1024px) 50vw, 800px"
     src="photo-800.webp" alt="..." loading="lazy" width="800" height="600">
```

Format priority: AVIF > WebP > JPEG. Never serve PNG for photos.

## Critical CSS

Inline only what the user sees before scrolling. Defer everything else.

```html
<head>
  <style>/* reset, nav, hero, typography — above-the-fold only */</style>
  <link rel="preload" href="styles.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="styles.css"></noscript>
</head>
```

Tools like `critical` (npm) automate extraction. For SSR frameworks (Next.js, Nuxt),
the bundler handles this — do not inline manually.

## Resource Hints

Place in `<head>` in priority order:

```html
<!-- 1. preconnect — early connections to critical origins -->
<link rel="preconnect" href="https://cdn.example.com" crossorigin>

<!-- 2. dns-prefetch — fallback for browsers without preconnect -->
<link rel="dns-prefetch" href="https://cdn.example.com">

<!-- 3. preload — critical resources the parser won't discover early -->
<link rel="preload" href="/fonts/instrument-sans.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="hero.webp" as="image" type="image/webp">

<!-- 4. prefetch — low-priority fetch for likely next navigations -->
<link rel="prefetch" href="/about">
```

Use `preconnect` for origins you will definitely hit. Use `dns-prefetch` for origins
you might hit. Use `preload` sparingly — overuse competes for bandwidth.

## What NOT To Do

- Don't load more than 3 font weights — each weight is a separate network request
- Don't skip `width`/`height` on images — the #1 cause of layout shift
- Don't use `loading="lazy"` on the hero image — it delays LCP
- Don't preload everything — limit to 2-3 truly critical resources
- Don't self-host Google Fonts without subsetting and compressing them
- Don't use `@import` in CSS for fonts — it chains blocking requests
- Don't serve uncompressed images — always use a build pipeline or image CDN
- Don't inline more than 14KB of critical CSS — beyond that you delay the first packet
