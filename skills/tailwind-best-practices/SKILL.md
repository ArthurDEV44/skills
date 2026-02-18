---
name: tailwind-best-practices
description: >-
  Tailwind CSS v4 best practices for Next.js App Router. CSS-first config with @theme,
  @utility, @custom-variant, @plugin directives. Use when writing, reviewing, or refactoring
  Tailwind CSS code: (1) Setting up Tailwind v4 with @import "tailwindcss" and PostCSS,
  (2) Design tokens with @theme (colors, fonts, spacing, breakpoints, animations),
  (3) Custom utilities with @utility and variants with @custom-variant, (4) Responsive
  design with mobile-first breakpoints and container queries @container, (5) Dark mode
  with dark: variant and @custom-variant overrides, (6) Animations and keyframes in @theme,
  (7) 3D transforms, perspective, gradients, text-shadow, (8) Migrating v3 to v4 (@tailwind
  to @import, tailwind.config.js to @theme, @layer to @utility), (9) Performance and
  production builds, (10) Next.js integration with App Router, global.css, Turbopack.
  Does NOT cover Tailwind v3 JS config or non-CSS-first patterns.
license: MIT
metadata:
  author: arthur
  version: "1.0.0"
---

# Tailwind CSS v4 Best Practices for Next.js

Comprehensive guide for building production-grade UIs with Tailwind CSS v4's CSS-first configuration in Next.js App Router applications.

## What Changed in v4

Tailwind CSS v4 is a ground-up rewrite. The biggest shift: **everything is CSS-first**.

| v3 | v4 |
|----|-----|
| `tailwind.config.js` | `@theme { }` in CSS |
| `@tailwind base/components/utilities` | `@import "tailwindcss"` |
| `@layer utilities { .foo { } }` | `@utility foo { }` |
| `addVariant()` in plugin JS | `@custom-variant` in CSS |
| `theme.extend.colors` in JS | `--color-*` in `@theme` |
| `darkMode: 'class'` in JS | `@custom-variant dark (...)` in CSS |

## References

| Reference | Impact | When to read |
|-----------|--------|--------------|
| `references/setup-nextjs.md` | CRITICAL | Installing Tailwind v4 in Next.js |
| `references/theme.md` | CRITICAL | Design tokens with @theme |
| `references/utilities-and-variants.md` | HIGH | Custom @utility and @custom-variant |
| `references/responsive.md` | HIGH | Breakpoints, container queries, mobile-first |
| `references/dark-mode.md` | HIGH | Dark mode strategies |
| `references/animations.md` | MEDIUM | Keyframes and transitions in @theme |
| `references/new-utilities.md` | MEDIUM | v4 new utilities (3D, gradients, shadows) |
| `references/migration-v3-to-v4.md` | HIGH | Upgrading from Tailwind v3 |
| `references/performance.md` | MEDIUM | Production optimization |

## Quick Start (Next.js + Tailwind v4)

### 1. Install

```bash
npm install tailwindcss @tailwindcss/postcss
```

### 2. PostCSS config

```js
// postcss.config.mjs
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
```

### 3. Global CSS

```css
/* app/globals.css */
@import "tailwindcss";

@theme {
  --color-primary: oklch(0.6 0.2 260);
  --color-secondary: oklch(0.7 0.15 180);
  --font-sans: "Inter", sans-serif;
  --font-display: "Cal Sans", sans-serif;
}
```

### 4. Import in root layout

```tsx
// app/layout.tsx
import "./globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
```

## Critical Rules

1. **Use `@import "tailwindcss"` not `@tailwind`** - The v3 directives are removed
2. **Define tokens in `@theme { }`** - Not in `tailwind.config.js` (still supported for migration)
3. **Use `@utility` not `@layer utilities`** - New API sorts by property count, works with variants
4. **Use `@custom-variant` for custom variants** - Replaces `addVariant()` from JS plugins
5. **Mobile-first by default** - `sm:` means `@media (width >= 640px)`, design for mobile first
6. **Use oklch for colors** - v4 default palette uses oklch for perceptually uniform color mixing
7. **Arbitrary values with `[]`** - `bg-[#1da1f2]`, `grid-cols-[1fr_2fr]`, `w-[calc(100%-2rem)]`
8. **Compose, don't `@apply`** - Prefer utility classes in markup; use `@apply` sparingly for third-party overrides

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| No styles generated | Missing `@import "tailwindcss"` | Replace `@tailwind` directives with `@import` |
| Custom color not working | Wrong namespace in `@theme` | Use `--color-*` not `--colors-*` |
| `@layer utilities` ignored | v4 uses `@utility` directive | Replace `@layer utilities { .foo {} }` with `@utility foo {}` |
| Dark mode not toggling | Default is system preference | Add `@custom-variant dark (&:where(.dark, .dark *))` |
| Container has no centering | v4 removed container config | Use `@utility container { margin-inline: auto; }` |
| `theme()` function errors | Removed in v4 | Use CSS `var()` instead: `var(--color-primary)` |
| PostCSS not processing | Wrong plugin name | Use `@tailwindcss/postcss` not `tailwindcss` |
| Turbopack not working | Need `@tailwindcss/postcss` | PostCSS plugin works with both webpack and Turbopack |

## @theme Namespace Reference

| Namespace | Generates | Example |
|-----------|-----------|---------|
| `--color-*` | `bg-*`, `text-*`, `border-*`, `ring-*` | `--color-brand: oklch(...)` |
| `--font-*` | `font-*` | `--font-display: "Cal Sans"` |
| `--breakpoint-*` | `sm:`, `md:`, `lg:` etc. | `--breakpoint-3xl: 120rem` |
| `--spacing` | All spacing (`p-*`, `m-*`, `gap-*`, `w-*`, `h-*`) | `--spacing: 0.25rem` |
| `--animate-*` | `animate-*` | `--animate-fade-in: fade-in 0.3s ease-out` |
| `--ease-*` | `ease-*` | `--ease-snappy: cubic-bezier(0.2, 0, 0, 1)` |
| `--radius-*` | `rounded-*` | `--radius-pill: 9999px` |
| `--shadow-*` | `shadow-*` | `--shadow-soft: 0 2px 8px rgb(0 0 0 / 0.08)` |
| `--text-*` | `text-*` (font size) | `--text-display: 3.5rem` |
| `--tracking-*` | `tracking-*` | `--tracking-tight: -0.025em` |
| `--leading-*` | `leading-*` | `--leading-relaxed: 1.75` |

## Documentation

- [Tailwind CSS v4 Docs](https://tailwindcss.com/docs)
- [Upgrade Guide (v3 to v4)](https://tailwindcss.com/docs/upgrade-guide)
- [Theme Configuration](https://tailwindcss.com/docs/theme)
- [Functions and Directives](https://tailwindcss.com/docs/functions-and-directives)
- [Next.js Installation](https://tailwindcss.com/docs/installation/using-postcss)
