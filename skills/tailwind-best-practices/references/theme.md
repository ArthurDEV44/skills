# @theme: Design Tokens

The `@theme` directive defines design tokens as CSS custom properties. Each namespace auto-generates corresponding utility classes.

## Basic Structure

```css
@import "tailwindcss";

@theme {
  /* Colors: generates bg-*, text-*, border-*, ring-*, etc. */
  --color-brand: oklch(0.72 0.11 178);
  --color-brand-light: oklch(0.85 0.08 178);
  --color-brand-dark: oklch(0.55 0.14 178);
  --color-accent: oklch(0.84 0.18 117.33);
  --color-surface: oklch(0.98 0.005 260);
  --color-muted: oklch(0.55 0 0);

  /* Fonts: generates font-* */
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-display: "Cal Sans", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  /* Spacing multiplier: all spacing utilities use this base */
  --spacing: 0.25rem;

  /* Breakpoints: generates sm:, md:, lg:, etc. */
  --breakpoint-xs: 475px;
  --breakpoint-3xl: 1920px;

  /* Border radius: generates rounded-* */
  --radius-pill: 9999px;
  --radius-card: 0.75rem;

  /* Shadows: generates shadow-* */
  --shadow-soft: 0 2px 8px oklch(0 0 0 / 0.06);
  --shadow-card: 0 4px 12px oklch(0 0 0 / 0.08);

  /* Easing: generates ease-* */
  --ease-fluid: cubic-bezier(0.3, 0, 0, 1);
  --ease-snappy: cubic-bezier(0.2, 0, 0, 1);

  /* Letter spacing: generates tracking-* */
  --tracking-display: -0.04em;
}
```

## Color Recommendations

Use **oklch** for perceptually uniform colors (v4 default):

```css
@theme {
  /* oklch(lightness chroma hue) */
  --color-blue-500: oklch(0.62 0.19 260);
  --color-blue-600: oklch(0.55 0.22 260);

  /* With opacity: oklch(L C H / alpha) */
  --color-overlay: oklch(0 0 0 / 0.5);
}
```

## Overriding Default Theme

By default, `@theme` **merges** with defaults. To replace a namespace entirely:

```css
@theme {
  /* Replaces ALL default colors */
  --color-*: initial;

  /* Then define only your colors */
  --color-primary: oklch(0.6 0.2 260);
  --color-gray-50: oklch(0.98 0 0);
  --color-gray-900: oklch(0.15 0 0);
}
```

## @theme inline

Prevent `@theme` values from generating utility classes (useful for internal tokens):

```css
@theme inline {
  --color-bg: var(--color-surface);
  --sidebar-width: 18rem;
}
```

## @theme reference

Import another file's theme without emitting CSS variables (useful for shared design systems):

```css
@import "tailwindcss";
@import "./shared-theme.css" theme(reference);
```

## Using Theme Values in CSS

Use standard CSS `var()` (not `theme()` which is removed in v4):

```css
.custom-element {
  background: var(--color-brand);
  font-family: var(--font-display);
  border-radius: var(--radius-card);
  box-shadow: var(--shadow-soft);
}
```

## Namespace Reference

| Namespace | Utilities generated | Example token |
|-----------|-------------------|---------------|
| `--color-*` | `bg-*`, `text-*`, `border-*`, `ring-*`, `fill-*`, `stroke-*` | `--color-brand: oklch(...)` |
| `--font-*` | `font-*` | `--font-display: "Cal Sans"` |
| `--spacing` | `p-*`, `m-*`, `gap-*`, `w-*`, `h-*`, `size-*`, `inset-*` | `--spacing: 0.25rem` |
| `--breakpoint-*` | `sm:`, `md:`, etc. | `--breakpoint-3xl: 120rem` |
| `--animate-*` | `animate-*` | `--animate-spin: spin 1s linear infinite` |
| `--ease-*` | `ease-*` | `--ease-fluid: cubic-bezier(...)` |
| `--shadow-*` | `shadow-*` | `--shadow-card: 0 4px 12px ...` |
| `--radius-*` | `rounded-*` | `--radius-pill: 9999px` |
| `--text-*` | `text-*` (font-size) | `--text-display: 3.5rem` |
| `--tracking-*` | `tracking-*` | `--tracking-tight: -0.025em` |
| `--leading-*` | `leading-*` | `--leading-relaxed: 1.75` |
| `--inset-shadow-*` | `inset-shadow-*` | `--inset-shadow-sm: inset 0 1px 1px ...` |
| `--drop-shadow-*` | `drop-shadow-*` | `--drop-shadow-lg: ...` |

[Docs](https://tailwindcss.com/docs/theme)
