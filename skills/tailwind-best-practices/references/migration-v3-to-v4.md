# Migration: v3 to v4

## Automated Upgrade

```bash
npx @tailwindcss/upgrade
```

This handles most changes automatically. Review the diff carefully.

## Key Breaking Changes

### 1. Import Syntax

```css
/* v3 */
@tailwind base;
@tailwind components;
@tailwind utilities;

/* v4 */
@import "tailwindcss";
```

### 2. Configuration

```js
// v3: tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        brand: '#3b82f6',
      },
    },
  },
};
```

```css
/* v4: globals.css */
@import "tailwindcss";

@theme {
  --color-brand: #3b82f6;
}
```

### 3. Custom Utilities

```css
/* v3 */
@layer utilities {
  .tab-4 {
    tab-size: 4;
  }
}

/* v4 */
@utility tab-4 {
  tab-size: 4;
}
```

### 4. Custom Variants

```js
// v3: plugin JS
addVariant('hocus', ['&:hover', '&:focus'])
```

```css
/* v4: CSS */
@custom-variant hocus (&:hover, &:focus);
```

### 5. Dark Mode

```js
// v3
module.exports = {
  darkMode: 'class',
}
```

```css
/* v4 */
@custom-variant dark (&:where(.dark, .dark *));
```

### 6. theme() Function

```css
/* v3 */
.foo {
  color: theme('colors.blue.500');
}

/* v4 */
.foo {
  color: var(--color-blue-500);
}
```

### 7. Gradient Rename

```html
<!-- v3 -->
<div class="bg-gradient-to-r">

<!-- v4 -->
<div class="bg-linear-to-r">
```

### 8. Shadow/Ring Defaults

v4 changed defaults:
- `ring` was 3px, now is 1px. Use `ring-3` for old behavior
- `shadow` now uses `inset-shadow` for inner shadows
- `ring-*` color now defaults to `currentColor`, not blue-500

To restore v3 behavior:

```css
@theme {
  --default-ring-width: 3px;
  --default-ring-color: var(--color-blue-500);
}
```

### 9. Container Utility

```js
// v3
module.exports = {
  theme: {
    container: {
      center: true,
      padding: '2rem',
    },
  },
}
```

```css
/* v4 */
@utility container {
  margin-inline: auto;
  padding-inline: 2rem;
}
```

### 10. @apply in Components

```css
/* v3 */
@layer components {
  .btn {
    @apply px-4 py-2 rounded-lg font-semibold;
  }
}

/* v4 - prefer @utility instead */
@utility btn {
  border-radius: var(--radius-lg);
  padding-inline: 1rem;
  padding-block: 0.5rem;
  font-weight: 600;
}
```

## Incremental Migration

Use `@config` and `@plugin` alongside `@theme` during migration:

```css
@import "tailwindcss";

/* Keep using v3 config while migrating */
@config "../../tailwind.config.js";

/* Incrementally move tokens to @theme */
@theme {
  --color-brand: oklch(0.6 0.2 260);
}

/* Keep using v3 plugins */
@plugin "@tailwindcss/typography";
@plugin "@tailwindcss/forms";
```

CSS definitions take precedence over JS config when both exist.

## Removed Features

| v3 Feature | v4 Replacement |
|-----------|----------------|
| `tailwind.config.js` | `@theme {}` in CSS |
| `theme()` function | `var()` CSS function |
| `@tailwind` directives | `@import "tailwindcss"` |
| `@layer utilities` | `@utility` directive |
| `addVariant()` JS | `@custom-variant` CSS |
| `corePlugins` | Always enabled |
| `safelist` | `@source inline("...")` |
| `separator` option | Not configurable |
| `important` option | Use `@import "tailwindcss" important` |
| Container plugin config | `@utility container {}` |

[Docs](https://tailwindcss.com/docs/upgrade-guide)
