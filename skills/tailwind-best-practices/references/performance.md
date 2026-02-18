# Performance Optimization

## v4 Performance Improvements

Tailwind CSS v4 is a ground-up Rust rewrite:
- **Full builds**: Up to 3.5x faster than v3
- **Incremental builds**: Up to 8x faster (HMR ~5ms)
- **Smaller output**: Only generates CSS for classes you use (tree-shaking built-in)

## Source Detection

v4 automatically scans your project for class names. Control what gets scanned:

### Add Extra Sources

```css
/* Scan packages in a monorepo */
@source "../node_modules/@my-org/ui/src";
@source "../../packages/shared/components";
```

### Safelist Classes

For dynamic class names that Tailwind can't detect from static analysis:

```css
/* Safelist specific utilities */
@source inline("bg-red-500 bg-green-500 bg-blue-500");
@source inline("underline font-bold");
```

### Exclude Paths

Tailwind ignores `.gitignore` paths by default. No need to exclude `node_modules`.

## Writing Efficient Classes

### Do: Use Complete Class Names

```tsx
// GOOD - Tailwind can detect these
const color = isError ? "text-red-500" : "text-green-500"
```

```tsx
// BAD - Tailwind can't detect dynamic interpolation
const color = `text-${isError ? "red" : "green"}-500`
```

### Do: Use Data Attributes for States

```html
<!-- GOOD - Pure CSS, no JS re-render -->
<div class="opacity-0 data-[visible]:opacity-100 transition-opacity" data-visible={visible || undefined}>
```

```tsx
// AVOID - Conditional classes cause re-renders
<div className={visible ? "opacity-100" : "opacity-0"}>
```

### Do: Prefer Tailwind Utilities Over Inline Styles

```html
<!-- GOOD -->
<div class="w-full max-w-sm mx-auto">

<!-- AVOID -->
<div style={{ width: '100%', maxWidth: '24rem', margin: '0 auto' }}>
```

## className Composition (Next.js)

### With clsx (Recommended)

```bash
npm install clsx
```

```tsx
import { clsx } from 'clsx'

function Button({ variant, className, ...props }: ButtonProps) {
  return (
    <button
      className={clsx(
        'rounded-lg px-4 py-2 font-semibold transition-colors',
        variant === 'primary' && 'bg-blue-500 text-white hover:bg-blue-600',
        variant === 'secondary' && 'bg-gray-200 text-gray-900 hover:bg-gray-300',
        className
      )}
      {...props}
    />
  )
}
```

### With tailwind-merge (Conflict Resolution)

```bash
npm install tailwind-merge
```

```tsx
import { twMerge } from 'tailwind-merge'

// Last class wins for conflicting utilities
twMerge('px-4 py-2', 'px-6') // → 'py-2 px-6'
```

### Combined Helper

```tsx
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

## Minimizing @apply

`@apply` is not recommended for most cases. Prefer:

1. **Utility classes in markup** - Readable and scannable
2. **`@utility` directive** - For truly reusable patterns
3. **Component extraction** - React components as abstraction

Only use `@apply` for:
- Third-party component overrides
- Base styles that can't use utility classes

## Production Build

Tailwind v4 tree-shakes automatically. No purge config needed.

```bash
next build
```

The output CSS only includes classes actually used in your source files.

## Best Practices

1. **Never interpolate class names** - Use complete string literals
2. **Use `cn()` helper** for composable components with `clsx` + `tailwind-merge`
3. **Prefer data attributes** over conditional classes for state changes
4. **Avoid `@apply`** in most cases - extract components instead
5. **Use `@source`** for monorepo packages or dynamic class safelisting
6. **Keep `globals.css` lean** - Only `@import`, `@theme`, and critical `@utility` definitions

[Docs](https://tailwindcss.com/docs/detecting-classes-in-source-files)
