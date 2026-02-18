# Dark Mode

## Default: System Preference

By default, `dark:` uses `prefers-color-scheme: dark` media query:

```html
<div class="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
  <h1 class="text-black dark:text-white">Adaptive Heading</h1>
  <p class="text-gray-600 dark:text-gray-400">Body text</p>
  <button class="bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-500">
    Click me
  </button>
</div>
```

Generated CSS:

```css
.dark\:bg-gray-900 {
  @media (prefers-color-scheme: dark) {
    background-color: var(--color-gray-900);
  }
}
```

## Manual Toggle (Class-Based)

Override the dark variant to use a `.dark` class instead of system preference:

```css
/* globals.css */
@import "tailwindcss";

@custom-variant dark (&:where(.dark, .dark *));
```

Then toggle with JavaScript:

```tsx
// ThemeToggle.tsx
'use client'
import { useEffect, useState } from 'react'

export function ThemeToggle() {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <button onClick={() => setDark(!dark)}>
      {dark ? 'Light' : 'Dark'}
    </button>
  )
}
```

```html
<html class="dark">
  <body class="bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100">
    ...
  </body>
</html>
```

## Combining Dark with Other Variants

```html
<!-- Dark + hover -->
<button class="bg-blue-500 hover:bg-blue-600 dark:bg-blue-600 dark:hover:bg-blue-500">
  Submit
</button>

<!-- Dark + responsive -->
<div class="lg:bg-white lg:dark:bg-black">
  Desktop-specific dark background
</div>

<!-- Dark + focus -->
<input class="border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400">
```

## Dark Mode Design Tokens

Define separate tokens for light/dark in `@theme`:

```css
@theme {
  --color-surface: oklch(0.98 0 0);
  --color-surface-dark: oklch(0.15 0 0);
  --color-text: oklch(0.15 0 0);
  --color-text-dark: oklch(0.9 0 0);
}
```

Or use CSS custom properties outside `@theme` for runtime switching:

```css
@import "tailwindcss";

@theme inline {
  --color-bg: var(--bg);
  --color-fg: var(--fg);
}

:root {
  --bg: oklch(0.98 0 0);
  --fg: oklch(0.15 0 0);
}

.dark {
  --bg: oklch(0.12 0 0);
  --fg: oklch(0.95 0 0);
}
```

## Best Practices

1. **Pick one strategy** - System preference OR manual toggle, not both fighting
2. **Use semantic color names** - `bg-surface` not `bg-white dark:bg-gray-900` everywhere
3. **Test both modes** - Easy to forget dark mode contrast and readability
4. **Persist preference** - Use `localStorage` and check on load to avoid flash
5. **Combine with `prefers-color-scheme`** for initial state in manual mode

[Docs](https://tailwindcss.com/docs/dark-mode)
