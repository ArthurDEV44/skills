# Setup: Tailwind CSS v4 + Next.js

## Installation

```bash
npm install tailwindcss @tailwindcss/postcss
```

## PostCSS Configuration

```js
// postcss.config.mjs
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
```

No `tailwind.config.js` needed for new projects. All configuration goes in CSS.

## Global Stylesheet

```css
/* app/globals.css */
@import "tailwindcss";

@theme {
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --color-primary: oklch(0.6 0.2 260);
  --color-background: oklch(0.98 0 0);
  --color-foreground: oklch(0.15 0 0);
}
```

## Root Layout

```tsx
// app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "My App",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-background text-foreground font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
```

## With Turbopack

`@tailwindcss/postcss` works with both webpack and Turbopack. No extra config needed:

```bash
next dev --turbopack
```

## With Vite (alternative)

If using Vite instead of Next.js:

```bash
npm install tailwindcss @tailwindcss/vite
```

```js
// vite.config.ts
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss()],
});
```

## Custom Fonts (Next.js)

Use `next/font` with CSS variables:

```tsx
// app/layout.tsx
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const playfair = Playfair_Display({ subsets: ["latin"], variable: "--font-playfair" });

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${playfair.variable}`}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
```

```css
/* globals.css */
@import "tailwindcss";

@theme {
  --font-sans: var(--font-inter), ui-sans-serif, system-ui, sans-serif;
  --font-display: var(--font-playfair), serif;
}
```

## Source Detection

Tailwind v4 auto-detects source files. To add extra sources (e.g., monorepo packages):

```css
@source "../node_modules/@my-org/ui/src";
@source "../../packages/shared/components";
```

To safelist specific utilities that aren't in markup:

```css
@source inline("underline");
@source inline("bg-red-500 text-white font-bold");
```

## Common Setup Issues

| Issue | Fix |
|-------|-----|
| PostCSS not processing | Use `@tailwindcss/postcss` not `tailwindcss` as plugin name |
| Styles not appearing | Ensure `@import "tailwindcss"` is in `globals.css` |
| Font not loading | Set CSS variable on `<html>`, reference in `@theme` |
| Monorepo classes missing | Add `@source` for external package paths |

[Docs](https://tailwindcss.com/docs/installation/using-postcss)
