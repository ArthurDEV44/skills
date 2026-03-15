# Texture & Depth — What Separates Human Design from AI Flatness

AI output is perfectly flat. Every surface is a solid color with a solid border. This creates
the "hospital clean" aesthetic that immediately signals machine generation. Human-designed sites
have material depth — subtle grain, tonal shifts, layered surfaces.

## The Grain/Noise Overlay — The Single Highest-Impact Technique

Adding a barely-visible noise texture over backgrounds is the single most effective way to make
a design feel human-crafted. Every premium Framer template uses this. Zero AI-generated sites do.

### SVG Noise Filter (Recommended — Pure CSS, No Image)

```css
/* Define the SVG filter once in your HTML or as an inline SVG */
/*
<svg width="0" height="0" style="position:absolute">
  <filter id="grain">
    <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/>
    <feColorMatrix type="saturate" values="0"/>
  </filter>
</svg>
*/

/* Apply via a pseudo-element on the section/page */
.with-grain {
  position: relative;
  isolation: isolate;
}

.with-grain::after {
  content: '';
  position: absolute;
  inset: 0;
  filter: url(#grain);
  opacity: 0.04;           /* 3-5% — barely visible but subconsciously felt */
  mix-blend-mode: multiply;
  pointer-events: none;
  z-index: 1;
}
```

### CSS-Only Noise (Simpler, Slightly Less Control)

```css
.with-grain::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity: 0.04;
  mix-blend-mode: multiply;
  pointer-events: none;
  z-index: 1;
}
```

### Tailwind v4 Utility Approach

```css
@layer utilities {
  .grain {
    position: relative;
    isolation: isolate;

    &::after {
      content: '';
      position: absolute;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
      opacity: 0.04;
      mix-blend-mode: multiply;
      pointer-events: none;
      z-index: 1;
    }
  }
}
```

### When to Use Grain
- **Always:** Landing pages, hero sections, marketing sites
- **Usually:** Portfolio sites, creative sites, editorial layouts
- **Sometimes:** SaaS dashboards (on sidebar or header only, not on data areas)
- **Never:** Dense data tables, code editors, technical documentation

### Grain Opacity Guide
- Light backgrounds: `opacity: 0.03-0.05` with `mix-blend-mode: multiply`
- Dark backgrounds: `opacity: 0.03-0.06` with `mix-blend-mode: soft-light`
- Over images: `opacity: 0.02-0.03` — just enough to unify the surface

## Tonal Background Shifts

Instead of solid color sections, use very subtle gradients that shift the background tone.
This creates depth without the "decorative gradient" AI tell.

```css
/* Subtle warm-to-cool shift on a light background */
.hero {
  background: linear-gradient(
    180deg,
    oklch(97% 0.01 80) 0%,     /* Very slight warm cream at top */
    oklch(97% 0.005 250) 100%  /* Very slight cool at bottom */
  );
}

/* Subtle radial warmth center on dark background */
.dark-section {
  background: radial-gradient(
    ellipse at 30% 50%,
    oklch(15% 0.02 60) 0%,     /* Slightly warm center */
    oklch(10% 0.01 250) 100%   /* Cool edges */
  );
}
```

**Rules for tonal shifts:**
- Maximum chroma difference: 0.02 (barely perceptible)
- The shift should feel like natural light, not a design choice
- Never use bright, saturated gradients as backgrounds (that's an AI tell)

## Layered Surface Depth

Create depth through layered surfaces with slight color differences, not through shadows.

```css
:root {
  /* Light theme depth layers */
  --surface-0: oklch(97% 0.01 80);   /* Page background — warm */
  --surface-1: oklch(98% 0.008 80);  /* Card on page — slightly lighter */
  --surface-2: oklch(99% 0.005 80);  /* Element on card — lightest */
  --surface-3: oklch(95% 0.012 80);  /* Recessed area — darker */

  /* Dark theme depth layers */
  --surface-0: oklch(12% 0.01 250);  /* Page background — cool dark */
  --surface-1: oklch(15% 0.012 250); /* Card — slightly lighter */
  --surface-2: oklch(18% 0.014 250); /* Element on card */
  --surface-3: oklch(10% 0.008 250); /* Recessed area */
}
```

**The key insight:** In a layered system, shadows become almost unnecessary. The surface color
difference IS the elevation indicator. Use shadows only for truly floating elements (modals,
dropdowns, tooltips).

## Border Treatments

AI uses `border: 1px solid var(--border)` on everything. Human designers use borders
strategically:

```css
/* Light theme: barely-visible warm borders */
.card {
  border: 1px solid oklch(90% 0.01 80 / 0.5);  /* 50% opacity warm line */
}

/* Dark theme: barely-visible light borders */
.card-dark {
  border: 1px solid oklch(100% 0.01 250 / 0.05); /* 5% opacity white line */
}

/* Accent border: only on the side that matters */
.featured-card {
  border-left: 3px solid var(--color-accent);
  border-top: none;
  border-right: none;
  border-bottom: none;
}

/* No border at all — use surface contrast */
.elevated-card {
  border: none;
  background: var(--surface-1); /* Lighter than page background = enough contrast */
}
```

## Image Treatment

AI drops images into layouts raw. Human designers treat images as part of the composition:

```css
/* Desaturated, high-contrast editorial treatment */
.editorial-image {
  filter: grayscale(100%) contrast(1.1);
  mix-blend-mode: multiply;
}

/* Warm tone overlay */
.warm-image {
  filter: sepia(0.1) saturate(0.9);
}

/* Clip path for non-rectangular images */
.angled-image {
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 3rem), 0 100%);
}
```

## What NOT To Do

- Don't use glassmorphism (`backdrop-filter: blur()` on semi-transparent backgrounds) — it's
  a 2021 trend that AI has overlearned
- Don't use multiple gradient backgrounds stacked (the "Aurora" effect) — instant AI tell
- Don't use neon glow effects (`text-shadow` or `box-shadow` with saturated colors)
- Don't add texture to EVERY surface — use it strategically on 2-3 key areas
- Don't use pure black (#000) or pure white (#fff) backgrounds — always tint with OKLCH
