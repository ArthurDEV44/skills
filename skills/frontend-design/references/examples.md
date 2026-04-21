# Gold Standard Examples — Few-Shot References

Before writing code in Phase 3, review the example closest to what you're building.
These annotated snippets demonstrate the key techniques from this skill in practice.
Adapt them to the project's stack — do NOT copy verbatim.

## Example 1: Typography-Driven Hero

A hero section where typography IS the design. No image needed.

```css
.hero {
  min-height: 85vh;
  display: grid;
  place-content: center;
  padding: clamp(64px, 10vw, 128px) var(--space-8);
}

.hero-label {
  font-family: var(--font-body);
  font-size: var(--text-xs);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;                /* Wide tracking on labels */
  color: var(--color-text-muted);
  margin-bottom: var(--space-6);
}

.hero-headline {
  font-family: var(--font-display);
  font-size: var(--text-display);         /* clamp(3.5rem, 2rem + 5vw, 8rem) */
  font-weight: 900;
  line-height: 0.95;                      /* Tight — creates visual mass */
  letter-spacing: -0.04em;               /* Negative tracking at display sizes */
  color: var(--color-text);
  max-width: 15ch;                        /* Forces dramatic line breaks */
  margin-bottom: var(--space-8);
}

.hero-body {
  font-family: var(--font-body);
  font-size: var(--text-base);
  font-weight: 300;                       /* Light body vs 900 headline = extreme contrast */
  line-height: 1.6;
  color: var(--color-text-muted);
  max-width: 45ch;                        /* Readable line length */
}
```

**What makes this work:**
- 900 vs 300 weight contrast (not timid 600 vs 400)
- Display text at 8rem creates "typography as hero" effect
- `max-width: 15ch` on headline forces interesting line breaks
- `letter-spacing: -0.04em` at display size — AI never does this
- Label is uppercase with wide tracking — opposite of headline treatment

## Example 2: Asymmetric Feature Grid

Cards that encode importance through size variation.

```css
.features {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--space-4);
  padding-block: clamp(64px, 10vw, 128px);
}

/* Primary feature — large, spans 7 columns */
.feature:nth-child(1) {
  grid-column: 1 / 8;
  padding: var(--space-12);
  background: var(--color-bg-subtle);
  border-left: 3px solid var(--color-accent);  /* Single-border card, not full frame */
}

/* Secondary features — smaller, staggered */
.feature:nth-child(2) {
  grid-column: 8 / 13;
  padding: var(--space-8);
  margin-top: var(--space-16);            /* Offset creates visual rhythm */
}

.feature:nth-child(3) {
  grid-column: 2 / 7;                    /* Inset from left — breaks the grid edge */
  padding: var(--space-8);
}

.feature:nth-child(4) {
  grid-column: 7 / 12;
  padding: var(--space-8);
  margin-top: calc(-1 * var(--space-8)); /* Negative margin creates overlap tension */
}

/* Responsive: stack on mobile but preserve spacing variation */
@media (max-width: 768px) {
  .features { grid-template-columns: 1fr; }
  .feature:nth-child(n) {
    grid-column: 1;
    margin-top: 0;
  }
  .feature:nth-child(1) { margin-bottom: var(--space-8); }
}
```

**What makes this work:**
- Unequal column spans — 7/5 split, not 6/6
- `margin-top` offset on second card creates stagger, not rigid alignment
- Single left-border card, not the standard rounded-everything-with-shadow
- Grid inset on third card (`2 / 7`) breaks the left edge expectation
- Mobile stacks to single column but with varied spacing

## Example 3: Distinctive Navigation

A floating pill nav — not the default logo-links-CTA horizontal bar.

```css
.nav {
  position: fixed;
  top: var(--space-6);
  left: 50%;
  transform: translateX(-50%);
  z-index: 50;
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: 6px;
  background: var(--color-bg-subtle);
  border: 1px solid var(--color-border);
  border-radius: 999px;                  /* Pill shape — a deliberate choice */
}

.nav-link {
  padding: 8px 16px;
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--color-text-muted);
  border-radius: 999px;
  min-height: 44px;                       /* WCAG touch target */
  display: inline-flex;
  align-items: center;
  transition: color 150ms ease;           /* Only functional transition */
}

.nav-link:hover {
  color: var(--color-text);
}

.nav-link[aria-current="page"] {
  background: var(--color-bg);
  color: var(--color-text);
  font-weight: 600;
}
```

**What makes this work:**
- Breaks the "logo left, links center, CTA right" default
- Pill shape is a confident choice (not `rounded-md` on everything)
- Active state uses background shift, not underline or color
- 44px min-height satisfies WCAG 2.5.8
- Only hover transition is color — no scale, no translateY

## Example 4: Grain Texture Application

Background grain that signals "designed by a human."

```css
/* Define once — reuse via .grain class */
.grain {
  position: relative;
  isolation: isolate;
}

.grain::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity: 0.04;                          /* 3-5% — felt, not seen */
  mix-blend-mode: multiply;              /* multiply for light bg, soft-light for dark */
  pointer-events: none;
  z-index: 1;
}

/* Apply selectively — hero and key sections, not data areas */
.hero { @extend .grain; }
.cta-section { @extend .grain; }
/* NOT on: tables, code blocks, form areas */
```

**What makes this work:**
- SVG noise filter — zero network requests, pure CSS
- 4% opacity is the sweet spot: subconsciously felt, not consciously visible
- `mix-blend-mode: multiply` on light backgrounds, `soft-light` on dark
- Applied selectively, not globally — data areas stay clean
- `pointer-events: none` prevents the overlay from blocking interactions

## Example 5: Editorial Section with Tonal Shift

A content section that uses subtle background gradient instead of hard color breaks.

```css
.editorial-section {
  padding-block: clamp(96px, 12vw, 192px);  /* Generous — double what feels "enough" */
  background: linear-gradient(
    180deg,
    oklch(97% 0.01 80) 0%,               /* Warm cream at top */
    oklch(97% 0.005 250) 100%            /* Very slight cool shift at bottom */
  );
}

.editorial-section .container {
  max-width: 640px;                       /* Narrow for text-heavy content */
  margin-inline: auto;
}

.editorial-section h2 {
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin-bottom: var(--space-12);         /* Generous gap before body */
}

.editorial-section p {
  font-family: var(--font-body);
  font-weight: 300;
  line-height: 1.6;
  max-width: 65ch;
}

.editorial-section p + p {
  margin-top: 2em;                        /* WCAG 1.4.12 paragraph spacing */
}
```

**What makes this work:**
- Tonal gradient (warm→cool) creates depth without decorative gradient
- Maximum chroma difference of 0.005 — barely perceptible, like natural light
- 640px container for text — narrower than the image sections around it
- Padding at 96-192px — AI would use 48px. Double it.
- `max-width: 65ch` on paragraphs for optimal reading length
