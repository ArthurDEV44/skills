# Layout Patterns

Layout is composition. Not stacking. Not filling. Composition — the deliberate arrangement
of elements to create visual rhythm, hierarchy, and movement.

## The 8px Spacing Grid

All spacing values must be multiples of 8px: 8, 16, 24, 32, 48, 64, 96, 128, 192, 256.

The 4px half-unit is allowed ONLY for fine-grain component internals (icon-to-text gap,
input padding). Never for section spacing.

```css
:root {
  --space-1:  0.25rem;  /*  4px — micro only */
  --space-2:  0.5rem;   /*  8px */
  --space-3:  0.75rem;  /* 12px — avoid, prefer 8 or 16 */
  --space-4:  1rem;     /* 16px */
  --space-6:  1.5rem;   /* 24px */
  --space-8:  2rem;     /* 32px */
  --space-12: 3rem;     /* 48px */
  --space-16: 4rem;     /* 64px */
  --space-24: 6rem;     /* 96px */
  --space-32: 8rem;     /* 128px */
  --space-48: 12rem;    /* 192px */
  --space-64: 16rem;    /* 256px */
}
```

### Section Spacing — BE GENEROUS
AI tends to cram sections together. Minimum section padding: `--space-24` (96px) on desktop.
More is usually better. Use `clamp()` for fluid section spacing:

```css
.section {
  padding-block: clamp(64px, 10vw, 128px);
}
```

**Double the vertical padding.** AI is afraid of white space; human designers embrace it.

### Component Spacing — Proximity = Relationship
Within a section, group related elements tightly and separate groups with more space.
This creates visual clustering that communicates hierarchy.

```
[Section padding: 96px]

  LABEL              ← tight: 8px below
  HEADLINE           ← moderate: 24px below
  BODY TEXT           ← generous: 48px below
  BUTTON

[Section padding: 96px]
```

## Negative Space as Design Element

Whitespace is not "empty." It is the most powerful compositional tool you have.

**Rules:**
- If something feels "too empty," resist the urge to fill it. The emptiness IS the design.
- Generous whitespace signals confidence and quality. Cramped layouts signal insecurity.
- Use whitespace to FRAME important content, not just to separate sections.
- Asymmetric whitespace (more on one side than the other) creates tension and interest.
- AI always fills space; professional designers deliberately leave some areas sparse.

## Editorial Layout Patterns

### BANNED Default Layout: The SaaS Template
This sequence is used by AI 90% of the time. Do NOT use it unless explicitly requested:

1. Hero (centered headline + 2 CTAs + screenshot)
2. Logo bar ("Trusted by...")
3. Three-column features with icons
4. Big feature with image left, text right
5. Testimonials
6. Pricing table
7. FAQ accordion
8. CTA banner
9. Footer

Design the layout FROM the content, not from this template.

### Use These Instead

**Asymmetric split:**
```
┌──────────────────────────────────────────┐
│                                          │
│  Small label              Large image    │
│  LARGE HEADLINE           that bleeds    │
│  Body text, max 45ch      to the edge    │
│  Button                                  │
│                                          │
└──────────────────────────────────────────┘
```
The text occupies ~40% width. The image occupies ~60%. Not centered. Not equal.

```css
.hero {
  display: grid;
  grid-template-columns: 2fr 3fr;
  align-items: center;
  min-height: 85vh;
}
.hero-image {
  margin-right: calc(-1 * var(--space-8)); /* Bleed to edge */
}
```

**Offset/staggered grid:**
```
┌──────────────────────────────────────────┐
│                                          │
│        HEADLINE spanning full width      │
│                                          │
│  Card 1 (large)    │ Card 2 (small)      │
│                    │                     │
│                    │ Card 3 (small)      │
│                    │                     │
└──────────────────────────────────────────┘
```
Unequal card sizes. The large card is the primary message.

```css
.features {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--space-6);
}
.feature:nth-child(1) { grid-column: 1 / 7; }
.feature:nth-child(2) { grid-column: 7 / 13; margin-top: var(--space-16); }
.feature:nth-child(3) { grid-column: 2 / 8; }
.feature:nth-child(4) { grid-column: 8 / 12; margin-top: calc(-1 * var(--space-8)); }
```

**Typography-driven hero (from Awwwards winners):**
```
┌──────────────────────────────────────────┐
│                                          │
│                                          │
│   ENORMOUS HEADLINE                      │
│   THAT FILLS 70% OF                      │
│   THE VIEWPORT HEIGHT                    │
│                                          │
│          Small body text below,          │
│          narrow column (45ch)            │
│                                          │
└──────────────────────────────────────────┘
```
No image needed. The typography IS the hero. Display size at 8-12rem, tight line-height (0.95).

```css
.hero-headline {
  font-size: var(--text-display); /* clamp(3.5rem, 2rem + 5vw, 8rem) */
  line-height: 0.95;
  letter-spacing: -0.04em;
  font-weight: 900;
  max-width: 15ch; /* Force dramatic line breaks */
}
```

**Full-bleed image with overlapping text:**
```
┌──────────────────────────────────────────┐
│                                          │
│   [Full-width image]                     │
│                                          │
│       ┌────────────────────────┐         │
│       │  Text block that       │         │
│       │  overlaps the image    │         │
│       │  above, creating depth │         │
│       └────────────────────────┘         │
│                                          │
└──────────────────────────────────────────┘
```

```css
.overlap-section {
  display: grid;
  grid-template-rows: auto auto;
}
.overlap-image {
  grid-row: 1;
  grid-column: 1;
}
.overlap-text {
  grid-row: 1 / 3;
  grid-column: 1;
  align-self: end;
  max-width: 560px;
  margin-left: 10%;
  padding: var(--space-8);
  background: var(--color-bg);
  transform: translateY(30%);
}
```

**Single-column narrative (editorial/magazine):**
```
┌──────────────────────────────────────────┐
│                                          │
│          Small label                     │
│                                          │
│    VERY LARGE HEADLINE                   │
│    THAT TAKES UP                         │
│    SIGNIFICANT SPACE                     │
│                                          │
│          Body text in a narrow           │
│          column (max 55ch) with          │
│          generous line-height.           │
│                                          │
│          [Image, full content width]     │
│                                          │
└──────────────────────────────────────────┘
```
Centered, generous, editorial. Like a magazine article opening.

**Bento grid with size-encoding-importance:**
```
┌────────────────────────────────────────┐
│ ┌──────────────────┐ ┌──────┐ ┌────┐  │
│ │                  │ │      │ │    │  │
│ │   PRIMARY        │ │ SEC  │ │ SM │  │
│ │   FEATURE        │ │      │ │    │  │
│ │   (3x2 cells)    │ │      │ └────┘  │
│ │                  │ └──────┘ ┌────┐  │
│ │                  │          │ SM │  │
│ └──────────────────┘          └────┘  │
└────────────────────────────────────────┘
```
Base unit: 100px cells with 16px gutters. Size encodes importance.

## CSS Grid for Editorial Layouts

Use CSS Grid with named areas for complex layouts. NOT flexbox for page-level composition.

```css
/* Asymmetric two-column with overlap */
.hero {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 0;
  align-items: center;
  min-height: 80vh;
}

.hero-text {
  grid-column: 1;
  padding-right: var(--space-16);
  z-index: 1;
}

.hero-image {
  grid-column: 2;
  grid-row: 1 / -1;
  margin-right: calc(-1 * var(--space-8));
}
```

## Container Width — VARY It

```css
.container-narrow  { max-width: 640px; margin-inline: auto; }  /* Text-heavy */
.container-default { max-width: 1024px; margin-inline: auto; } /* Mixed */
.container-wide    { max-width: 1280px; margin-inline: auto; } /* Visual */
.container-full    { max-width: none; padding-inline: var(--space-8); } /* Bleed */
```

Use DIFFERENT container widths on the same page. A text section should be narrow.
An image gallery should be wide or full-bleed. This variation creates rhythm.

## Magazine Layout Techniques

### Pull Quote Interruption
Break the reading flow with a large pull quote in the margin or overlapping the text column:
```css
.pull-quote {
  font-size: var(--text-2xl);
  font-weight: 300;
  font-style: italic;
  border-left: 3px solid var(--color-accent);
  padding-left: var(--space-6);
  margin: var(--space-12) calc(-1 * var(--space-16));
  max-width: 35ch;
}
```

### Full-Bleed Image Between Sections
An image that breaks out of the content container signals editorial intent:
```css
.full-bleed-image {
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  height: clamp(300px, 40vw, 600px);
  object-fit: cover;
}
```

### Section Dividers Through Typography
Instead of lines or spacing, use large faded numbers or words:
```css
.section-divider {
  font-size: var(--text-display);
  font-weight: 900;
  color: var(--color-text);
  opacity: 0.06;
  line-height: 0.85;
  user-select: none;
}
```

## Responsive Approach

- Use CSS Grid `auto-fit` / `auto-fill` for grids that adapt without breakpoints
- Use `clamp()` for spacing as well as typography
- Reduce column counts on mobile but maintain the ASYMMETRIC feel
- Stack elements vertically on mobile but keep intentional spacing ratios
- Don't make everything centered on mobile — left-aligned text is more readable
- On mobile, hero headlines should still be large (at least `clamp(2.5rem, 8vw, 5rem)`)

## What NOT To Do

- Don't put equal padding on all four sides of every section
- Don't use the same max-width for text and images
- Don't center everything (centering is a specific choice, not a default)
- Don't make all columns equal width
- Don't stack sections with identical spacing between every one
- Don't use alternating background colors as visual separators
- Don't make all cards the same size in a grid — vary sizes to encode importance
- Don't use a fixed 12-column grid and then make everything span 4 columns each
