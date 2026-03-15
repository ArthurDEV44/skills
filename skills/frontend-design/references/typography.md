# Typography System

Typography is not decoration. It IS the design. A page with perfect typography and nothing else
will always look better than a page with mediocre typography and elaborate visuals.

## The #1 Rule: Extreme Contrast

AI produces timid typography: weight 400 vs 600, size jumps of 1.5x, no tracking variation.
Human designers use extreme contrast that creates genuine hierarchy:

- **Weight:** 200-300 for body, 800-900 for headlines (not 400 vs 600)
- **Size:** 3x+ jump between heading and body (not timid 1.5x)
- **Tracking:** -0.04em at display sizes, 0 at body, +0.08em at small caps/labels
- **Line-height:** 0.95 for hero headlines, 1.1-1.2 for headings, 1.6 for body

This contrast is what makes typography feel designed rather than default.

## Font Selection Rules

### Banned Fonts (AI Default Layer 1)
NEVER use these unless the user explicitly names them:
- Inter
- Roboto
- Open Sans
- Lato
- Montserrat
- Poppins
- System font stack (system-ui, -apple-system, etc.)

### Banned Fonts (AI Default Layer 2)
Also avoid these — AI converges to them after Layer 1 is blocked:
- Space Grotesk
- DM Sans (acceptable as body if paired with a distinctive display face)
- Outfit
- Plus Jakarta Sans
- Sora

### Recommended Approach — Choose by Personality

**Confident and editorial:**
- Instrument Serif (display) + Instrument Sans (body)
- Fraunces (display) + Source Sans 3 (body)
- Playfair Display (display) + IBM Plex Sans (body)

**Technical and precise:**
- JetBrains Mono (display/labels) + Geist Sans (body)
- IBM Plex Mono (display) + IBM Plex Sans (body)
- Fira Code (code/labels) + Atkinson Hyperlegible (body)

**Warm and approachable:**
- Bricolage Grotesque (display + body, use weight axis for contrast)
- Gambetta (display) + General Sans (body)
- Clash Display (display) + Satoshi (body)

**Bold and contemporary:**
- Cabinet Grotesk (display) + General Sans (body)
- Switzer (display + body, variable weight)
- Cal Sans (display) + Geist Sans (body)

**Elegant and refined:**
- Cormorant Garamond (display) + Questrial (body)
- Libre Caslon Display (display) + Libre Franklin (body)
- Bodoni Moda (display) + Work Sans (body)

**Award-winning pairings (from current Awwwards/Codrops leaders 2025-2026):**
- Neue Haas Grotesk + Freight Display — Swiss precision + editorial warmth
- Söhne + Canela — contemporary grotesque + contemporary serif
- GT Alpina + GT America — within-family contrast, extremely clean
- Any variable grotesque + a high-contrast serif (Domaine, Editorial New, Canela)

### Single Variable Font Strategy
Instead of pairing two fonts, use ONE variable font across the entire hierarchy by exploiting
its weight axis at extreme ranges:

```css
:root {
  --font-display-weight: 900;  /* Black for headlines */
  --font-body-weight: 300;     /* Light for body */
  --font-label-weight: 200;    /* Thin for labels/captions */
}

h1 { font-weight: var(--font-display-weight); }
p  { font-weight: var(--font-body-weight); }
.label {
  font-weight: var(--font-label-weight);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
```

Good variable fonts for this: Bricolage Grotesque, Switzer, Cabinet Grotesk, Geist Sans.

## Type Scale

### Recommended Scale — Perfect Fourth (1.333 ratio)

This ratio is currently dominant among award-winning editorial sites:

```css
:root {
  --text-xs:      clamp(0.75rem, 0.7rem + 0.15vw, 0.875rem);
  --text-sm:      clamp(0.875rem, 0.8rem + 0.2vw, 1rem);
  --text-base:    clamp(1rem, 0.93rem + 0.37vw, 1.125rem);
  --text-lg:      clamp(1.125rem, 1rem + 0.5vw, 1.5rem);
  --text-xl:      clamp(1.5rem, 1.3rem + 0.8vw, 2rem);
  --text-2xl:     clamp(2rem, 1.6rem + 1.5vw, 3.5rem);
  --text-3xl:     clamp(2.5rem, 1.8rem + 2.5vw, 5rem);
  --text-display: clamp(3.5rem, 2rem + 5vw, 8rem);
}
```

The `--text-display` size is critical. AI tends to make hero headlines too small. Award-winning
sites use display text at 80-120px on desktop. This creates the "typography IS the hero" effect.

### Line Height Rules

```css
/* Tight for display — creates visual mass */
.hero-headline { line-height: 0.95; }
h1 { line-height: 1.05; }
h2 { line-height: 1.1; }

/* Comfortable for subheadings */
h3, h4 { line-height: 1.3; }

/* Generous for body text — readability */
p, li { line-height: 1.6; }
```

### Letter Spacing Rules (Optical Tracking)

At large sizes, letterforms need negative tracking to read as a visual unit.
At small sizes, tracking should loosen for legibility. AI ignores this entirely.

```css
/* Display: tight tracking creates visual density */
.hero-headline { letter-spacing: -0.04em; }
h1 { letter-spacing: -0.03em; }
h2 { letter-spacing: -0.02em; }
h3 { letter-spacing: -0.01em; }

/* Body: normal */
p { letter-spacing: 0; }

/* Small caps/labels: wide tracking for legibility */
.label, .caption {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 500;
}
```

### Hierarchy Through Contrast

Create hierarchy through EXTREME contrast, not just modest size differences:

- **Weight contrast:** 900 heading vs 300 body (not 600 vs 400)
- **Size contrast:** Display at 8rem vs body at 1rem (8:1 ratio, not 2:1)
- **Case contrast:** Uppercase small labels vs sentence-case body
- **Spacing contrast:** Tight tracking on headlines vs normal on body
- **Color contrast:** Full-darkness headline vs slightly muted body text

A heading that is only slightly larger than body text creates NO hierarchy. Be bold with jumps.

## Editorial Typography Techniques

### Pull Quotes
```css
.pull-quote {
  font-family: var(--font-display);
  font-size: var(--text-2xl);
  font-weight: 300;
  font-style: italic;
  border-left: 3px solid var(--color-accent);
  padding-left: var(--space-6);
  margin-block: var(--space-12);
  color: var(--color-accent);
  max-width: 35ch;
}
```

### Drop Caps
```css
.article-body > p:first-of-type::first-letter {
  font-family: var(--font-display);
  font-size: 4.5em;
  font-weight: 900;
  float: left;
  line-height: 0.8;
  padding-right: 0.15em;
  color: var(--color-accent);
}
```

### Large Section Numbers
```css
.section-number {
  font-family: var(--font-display);
  font-size: var(--text-display);
  font-weight: 900;
  line-height: 0.85;
  color: var(--color-text-muted);
  opacity: 0.15;
}
```

## Practical Implementation

### Google Fonts Import Pattern
```html
<!-- Preconnect for performance -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<!-- Load only the weights you actually use — typically 3 max -->
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@300;500;800&display=swap" rel="stylesheet">
```

### Tailwind v4 Custom Font Setup
```css
@theme {
  --font-display: "Instrument Serif", serif;
  --font-body: "Instrument Sans", sans-serif;
}
```

### Max Width for Readable Text
```css
/* Body text: 65 characters per line max */
.prose { max-width: 65ch; }

/* Headings can be wider but constrained for interesting line breaks */
h1 { max-width: 20ch; }   /* Forces line breaks that create visual shapes */
h2 { max-width: 30ch; }
```

## What NOT To Do

- Don't use weight 400 vs 600 for hierarchy — use 300 vs 900
- Don't set all headings at similar sizes — make the h1 dramatically larger
- Don't leave default letter-spacing — apply negative tracking at display sizes
- Don't use more than 2 font families (1 display + 1 body)
- Don't set line-height above 1.2 for display text — it kills visual density
- Don't leave body text at full container width — always constrain with `max-width: 65ch`
- Don't use the same line-height for headings and body text
- Don't skip `font-display: swap` — always add it for web fonts
