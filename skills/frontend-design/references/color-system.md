# Color System

Color is a decision, not a decoration. Every color in your palette must answer the question:
"What does this communicate?" If it communicates nothing, remove it.

## OKLCH — The Only Color Space You Should Use

**NEVER use hex or HSL for color definitions.** Use `oklch()` exclusively.

Why: OKLCH is perceptually uniform — a 10% lightness change looks like 10% everywhere.
Hex and HSL produce muddy mid-tones when you generate scales. OKLCH scales look consistent
without manual tuning.

```css
/* HSL problem: these don't look evenly spaced */
--blue-300: hsl(210, 70%, 70%);
--blue-500: hsl(210, 70%, 50%);
--blue-700: hsl(210, 70%, 30%);

/* OKLCH solution: these are perceptually even */
--blue-300: oklch(75% 0.12 240);
--blue-500: oklch(55% 0.18 240);
--blue-700: oklch(35% 0.14 240);
```

### OKLCH Syntax
```
oklch(L% C H / A)
  L = Lightness (0% black → 100% white)
  C = Chroma (0 grey → 0.4 vivid saturated)
  H = Hue (0-360 degrees, like a color wheel)
  A = Alpha (optional, 0 transparent → 1 opaque)
```

### Hue Reference
- 0-30: Red/pink
- 30-70: Orange/yellow
- 70-150: Yellow-green/green
- 150-210: Cyan/teal
- 210-270: Blue
- 270-330: Purple/magenta
- 330-360: Pink/red

## Banned Color Patterns

### Layer 1 Bans (Instant AI Tells)
- Purple-to-blue gradients (the #1 AI design signature)
- Indigo as primary/accent color (`indigo-500`, `indigo-600`, Linear's `#5E6AD2`)
- Neon glow effects on any element
- Multi-color gradient backgrounds
- Glassmorphism (semi-transparent blurred panels)
- Pure black (`oklch(0% 0 0)`) backgrounds — use tinted near-black
- Pure white (`oklch(100% 0 0)`) backgrounds — use warm off-white
- Pure grey (chroma 0) anywhere — always add slight chroma

### Layer 2 Bans (Second-Ring AI Defaults)
- Teal/cyan as primary accent (AI's second choice after purple)
- Rainbow icon sets (different colored icons for each feature)
- Gradient text on headings
- Tailwind palette names used directly (`bg-slate-50`, `text-gray-900`)
  — always define your own CSS custom properties

## Palette Construction with OKLCH

### The 60-30-10 Rule
- **60% — Base neutral:** The background, the canvas. Warm off-white or tinted near-black.
- **30% — Supporting tone:** Cards, secondary surfaces, borders. A shade of the base.
- **10% — Accent:** The single color that draws attention. Used ONLY on primary actions,
  important indicators, and key highlights. Never on decoration.

### Building Warm Neutrals (OKLCH)

```css
/* ─── Warm Light Palette (cream/stone direction) ─── */
:root {
  --bg-primary:   oklch(97% 0.01 80);     /* Warm off-white (like old paper) */
  --bg-secondary: oklch(95% 0.012 80);    /* Light stone */
  --bg-tertiary:  oklch(92% 0.015 80);    /* Medium stone */
  --border:       oklch(85% 0.01 80);     /* Warm border */
  --text-muted:   oklch(55% 0.02 80);     /* Warm grey text */
  --text-primary: oklch(15% 0.015 80);    /* Warm near-black */
}

/* ─── Cool Dark Palette (slate/ink direction) ─── */
:root {
  --bg-primary:   oklch(10% 0.015 250);   /* Cool near-black */
  --bg-secondary: oklch(14% 0.018 250);   /* Dark slate */
  --bg-tertiary:  oklch(18% 0.02 250);    /* Medium dark */
  --border:       oklch(25% 0.015 250);   /* Subtle border */
  --text-muted:   oklch(55% 0.01 250);    /* Cool grey text */
  --text-primary: oklch(92% 0.005 250);   /* Soft white */
}

/* ─── Warm Dark Palette (charcoal direction) ─── */
:root {
  --bg-primary:   oklch(10% 0.01 60);     /* Warm near-black */
  --bg-secondary: oklch(14% 0.015 60);    /* Dark charcoal */
  --bg-tertiary:  oklch(18% 0.018 60);    /* Medium dark */
  --border:       oklch(25% 0.012 60);    /* Warm border */
  --text-muted:   oklch(55% 0.015 60);    /* Warm grey text */
  --text-primary: oklch(93% 0.01 60);     /* Warm white */
}
```

### Real-World Palette References (Converted to OKLCH)

**Ottografie (Exo Ape) — Award-winning 2-color palette:**
```css
--bg: oklch(95.5% 0.01 80);    /* #F5F0EB — warm cream */
--text: oklch(13% 0.01 60);    /* #181716 — warm near-black */
```

**Aether 1 — Navy + Electric Blue:**
```css
--bg: oklch(12% 0.04 240);     /* #001a35 — deep navy */
--accent: oklch(55% 0.15 240); /* #2274ca — electric blue */
```

### Accent Color Selection (OKLCH)

Choose accents that feel organic and intentional, not synthetic:

**Strong accents:**
```css
--accent-forest:     oklch(42% 0.1 160);    /* Forest green — grounded */
--accent-terracotta: oklch(52% 0.15 30);    /* Rust/terracotta — warm, human */
--accent-navy:       oklch(30% 0.08 250);   /* Deep navy — authoritative */
--accent-burgundy:   oklch(35% 0.1 15);     /* Burgundy — refined */
```

**Subtle accents:**
```css
--accent-sage:       oklch(60% 0.08 145);   /* Sage — natural, calm */
--accent-dusty-rose: oklch(62% 0.07 15);    /* Dusty rose — warm, gentle */
--accent-ochre:      oklch(62% 0.12 85);    /* Ochre/mustard — creative */
--accent-slate-blue: oklch(52% 0.08 250);   /* Slate blue — measured */
```

**High-contrast accents:**
```css
--accent-vermillion: oklch(55% 0.2 25);     /* Vermillion — urgent (sparingly) */
--accent-emerald:    oklch(55% 0.15 155);   /* Emerald — growth, success */
```

### Generating a Full Scale from One OKLCH Color

```css
/* Start with your accent hue, then vary L and C */
:root {
  --accent-50:  oklch(95% 0.03 160);  /* Very light tint */
  --accent-100: oklch(90% 0.05 160);
  --accent-200: oklch(80% 0.08 160);
  --accent-300: oklch(70% 0.1 160);
  --accent-400: oklch(60% 0.12 160);
  --accent-500: oklch(50% 0.14 160);  /* Base accent */
  --accent-600: oklch(42% 0.12 160);
  --accent-700: oklch(35% 0.1 160);
  --accent-800: oklch(28% 0.08 160);
  --accent-900: oklch(20% 0.06 160);
}
```

### The One-Accent Discipline

You get ONE accent color. Not two. Not three. One.

This accent is reserved for:
- Primary call-to-action buttons
- Active/selected states
- Important notifications or highlights
- Links (optionally)

It is NOT used for:
- Section backgrounds
- Decorative elements
- Icons (unless they indicate status)
- Multiple CTAs at the same visual weight

## Semantic Token Architecture

```css
/* Layer 1: Primitive OKLCH values */
:root {
  --stone-50:  oklch(97% 0.01 80);
  --stone-100: oklch(95% 0.012 80);
  --stone-200: oklch(92% 0.015 80);
  --stone-300: oklch(85% 0.01 80);
  --stone-500: oklch(55% 0.02 80);
  --stone-900: oklch(15% 0.015 80);
  --green-600: oklch(42% 0.1 160);
  --green-700: oklch(35% 0.1 160);
}

/* Layer 2: Semantic tokens */
:root {
  --color-bg:           var(--stone-50);
  --color-bg-subtle:    var(--stone-100);
  --color-bg-muted:     var(--stone-200);
  --color-border:       var(--stone-300);
  --color-text:         var(--stone-900);
  --color-text-muted:   var(--stone-500);
  --color-accent:       var(--green-600);
  --color-accent-hover: var(--green-700);
}

/* Layer 3: Component tokens */
.card {
  background: var(--color-bg-subtle);
  border: 1px solid var(--color-border);
}
.button-primary {
  background: var(--color-accent);
  color: oklch(100% 0 0);
}
```

## Dark Mode

If implementing dark mode, do NOT simply invert colors. Design it as a separate OKLCH palette:

- Near-black background with a slight hue tint (not `oklch(8% 0 0)` but `oklch(10% 0.015 250)`)
- Reduce text lightness slightly (not `oklch(100% 0 0)` — aim for `oklch(92% 0.005 ...)`)
- Accent colors may need higher lightness to maintain contrast
- Borders become more visible (higher lightness) as they're primary separators in dark mode
- Shadows become nearly invisible — use borders and background shifts instead

## What NOT To Do

- Don't use more than 2 chromatic colors beyond your neutrals (1 accent + 1 semantic like error)
- Don't apply color to "make it interesting" — if the layout is boring, fix the layout
- Don't use gradient backgrounds as a substitute for good composition
- Don't use different accent colors for different sections of the same page
- Don't use Tailwind palette names (`bg-slate-100`) — define custom OKLCH properties
- Don't use hex values — always OKLCH
- Don't use pure grey (chroma 0) — always add a slight warmth or coolness
- Don't use alternating colored backgrounds (white → grey → white → grey) unless there's
  a genuine content-hierarchy reason
