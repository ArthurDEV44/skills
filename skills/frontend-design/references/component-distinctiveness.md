# Component Distinctiveness — Breaking Away from shadcn/ui Defaults

The visual sameness of AI output comes from using component library defaults without
customization. shadcn/ui is excellent as a foundation, but its defaults have been memorized
by every LLM, producing visual monoculture. Three CSS variables do more to differentiate
a shadcn app than changing its entire color palette:

1. `--radius` (border-radius)
2. Font family
3. Shadow depth

## Border-Radius Strategy

Border-radius is NOT a global setting. It varies by product tone AND by element type.

### By Product Tone

```css
/* Enterprise / Serious / Fintech */
:root { --radius: 2px; }
/* Signal: precision, trust, authority */

/* Consumer / Friendly / Social */
:root { --radius: 12px; }
/* Signal: approachable, warm, playful */

/* Brutalist / Creative / Agency */
:root { --radius: 0px; }
/* Signal: bold, opinionated, distinctive */

/* Luxury / Fashion / Editorial */
:root { --radius: 0px; }
/* Signal: sharp, refined, precise */
```

### By Element Type (Within a Design)

```css
/* Even within one product, radius should vary */
.button       { border-radius: var(--radius); }
.card         { border-radius: calc(var(--radius) * 1.5); }  /* Slightly larger */
.avatar       { border-radius: 50%; }                        /* Always circular */
.badge, .pill { border-radius: 999px; }                      /* Always pill-shaped */
.input        { border-radius: var(--radius); }
.modal        { border-radius: calc(var(--radius) * 2); }    /* Larger for floating */
.tooltip      { border-radius: calc(var(--radius) * 0.75); } /* Smaller for compact */
```

**NEVER:** Apply the same border-radius to everything. That's the #1 shadcn tell.

## Shadow System

Shadows should be graduated — different depths for different elevations. AI uses either
no shadow or the same shadow everywhere.

### Soft / Diffuse System (Modern SaaS)
```css
:root {
  --shadow-xs: 0 1px 2px oklch(0% 0 0 / 0.04);
  --shadow-sm: 0 2px 8px oklch(0% 0 0 / 0.06);
  --shadow-md: 0 4px 16px oklch(0% 0 0 / 0.08);
  --shadow-lg: 0 12px 40px oklch(0% 0 0 / 0.1);
  --shadow-xl: 0 20px 60px oklch(0% 0 0 / 0.12);
}
```

### Hard Offset System (Brutalist / Creative)
```css
:root {
  --shadow-sm: 2px 2px 0 oklch(20% 0.01 250);
  --shadow-md: 4px 4px 0 oklch(20% 0.01 250);
  --shadow-lg: 6px 6px 0 oklch(20% 0.01 250);
}

/* With hover interaction */
.card {
  box-shadow: var(--shadow-md);
  transition: box-shadow 100ms ease, translate 100ms ease;
}
.card:hover {
  translate: -2px -2px;
  box-shadow: var(--shadow-lg);
}
```

### Tinted Shadow System (Warm / Luxury)
```css
:root {
  /* Shadows tinted with the accent color — feels more designed */
  --shadow-sm: 0 2px 8px oklch(40% 0.05 250 / 0.08);
  --shadow-md: 0 4px 24px oklch(40% 0.05 250 / 0.1);
  --shadow-lg: 0 12px 48px oklch(40% 0.05 250 / 0.12);
}
```

### Shadow Assignment
```css
/* Elements at rest on a surface */
.card, .input        { box-shadow: none; }     /* Use border or bg contrast instead */

/* Elements that float */
.dropdown, .popover  { box-shadow: var(--shadow-md); }

/* Elements that overlay */
.modal, .dialog      { box-shadow: var(--shadow-xl); }

/* Interactive elements on hover (sparingly) */
.product-card:hover  { box-shadow: var(--shadow-sm); }
```

## Button Distinctiveness

The default shadcn button is `bg-primary text-primary-foreground rounded-md h-10 px-4 py-2`.
Every AI-generated site uses this exact pattern. Break it.

### Button Variants by Product Tone

```css
/* Enterprise: Flat, sharp, subtle */
.btn-enterprise {
  background: var(--color-accent);
  color: white;
  border-radius: 2px;
  padding: 0.625rem 1.25rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  font-size: var(--text-xs);
}

/* Consumer: Warm, rounded, inviting */
.btn-consumer {
  background: var(--color-accent);
  color: white;
  border-radius: 12px;
  padding: 0.75rem 1.5rem;
  font-weight: 600;
  font-size: var(--text-sm);
}

/* Creative: Outlined, brutalist */
.btn-creative {
  background: transparent;
  color: var(--color-text);
  border: 2px solid currentColor;
  border-radius: 0;
  padding: 0.75rem 2rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: var(--text-xs);
}

/* Luxury: Minimal, understated */
.btn-luxury {
  background: transparent;
  color: var(--color-text);
  border-bottom: 1px solid currentColor;
  border-radius: 0;
  padding: 0.25rem 0;
  font-weight: 400;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: var(--text-xs);
}

/* Editorial: Text-link style */
.btn-editorial {
  background: transparent;
  color: var(--color-accent);
  border: none;
  padding: 0;
  font-weight: 500;
  text-decoration: underline;
  text-underline-offset: 3px;
  font-size: var(--text-base);
}
```

## Card Distinctiveness

AI cards are always: white background, rounded corners, light shadow, equal padding, equal height.
Human-designed cards vary.

### Strategies to Break Card Monotony

1. **No card at all** — Use spacing and typography to group content, not boxes:
   ```css
   .feature {
     padding-block: var(--space-8);
     border-bottom: 1px solid var(--color-border);
   }
   /* No background, no shadow, no border-radius — just content + separator */
   ```

2. **Asymmetric cards** — Cards in a grid should NOT all be the same size:
   ```css
   .feature-grid {
     display: grid;
     grid-template-columns: repeat(12, 1fr);
     gap: var(--space-4);
   }
   .feature:nth-child(1) { grid-column: 1 / 7; }   /* Large */
   .feature:nth-child(2) { grid-column: 7 / 13; }   /* Large */
   .feature:nth-child(3) { grid-column: 1 / 5; }    /* Small */
   .feature:nth-child(4) { grid-column: 5 / 9; }    /* Small */
   .feature:nth-child(5) { grid-column: 9 / 13; }   /* Small */
   ```

3. **Single-border cards** — Use only a left or top border, not a full frame:
   ```css
   .card {
     border: none;
     border-left: 3px solid var(--color-accent);
     padding-left: var(--space-6);
     background: transparent;
   }
   ```

4. **Elevated vs recessed** — Some cards pop, others sink:
   ```css
   .card-elevated { background: var(--surface-1); }
   .card-recessed { background: var(--surface-3); }
   ```

## Input Fields

Default: white background, grey border, rounded-md. Break it:

```css
/* Clean underline style */
.input-underline {
  border: none;
  border-bottom: 1px solid var(--color-border);
  border-radius: 0;
  background: transparent;
  padding: 0.5rem 0;
  font-size: var(--text-base);
}
.input-underline:focus {
  border-bottom-color: var(--color-accent);
  outline: none;
}

/* Filled style */
.input-filled {
  border: none;
  background: var(--surface-3);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
}
.input-filled:focus {
  background: var(--surface-2);
  outline: 2px solid var(--color-accent);
  outline-offset: -2px;
}
```

## Navigation

AI navigation is always: logo left, links center, CTA right, full-width bar with border-bottom.

### Alternatives

1. **Side-anchored nav** — Navigation as a vertical sidebar strip:
   ```css
   .nav-sidebar {
     position: fixed;
     left: 0;
     top: 0;
     height: 100vh;
     width: 64px;
     display: flex;
     flex-direction: column;
     align-items: center;
     padding-block: var(--space-8);
   }
   ```

2. **Floating pill nav** — Detached, centered, floating:
   ```css
   .nav-pill {
     position: fixed;
     top: var(--space-6);
     left: 50%;
     transform: translateX(-50%);
     background: var(--surface-1);
     border-radius: 999px;
     padding: 0.5rem;
     border: 1px solid var(--color-border);
     z-index: 50;
   }
   ```

3. **Minimal text nav** — No container, just links in the corner:
   ```css
   .nav-minimal {
     position: fixed;
     top: var(--space-8);
     right: var(--space-8);
     display: flex;
     gap: var(--space-6);
     z-index: 50;
   }
   ```

## The Override Checklist

Before finishing implementation, verify these shadcn defaults have been intentionally addressed:

- [ ] `--radius: 0.5rem` — Changed to product-appropriate value?
- [ ] `font-family: Inter` — Replaced with personality-driven font?
- [ ] Default shadows — Replaced with graduated shadow system?
- [ ] Button height `h-10` — Adjusted to match your typography scale?
- [ ] Card styling — Not all identical rectangles with identical shadows?
- [ ] Input styling — Not all white boxes with grey borders?
- [ ] Navigation — Not the standard logo-links-CTA bar?
- [ ] `rounded-md` on everything — Varied by element type?
