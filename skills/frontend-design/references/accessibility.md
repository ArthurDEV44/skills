# Accessibility

Accessibility is not a feature you bolt on at the end. It is a constraint that shapes every
design decision from the start — color, typography, layout, interaction.

## Color Contrast (WCAG AA)

**Required ratios:** 4.5:1 for normal text (<24px / <18.66px bold), 3:1 for large text and UI components.

In OKLCH, contrast is driven by the **L (lightness)** channel. A lightness delta of ~50% between
foreground and background typically meets AA. Always verify with a contrast checker — OKLCH
perceptual uniformity helps but does not guarantee WCAG compliance.

```css
/* Safe pairings — verify with contrast checker */
--text-primary:   oklch(15% 0.015 80);  /* L=15% on L=97% bg = high contrast */
--bg-primary:     oklch(97% 0.01 80);

/* Danger zone: muted text must still hit 4.5:1 */
--text-muted:     oklch(40% 0.02 80);   /* L=40% on L=97% bg — check this */
/* oklch(55% ...) on white often FAILS AA for normal text */

/* UI components (borders, icons, form controls): 3:1 minimum */
--border-input:   oklch(65% 0.01 80);   /* 3:1 against white-ish bg */
```

## Focus States

Never remove focus outlines without replacement. `outline: none` on its own is an accessibility violation.

```css
/* Use :focus-visible — only shows for keyboard navigation, not clicks */
:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

/* Remove default only if you provide a visible custom style */
button:focus:not(:focus-visible) {
  outline: none;
}

/* High-contrast focus for dark backgrounds */
:focus-visible {
  outline: 2px solid oklch(85% 0.15 90);
  outline-offset: 2px;
  box-shadow: 0 0 0 4px oklch(0% 0 0 / 0.3);
}
```

## Semantic HTML

Use the correct element. ARIA cannot fix wrong element choices.

- `<nav>` — site navigation (use `aria-label` if more than one nav on the page)
- `<main>` — one per page, contains primary content
- `<header>` / `<footer>` — page or section-level
- `<section>` — thematic grouping, must have a heading
- `<article>` — self-contained content (blog post, card, comment)
- `<aside>` — tangentially related content (sidebar, callout)
- `<button>` for actions, `<a>` for navigation — never the reverse

**Heading hierarchy:** h1 through h6 in order. One `<h1>` per page. Never skip levels
(no h1 then h3). Style independently of semantics using classes.

```css
/* Decouple visual size from semantic level */
.heading-display { font-size: var(--text-display); }
.heading-section { font-size: var(--text-2xl); }
/* Apply to any h-level: <h2 class="heading-display"> */
```

## ARIA

Use ARIA only when semantic HTML is insufficient. The first rule of ARIA: don't use ARIA
if a native element exists.

```html
<!-- Label for icon-only buttons -->
<button aria-label="Close dialog"><svg>...</svg></button>

<!-- Link description to error message -->
<input aria-describedby="email-error" aria-invalid="true" />
<p id="email-error">Enter a valid email address.</p>

<!-- Live regions for dynamic content (toast notifications, status updates) -->
<div aria-live="polite" aria-atomic="true">3 items added to cart.</div>
<!-- Use "assertive" only for critical alerts -->
```

### ARIA Overuse Warning
AI-generated code tends to add ARIA liberally as an "accessibility signal." This creates
more problems than it solves — pages with heavy ARIA usage average 57 accessibility errors.

**Common AI ARIA mistakes to remove:**
- `<button role="button">` — redundant, `<button>` already has this role
- `<nav role="navigation">` — redundant, `<nav>` already implies navigation
- `<a aria-label="Click here">Click here</a>` — duplicates visible text
- `<h2 role="heading" aria-level="2">` — redundant, `<h2>` already conveys this
- `aria-hidden="false"` — does nothing, remove it

## Motion and User Preferences

```css
/* NON-NEGOTIABLE: every page with animation must include this */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Adapt to color scheme preference */
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: oklch(10% 0.015 250);
    --color-text: oklch(92% 0.005 250);
  }
}

/* Increase contrast for users who request it */
@media (prefers-contrast: more) {
  :root {
    --color-text-muted: oklch(25% 0.02 80);  /* Darken muted text */
    --color-border: oklch(45% 0.01 80);       /* Strengthen borders */
  }
}
```

## Touch Targets

WCAG 2.5.8 requires interactive targets of at least 44x44 CSS pixels, with at least
44px spacing between adjacent targets (center-to-center minus target size).

```css
/* Minimum interactive target size */
button, a, input, select, textarea {
  min-height: 44px;
}

/* Icon buttons need explicit sizing */
.icon-button {
  min-width: 44px;
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

/* Spacing between adjacent interactive elements */
.button-group > * + * {
  margin-left: 8px; /* Ensure 44px center-to-center clearance */
}
```

## Typography Accessibility

```css
/* Minimum body text: 16px (1rem). Never go below this for body copy. */
body {
  font-size: 1rem;        /* 16px minimum */
  line-height: 1.5;       /* WCAG SC 1.4.12 minimum for body */
}

/* Paragraph spacing: at least 2x font size for WCAG 1.4.12 */
p + p {
  margin-top: 2em;
}

/* Muted text still needs 4.5:1 contrast — don't go lighter than ~L40% on white */
.text-muted {
  color: oklch(40% 0.02 80);
}
```

## Forms

Every input must have a visible, associated label. Placeholder text is not a label.

```html
<div class="field">
  <label for="email">Email address <span aria-hidden="true">*</span></label>
  <input
    id="email"
    type="email"
    required
    aria-required="true"
    aria-describedby="email-hint email-error"
  />
  <p id="email-hint" class="hint">We will never share your email.</p>
  <p id="email-error" class="error" role="alert">Enter a valid email address.</p>
</div>
```

```css
/* Error state: don't rely on color alone */
.error {
  color: oklch(45% 0.2 25);
  font-size: var(--text-sm);
}
.error::before {
  content: "\26A0 "; /* Warning symbol as secondary indicator */
}
input[aria-invalid="true"] {
  border-color: oklch(45% 0.2 25);
  box-shadow: 0 0 0 1px oklch(45% 0.2 25);
}
```

## Images

- **Informative images:** `alt` describes the content or function, not the format
  (`alt="Revenue chart showing 40% growth in Q3"`, not `alt="chart.png"`)
- **Decorative images:** `alt=""` (empty string) and `aria-hidden="true"` — screen readers skip them
- **Complex images** (charts, diagrams): use `aria-describedby` pointing to a text description

## Skip Links

Provide a skip link as the first focusable element on the page.

```css
.skip-link {
  position: absolute;
  top: -100%;
  left: 16px;
  z-index: 9999;
  padding: 12px 24px;
  background: var(--color-accent);
  color: oklch(100% 0 0);
  font-weight: 600;
  border-radius: 0 0 4px 4px;
  text-decoration: none;
}
.skip-link:focus {
  top: 0;
}
```

```html
<body>
  <a href="#main" class="skip-link">Skip to main content</a>
  <nav>...</nav>
  <main id="main">...</main>
</body>
```

## WCAG 2.2 New Criteria

These WCAG 2.2 criteria are commonly missed by AI-generated code:

### Focus Not Obscured (2.4.11)
The focused element must not be fully hidden by other content (sticky headers, modals, banners).
```css
/* Ensure sticky headers don't hide focused elements */
:focus-visible {
  scroll-margin-top: 80px; /* Height of sticky header + buffer */
}
```

### Dragging Movements (2.5.7)
Any action performed by dragging must have a single-pointer alternative (click, tap).
Drag-to-reorder lists must also support up/down buttons or keyboard arrows.

### Accessible Authentication (3.3.8)
No cognitive function test (CAPTCHA, puzzle) for login without an accessible alternative.
Support: copy-paste into password fields, passkeys, OAuth, biometrics. Never disable paste
on password inputs.

## Phase 4 Checklist

Run these checks during design review before shipping:

1. All text meets 4.5:1 contrast ratio against its background (use a contrast checker, not eyeballing)
2. All UI components (borders, icons, controls) meet 3:1 contrast ratio
3. Every interactive element has a visible `:focus-visible` style
4. Tab order follows visual reading order — test by pressing Tab through the entire page
5. Heading levels are sequential (h1, h2, h3) with no skipped levels
6. Every form input has an associated `<label>` element (not just placeholder)
7. Error messages are linked to inputs via `aria-describedby`
8. All interactive targets are at least 44x44px
9. `prefers-reduced-motion: reduce` disables all animations
10. Skip link is present and functional (first Tab stop jumps to `<main>`)
11. All informative images have descriptive `alt` text; decorative images have `alt=""`
12. Dynamic content updates use `aria-live` regions
13. No information is conveyed by color alone (icons, text, or patterns as secondary indicators)
14. Page is fully operable with keyboard only — no mouse-dependent interactions
