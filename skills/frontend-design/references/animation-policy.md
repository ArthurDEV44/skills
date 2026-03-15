# Animation Policy

## Default: No Animation

The default state of every element is STATIC. No entrance animation, no scroll animation,
no hover animation, no loading animation.

This is not a suggestion. This is the rule. Animation is added ONLY when:
1. The user explicitly requests animation
2. The animation serves a clear functional purpose (see below)

## Acceptable Animations (always allowed)

These are functional animations that communicate state changes. They use CSS only — no
JavaScript animation libraries.

### Interactive State Transitions
```css
/* Buttons: subtle color/opacity change on hover */
.button {
  transition: background-color 150ms ease, opacity 150ms ease;
}
.button:hover {
  opacity: 0.9;
}

/* Links: underline or color shift */
a {
  transition: color 150ms ease;
}

/* Inputs: border color on focus */
input {
  transition: border-color 150ms ease;
}
input:focus {
  border-color: var(--color-accent);
}
```

### Disclosure/Expand
```css
/* Accordion content reveal */
.accordion-content {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 200ms ease;
}
.accordion-content[data-open] {
  grid-template-rows: 1fr;
}
```

### Loading States
```css
/* Simple opacity pulse for skeleton loaders — no bouncing, no sliding */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
.skeleton {
  animation: pulse 2s ease-in-out infinite;
}
```

### Page/Route Transitions (only if requested)
```css
/* Simple fade, not a full slide/scale production */
@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}
.page-enter {
  animation: fade-in 200ms ease;
}
```

## Forbidden Animations

These MUST NOT be used unless the user explicitly requests them by name:

- Scroll-triggered entrance animations (elements fading/sliding in on scroll)
- Staggered list item animations
- Card hover scale/translate effects
- Parallax scrolling
- Animated gradients or backgrounds
- Text character-by-character reveal
- Counter/number counting animations
- Infinite marquee/ticker animations
- Mouse-follow effects
- Page load sequences where elements animate in one by one
- Spring physics animations (bouncy overshoots)
- 3D perspective transforms on hover
- Animated borders or outlines
- Pulsing/breathing elements (except skeleton loaders)
- Animated SVG decorations

## If the User Requests Animation

When the user explicitly asks for animation, follow these principles:

### Less is More
One well-choreographed moment > scattered micro-animations everywhere.
Pick ONE moment in the user experience to animate and make it count.

### Prefer CSS
Use CSS transitions and animations for everything possible. Only reach for a JavaScript
animation library (GSAP, Motion) when the animation genuinely requires:
- Complex sequencing with stagger that CSS alone can't achieve
- Scroll-linked progress (not just scroll-triggered)
- Physics-based motion
- Animation coordination across multiple elements

### Respect Accessibility
```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```
This is NON-NEGOTIABLE. Every page with any animation must include this.

### Duration Guidelines
- Micro-interactions (hover, press): 100-200ms
- UI transitions (accordion, dropdown): 200-300ms
- Content transitions (page change, modal): 200-400ms
- Decorative animations (if requested): 400-800ms

Never exceed 1000ms for any single animation. Longer animations feel sluggish.

### Easing
- Use `ease` or `ease-out` for most transitions
- Use `ease-in-out` for looping/repeating animations
- Never use `linear` for UI motion (feels mechanical)
- Avoid custom spring/bounce easing unless specifically asked for a "playful" feel
