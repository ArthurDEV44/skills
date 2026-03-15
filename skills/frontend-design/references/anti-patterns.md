# Anti-Patterns — The Complete AI Design Tell Catalog

Use this as a checklist. Before presenting ANY design to the user, verify that NONE of these
patterns are present. If any are found, fix them before showing the result.

## The "AI Slop" Quick Test

If 3 or more of these are present, the design reads as machine-generated:

1. Inter or Roboto as the only typeface
2. Purple/indigo gradient as hero background or accent color
3. 3D floating illustration of abstract human or geometric shape
4. Glassmorphism card floating in soft pastel gradient void
5. "Trusted by" logo marquee immediately under hero
6. Three-column equal-weight feature section with icon + title + 2 sentences
7. Spacing that is uniform throughout (no breathing room hierarchy)
8. CTA button at `rounded-md` in the platform's default primary color
9. Cards with identical height, padding, and visual weight
10. No grain, no texture, no deliberate imperfection

## Tier 1: Instant AI Tells (detected by anyone)

### 1. The Purple/Indigo Problem
**What:** Indigo or purple as primary accent, purple-to-blue gradients on dark backgrounds.
**Root cause:** Tailwind UI's default button is `bg-indigo-500`. Adam Wathan publicly
acknowledged this in 2025 — the choice of `indigo-500` as a demo color literally shaped
what LLMs think "modern web" looks like.
**Fix:** Choose organic accents in OKLCH — forest green, rust, ochre, burgundy, slate blue.

### 2. Inter Everywhere
**What:** Inter as the only typeface, used at uniform weights (400/600).
**Root cause:** Inter is the #1 Google Font and appears in most web examples.
**Fix:** See `typography.md` for personality-driven alternatives. Use weight contrast
of 300 vs 900, not 400 vs 600.

### 3. Three-Column Feature Grid
**What:** Exactly three cards with icons in a row, each with identical structure and height.
**Root cause:** This is the most common pattern in SaaS landing page training data.
**Fix:** Use asymmetric layouts, staggered grids, bento grid with size-encoding-importance,
or single-column narrative. See `layout-patterns.md`.

### 4. Framer Motion on Everything
**What:** Every heading fades in on scroll. Every card staggers into view. Custom cursor.
Page transitions on every route change.
**Root cause:** AI associates "polished" with "animated" and adds motion indiscriminately.
**Fix:** No animation by default. See `animation-policy.md`.

### 5. Hover Scale on Cards
**What:** `transform: scale(1.02)` or `translateY(-4px)` on hover for every card element.
**Fix:** Remove entirely. Cards are content containers, not buttons. Only truly interactive
elements get hover states, and those should be subtle (opacity or color shift only).

### 6. Neon Glow Effects
**What:** Glowing borders, glowing text, glowing icons against dark backgrounds.
**Fix:** Use subtle luminosity through layered OKLCH backgrounds and careful contrast instead.

## Tier 2: Common AI Patterns (detected by designers)

### 7. Glassmorphism Cards
**What:** Semi-transparent cards with `backdrop-filter: blur()` over gradient backgrounds.
**Root cause:** Peaked in 2021, massively overrepresented in training data.
**Fix:** Use solid backgrounds with subtle borders. See `texture-and-depth.md`.

### 8. Generic Gradient Backgrounds
**What:** Full-section gradient backgrounds (especially radial behind hero sections).
**Fix:** Use solid OKLCH colors or very subtle tonal shifts. See `texture-and-depth.md`.

### 9. Rounded Corners on Everything (`rounded-md` Monoculture)
**What:** `border-radius: 0.375rem` (shadcn default) applied uniformly to every element.
**Root cause:** shadcn/ui's `--radius: 0.5rem` is memorized by every LLM.
**Fix:** Choose radius based on product tone: 2px enterprise, 0px brutalist, 12px consumer.
Vary by element type. See `component-distinctiveness.md`.

### 10. Drop Shadows on Everything
**What:** Same `box-shadow` on every card creating fake uniform depth.
**Fix:** Use a graduated shadow system. Cards at rest don't need shadows — use background
contrast or borders. See `component-distinctiveness.md`.

### 11. The SaaS Template Sequence
**What:** Hero → Logo bar → Features → Big feature → Testimonials → Pricing → FAQ → CTA → Footer.
Always this order. Regardless of product or content.
**Fix:** Design the page FROM the content. See `layout-patterns.md`.

### 12. Icon Overload
**What:** Lucide/Heroicons as decorative filler for every feature card or list item.
**Fix:** Icons clarify function, not decorate text. If removing the icon doesn't hurt
comprehension, remove it.

### 13. The "Linear Clone" Aesthetic
**What:** Dark UI + indigo/purple accent + subtle radial gradient + Inter/Geist + `rounded-lg`
cards. This is what AI thinks "serious SaaS product" looks like because Linear popularized it.
**Root cause:** Linear's `#5E6AD2` ("Magic Blue") saturated design tutorials from 2021-2023.
**Fix:** If building a dark SaaS UI, choose a different accent (forest green, rust, ochre),
a different border-radius (2px sharp or 0px brutalist), and a different font (IBM Plex, Bricolage).

## Tier 3: Subtle AI Patterns (detected by senior designers)

### 14. Uniform Spacing
**What:** Every section has the same padding. Every card has the same gap. The page
feels like a CSS framework demo.
**Fix:** Vary spacing intentionally. Hero: 128px padding. Regular sections: 96px.
Tight content groups: 48px. See `layout-patterns.md`.

### 15. Everything Centered
**What:** All text centered. All sections centered. All headings centered.
**Fix:** Left-align body text (always). Center headings only as a deliberate composition
choice. Use asymmetric layouts.

### 16. Safe Color Choices
**What:** Neutral grey palette with a "safe" blue accent. Nothing offensive, nothing memorable.
**Fix:** Choose colors with personality appropriate to the brand. Use OKLCH with actual chroma.
See `color-system.md`.

### 17. Decorative SVG Blobs
**What:** Abstract blob shapes, wave dividers, decorative dots/circles.
**Fix:** Remove them. If a section transition feels abrupt, fix the spacing.

### 18. Stock Illustration Style
**What:** Flat/geometric illustrations (unDraw, Blush, Storyset) as hero images.
**Fix:** Use real UI screenshots, photography, or no illustration at all.

### 19. Excessive Micro-Copy
**What:** Every button says something "clever." Every empty state has a cute message.
**Fix:** Write clear, direct copy. "Sign up" not "Get started on your journey."

### 20. No Texture
**What:** Every surface is a perfect solid color. No grain, no noise, no tonal variation.
**Root cause:** AI generates mathematically clean surfaces. Human designers add imperfection.
**Fix:** Add grain/noise overlay at 3-5% opacity on key surfaces. See `texture-and-depth.md`.

### 21. Tailwind Palette Names as Color System
**What:** Using `bg-slate-50 text-slate-900 border-slate-200` directly instead of custom
semantic properties.
**Root cause:** AI generates Tailwind utility classes from training data.
**Fix:** Always define custom CSS properties in OKLCH. Never use Tailwind palette names
for your color system.

### 22. Identical Card Heights
**What:** Every card in a grid is the exact same height, creating a rigid, template-like feel.
**Fix:** Allow natural height variation. Or use asymmetric grid where some cards span
multiple rows. See `layout-patterns.md`.

### 23. Default Button Patterns
**What:** Primary (filled) + Secondary (outlined) + Ghost (text) — the exact shadcn button
variant trio, styled identically to the library defaults.
**Fix:** Style buttons for your specific product tone. See `component-distinctiveness.md`.

## Self-Review Checklist

Before presenting the design, answer these questions:

1. Could a designer at Exo Ape or Zajno tell this was AI-generated? What gives it away?
2. Is there ANYTHING that exists "because that's what websites usually look like"?
3. Could I explain WHY every font size, color, and spacing value was chosen?
4. Does this look like a template or specifically designed for THIS product?
5. Is there any animation that doesn't serve a clear functional purpose?
6. Am I using more than one accent color? Why?
7. Are all my sections the same width? Should some be narrower or wider?
8. Is there enough whitespace that the design can "breathe"?
9. Would removing any element hurt the design? If not, remove it.
10. Does this design have a POINT OF VIEW, or is it trying to please everyone?
11. Is the typography contrast extreme enough (3x+ size jumps, 400+ weight difference)?
12. Are colors in OKLCH with warm/cool tinting (not pure grey)?
13. Is there texture/grain where appropriate?
14. Does the border-radius vary intentionally by element type?
15. Does at least one element break the expected grid?

## The Three-Second Test

After all fixes, look at the complete design for three seconds and answer:
- What makes this DISTINCTIVE? (If the answer is "it's clean," it has failed.)
- What is the ONE bold choice? (There must be at least one.)
- Would this feel at home on awwwards? (If not, push harder.)
