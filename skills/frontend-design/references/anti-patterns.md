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

### 14. The Hero Metric Layout
**What:** Large number + small label below + colored accent line to the left. Appears in ~90% of AI-generated dashboards and landing pages.
**Fix:** Use typography hierarchy and layout position to communicate importance instead of the formulaic metric-label-accent pattern. Let the content structure dictate the presentation.

### 15. The Frankenstein Layout
**What:** Individual components look fine in isolation, but the page has no unified visual hierarchy. Sections feel like they were designed separately and stacked.
**Root cause:** AI generates components one at a time without maintaining page-level compositional awareness.
**Fix:** After assembling all sections, review the page as a whole. Check: does the eye flow naturally? Is there a clear visual hierarchy across sections, not just within them?

## Tier 3: Subtle AI Patterns (detected by senior designers)

### 16. Uniform Spacing
**What:** Every section has the same padding. Every card has the same gap. The page
feels like a CSS framework demo.
**Fix:** Vary spacing intentionally. Hero: 128px padding. Regular sections: 96px.
Tight content groups: 48px. See `layout-patterns.md`.

### 17. Everything Centered
**What:** All text centered. All sections centered. All headings centered.
**Fix:** Left-align body text (always). Center headings only as a deliberate composition
choice. Use asymmetric layouts.

### 18. Safe Color Choices
**What:** Neutral grey palette with a "safe" blue accent. Nothing offensive, nothing memorable.
**Fix:** Choose colors with personality appropriate to the brand. Use OKLCH with actual chroma.
See `color-system.md`.

### 19. Decorative SVG Blobs
**What:** Abstract blob shapes, wave dividers, decorative dots/circles.
**Fix:** Remove them. If a section transition feels abrupt, fix the spacing.

### 20. Stock Illustration Style
**What:** Flat/geometric illustrations (unDraw, Blush, Storyset) as hero images.
**Fix:** Use real UI screenshots, photography, or no illustration at all.

### 21. Excessive Micro-Copy
**What:** Every button says something "clever." Every empty state has a cute message.
**Fix:** Write clear, direct copy. "Sign up" not "Get started on your journey."

### 22. No Texture
**What:** Every surface is a perfect solid color. No grain, no noise, no tonal variation.
**Root cause:** AI generates mathematically clean surfaces. Human designers add imperfection.
**Fix:** Add grain/noise overlay at 3-5% opacity on key surfaces. See `texture-and-depth.md`.

### 23. Tailwind Palette Names as Color System
**What:** Using `bg-slate-50 text-slate-900 border-slate-200` directly instead of custom
semantic properties.
**Root cause:** AI generates Tailwind utility classes from training data.
**Fix:** Always define custom CSS properties in OKLCH. Never use Tailwind palette names
for your color system.

### 24. Identical Card Heights
**What:** Every card in a grid is the exact same height, creating a rigid, template-like feel.
**Fix:** Allow natural height variation. Or use asymmetric grid where some cards span
multiple rows. See `layout-patterns.md`.

### 25. Default Button Patterns
**What:** Primary (filled) + Secondary (outlined) + Ghost (text) — the exact shadcn button
variant trio, styled identically to the library defaults.
**Fix:** Style buttons for your specific product tone. See `component-distinctiveness.md`.

### 26. Missing Accessibility Signals
**What:** No visible focus states, no skip links, heading hierarchy broken (h1 → h3), color contrast below WCAG AA.
**Root cause:** AI generates visually appealing output without considering keyboard navigation or screen readers.
**Fix:** See `accessibility.md`. Check contrast ratios, heading order, focus visibility, and semantic HTML.

## Tier 4: Emerging AI Patterns (2025-2026)

### 27. AI Spontaneity Layouts
**What:** Hyper-personalized, seemingly random layout variations that are actually generated
from the same underlying template with randomized parameters.
**Root cause:** AI tools now generate "creative variation" by default, producing layouts that
look different on every page load but share identical structural DNA.
**Fix:** Intentional composition is designed once, not randomized. If a layout feels "creative"
but you can't explain why each element is placed where it is, it's generated noise.

### 28. Y2K Retro-Saturation
**What:** Hyper-saturated neon palettes, retro-futurist gradients, and 2000s-era color
combinations marketed as "bold" or "anti-corporate."
**Root cause:** AI design tools over-indexed on the Y2K revival trend, producing saturated
palettes as the opposite of the muted corporate aesthetic they're trained to avoid.
**Fix:** Bold does not mean saturated. Use OKLCH chroma intentionally — high chroma on one
accent element, not across the entire palette.

### 29. Bento Grid Monoculture
**What:** Every feature section uses a bento grid (unequal rectangles in a grid) regardless
of whether the content calls for it. The bento grid has become the 2025 equivalent of the
three-column feature grid.
**Root cause:** Bento grids appeared as the "creative" alternative to equal columns, and AI
tools now default to them as the new "modern" layout.
**Fix:** Choose layout FROM the content: single-column narrative, asymmetric split, typography-
driven hero, or yes — a bento grid, but only when content has genuinely unequal importance.

### 30. ARIA Overuse
**What:** Redundant ARIA roles, labels, and attributes on elements that already have native
semantics. `<button role="button">`, `<nav role="navigation">`, `aria-label` that duplicates
visible text.
**Root cause:** AI treats ARIA as an "accessibility signal" and adds it liberally. Pages with
heavy ARIA usage average 57 accessibility errors — more than double those without.
**Fix:** First rule of ARIA: don't use ARIA if a native HTML element exists. Remove redundant
roles. Use `aria-label` only on elements without visible text (icon-only buttons).

### 31. Hero Metric Pattern (promoted from Tier 3)
**What:** Large number + small label below + colored accent line to the left. Now appears in
~90% of AI-generated dashboards and landing pages. Previously Tier 3 (#14), now detectable
by anyone as an AI tell.
**Fix:** Communicate importance through typography hierarchy and layout position, not through
the formulaic number-label-accent trio. Let the content structure vary — not every stat needs
the same visual treatment.

## Self-Review Checklist

Before presenting the design, answer these questions. Every "no" is a fix before shipping.

1. Could I explain WHY every font size, color, and spacing value was chosen?
2. Does this look specifically designed for THIS product, not a template?
3. Would removing any element hurt the design? If not, remove it.
4. Does the design have a POINT OF VIEW, or is it trying to please everyone?
5. Is the typography contrast extreme enough (3x+ size jumps, 400+ weight difference)?
6. Are colors in OKLCH with warm/cool tinting (no pure grey)?
7. Is there texture/grain where appropriate?
8. Does the border-radius vary intentionally by element type?
9. Does at least one element break the expected grid?
10. Is there enough whitespace that the design can "breathe"?
11. Are contrast ratios WCAG AA compliant (see `accessibility.md`)?

## Three-Second Test (Scored)

After all fixes, look at the complete design as a whole for three seconds. Score 0 or 1 on each:

1. **DISTINCTIVE:** Can you name what makes it visually unique? (not just "clean")
2. **BOLD CHOICE:** Is there one choice a safe designer would not have made?
3. **TENSION:** Is there visual contrast — large/small, dense/sparse, serif/sans?
4. **HUMAN TOUCH:** Is there texture, asymmetry, or deliberate imperfection?
5. **POINT OF VIEW:** Does the design say something about THIS product specifically?

**Score: __/5.** Below 3 → add one bold choice and re-score before presenting.
