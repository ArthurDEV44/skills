---
model: opus
name: frontend-design
description: >
  Create distinctive, production-grade frontend interfaces with the intentionality of a senior human
  web designer. Rejects generic AI aesthetics: no gratuitous animations, no neon glow, no purple
  gradients, no hover effects on every element, no cookie-cutter SaaS layouts. Produces design that
  feels considered, editorial, and human-crafted. Uses /meta-code pipeline to research current design
  trends before implementation. Use when the user asks to build web components, pages, landing pages,
  dashboards, or applications and wants high design quality. Also triggers on: "design this", "make
  it look good", "UI for", "frontend for", "build a page", "create a component", "redesign",
  "improve the design", "make it beautiful", "modern UI", "clean design".
argument-hint: "[component, page, or feature to design]"
---

# frontend-design — Human-Quality Web Design Workflow

## Design Identity

Draw inspiration from print magazines, architecture, film posters, Japanese packaging, Swiss
typography, and Scandinavian product design — NOT from other websites. Think about composition,
tension, rhythm, and negative space the way a photographer thinks about framing.

Every choice — font size, color, spacing, alignment — must be made deliberately.
Nothing should be there "because it's the default."

**North star references** (study these studios' work mentally before every design):
- Exo Ape — Interactive minimalism, 2-color palettes, content IS the interface
- Immersive Garden — Luxury brand world-building, cinematic pacing
- Zajno — Rule-breaking animation, bold color, interactive storytelling
- Locomotive — Scroll-driven narrative, editorial typography, spatial rhythm

## The Fundamental Problem You Solve

LLMs predict tokens from training data dominated by Tailwind defaults, shadcn/ui components,
and the post-2019 "Linear aesthetic." Without explicit intervention, you will produce the
statistical average of early-2020s web design: Inter font, indigo buttons, three-column feature
grids, purple gradients, glassmorphism cards. This is not because these are good choices — it's
because they are the MOST COMMON choices in training data.

**Your job is to systematically break away from this convergence** by:
1. Researching real award-winning work before designing
2. Injecting a non-web design constraint that forces unexplored territory
3. Using OKLCH colors, extreme type contrast, and intentional texture
4. Self-auditing against a comprehensive AI-tell checklist

## Runtime Output Format

Before each phase, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Phase N/5] PHASE_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Between major phases, print a thin separator: `───────────────────────────────`

## Workflow

### Phase 0: Research — MANDATORY

Print: `[Phase 0/5] RESEARCH`

**This phase is NOT optional.** Before designing anything, you MUST research current design trends
and relevant inspiration. This is the single biggest difference between mediocre and exceptional
AI design output.

Launch an `agent-websearch` subagent with a structured research query. Use the template in
`references/research-methodology.md` to construct the query based on:
- The industry/product type the user is building
- Current award-winning sites in that space
- Typography and color trends for that specific vertical
- Specific Framer templates, Awwwards winners, or creative agency work relevant to the brief

**Research extraction:** From the research results, extract and write down:
1. **3 specific reference sites** with what makes them distinctive
2. **Typography tokens:** font names, weight ranges, size ratios seen in the best examples
3. **Color tokens:** specific hex/OKLCH values from reference palettes
4. **Layout techniques:** specific CSS patterns used by the references
5. **One "constraint from outside web"** — a non-web design reference that could inform the project
   (e.g., "the density of a Bloomberg terminal", "the calm of a Kinfolk magazine spread",
   "the boldness of a Bauhaus poster", "the material warmth of Japanese ceramic packaging")

Print a brief research summary:
```markdown
───────────────────────────────
**Research Summary**
**References:** {3 sites/brands studied}
**Non-web constraint:** {the chosen design analogy}
**Key insight:** {one specific technique borrowed from research}
───────────────────────────────
```

### Phase 1: Understand Before Designing

Print: `[Phase 1/5] UNDERSTAND`

Before writing ANY code, gather context:

1. **Read the codebase** — Look for existing design tokens, color variables, font imports,
   spacing systems, component patterns. Respect what exists.
2. **Understand the product** — What does the product do? Who uses it? What emotion should the
   interface convey? A fintech dashboard feels different from a creative portfolio.
3. **Identify constraints** — What framework? What component library? What CSS approach?
   Design within real constraints, not in a vacuum.
4. **Choose the product archetype** — Consult `references/industry-archetypes.md` to select
   the right design approach for this product category.

Print a brief context summary:
```markdown
───────────────────────────────
**Stack:** {framework} | **CSS:** {approach} | **Components:** {library or custom}
**Existing design tokens:** {found / none}
**Target:** {what we're building}
**Archetype:** {industry/product type from archetypes reference}
**Non-web constraint:** {the design analogy chosen in Phase 0}
───────────────────────────────
```

### Phase 2: Propose a Direction

Print: `[Phase 2/5] DESIGN DIRECTION`

Before implementing, describe your design direction in plain language the user can understand.
Consult `references/question-templates.md` for how to communicate with the user.

**2a. Choose your design tokens FIRST:**

Before proposing the visual direction, decide on concrete values:

```
Typography:
  Display: {font name} at weight {N} — why this font
  Body: {font name} at weight {N} — why this font
  Scale ratio: {1.25 / 1.333 / 1.618}
  Display tracking: {-0.04em to -0.02em}

Color (OKLCH):
  Background: oklch({L}% {C} {H}) — {mood description}
  Surface: oklch({L}% {C} {H})
  Text: oklch({L}% {C} {H})
  Text muted: oklch({L}% {C} {H})
  Accent: oklch({L}% {C} {H}) — {why this accent}
  Border: oklch({L}% {C} {H})

Spacing: 8px grid, section padding {N}px
Border-radius: {value}px — {why: enterprise=2px, consumer=12px, brutalist=0px}
Shadow style: {diffuse / hard-offset / none}
Texture: {grain overlay / noise / clean}
```

**2b. Present the direction** using this template:

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DIRECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Feeling
{Visual analogy — "This should feel like..." with a concrete, non-web reference}
{Reference the non-web constraint chosen in Phase 0}

### Inspiration Sources
{Name 2-3 specific sites/brands from Phase 0 research + what you're borrowing from each}

### Key Choices

| | |
|---|---|
| **Typography** | {font names + what personality they bring — in plain language} |
| **Color palette** | {dominant color + accent — described as mood AND shown as color values} |
| **Spacing** | {tight/airy/generous + real-world analogy} |
| **Layout** | {approach described visually + which editorial pattern from references} |
| **Texture** | {grain/noise/clean — why} |
| **Border style** | {radius + shadow approach — what signal it sends} |

### What I Will NOT Do
- {specific default avoided — name the exact pattern, e.g., "No three-column feature grid"}
- {specific shadcn default overridden, e.g., "No rounded-md everywhere"}
- {specific AI tell avoided}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**2c. Interactive approval — Use AskUserQuestion:**

```json
{
  "questions": [{
    "question": "Does this design direction match what you have in mind?",
    "header": "Direction",
    "options": [
      { "label": "Yes, build it", "description": "Proceed with this direction — implement the design" },
      { "label": "Adjust", "description": "The direction is close but I want to tweak some choices" },
      { "label": "Different direction", "description": "This isn't what I had in mind — let me describe what I want" }
    ]
  }]
}
```

- If user selects **"Yes, build it"** → proceed to Phase 3.
- If user selects **"Adjust"** → ask which specific choices to change, revise, re-present.
- If user selects **"Different direction"** → ask the user to describe what they want using
  "Imagine..." framing from `references/question-templates.md`, then build a new direction.

**GATE:** User explicitly approves the design direction via AskUserQuestion.

### Phase 3: Implement

Print: `[Phase 3/5] IMPLEMENTATION`

Build the interface following the design system in the reference files:
- Consult `references/typography.md` for type scale and font selection
- Consult `references/color-system.md` for OKLCH palette construction
- Consult `references/layout-patterns.md` for spacing grid and editorial composition
- Consult `references/texture-and-depth.md` for grain, noise, and material depth
- Consult `references/component-distinctiveness.md` for breaking away from shadcn defaults
- Consult `references/anti-patterns.md` to verify you are not falling into AI design traps

**Implementation checklist (verify each during build):**

1. **Typography installed correctly** — Google Fonts preconnect, only needed weights loaded
2. **OKLCH colors defined as CSS custom properties** — not inline Tailwind palette names
3. **Spacing uses the 8px grid** — no arbitrary values
4. **At least one element breaks the grid** — asymmetric hero, offset content, bleed-to-edge
5. **Texture applied** — grain/noise overlay if specified in direction
6. **Border-radius is intentional** — not uniform `rounded-md` everywhere
7. **Shadows are graduated** — different depths for different elevation levels
8. **Body text has `max-width: 65ch`** — never full container width
9. **Display type has negative tracking** — `letter-spacing: -0.04em` at hero sizes
10. **Section padding is generous** — minimum 96px vertical on desktop

When building multiple components or sections, print progress:
```
── Building: {component/section name} ──
```

After each major piece is complete, print:
```
   {component/section name} — done
```

### Phase 4: Self-Review

Print: `[Phase 4/5] REVIEW`

**4a. Anti-pattern audit:**

Run through the COMPLETE anti-pattern checklist in `references/anti-patterns.md`.
Answer each of these questions honestly:

1. Could a senior designer at Exo Ape or Zajno tell this was AI-generated? What gives it away?
2. Is ANY element using shadcn/ui defaults without intentional customization?
3. Does the typography have extreme contrast (3x+ size jumps, 400+ weight difference)?
4. Are colors specified in OKLCH with warm/cool tinting (not pure grey)?
5. Is there texture/grain where specified?
6. Does at least one layout element break the expected grid?
7. Is there intentional whitespace asymmetry (not uniform padding everywhere)?
8. Could you remove ANY element without hurting the design? If yes, remove it.
9. Does this design have a POINT OF VIEW or is it trying to please everyone?
10. Does the border-radius vary intentionally (not uniform `rounded-md`)?

**4b. The "Three-Second Test":**
Look at the design as a whole. In three seconds, can you identify:
- What makes this design DISTINCTIVE (not just "clean")?
- What is the ONE unusual choice that a human designer would have made?
- Does anything feel "safe" in a way that makes it forgettable?

If the design passes as "clean but generic," it has FAILED. Go back and add one bold choice:
a larger headline, a surprising accent color, an asymmetric layout break, a grain texture.

**4c. Fix before presenting:**
If ANY anti-pattern is found or the Three-Second Test fails, fix it BEFORE presenting to the user.

**4d. Present the result summary:**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Built:** {list of components/sections created}
**Files changed:** {N} files

### Design Choices Applied
- **Typography:** {fonts used, weight contrast, scale ratio}
- **Colors:** {OKLCH palette summary}
- **Layout:** {editorial approach used}
- **Texture:** {grain/noise/clean}
- **Distinctive choice:** {the one bold decision that makes this not-generic}

### Anti-Pattern Check
- {N} potential AI tells checked — {all clear / N fixed}
- Three-Second Test: {passed / fixed — what was changed}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Phase 5: User Review

Print: `[Phase 5/5] USER REVIEW`

**Use AskUserQuestion:**

```json
{
  "questions": [{
    "question": "The design is complete. How would you like to proceed?",
    "header": "Review",
    "options": [
      { "label": "Looks great", "description": "I'm happy with the result — we're done" },
      { "label": "Small tweaks", "description": "It's close but I want to adjust specific details (colors, spacing, text)" },
      { "label": "Major revision", "description": "This needs significant changes — let's rethink the approach" }
    ]
  }]
}
```

- If user selects **"Looks great"** → print final summary and STOP.
- If user selects **"Small tweaks"** → ask which elements to adjust, apply changes,
  re-run anti-pattern check, re-present with AskUserQuestion.
- If user selects **"Major revision"** → ask what's not working, loop back to Phase 2.

**GATE:** User explicitly confirms satisfaction via AskUserQuestion.

## Core Design Rules

### No Animation by Default

Do NOT add any animation unless the user explicitly requests it.
No framer-motion. No entrance animations. No scroll-triggered reveals.
No hover scale effects. No button press animations. No staggered list animations.
No loading skeletons that pulse. No animated gradients.

CSS `transition` on interactive elements (buttons, links, inputs) for state changes (hover, focus,
active) is acceptable ONLY if subtle (150-200ms, opacity or color change only, no transform).

If the user asks for animation, consult `references/animation-policy.md`.

### No Decoration Without Purpose

Every visual element must earn its place. Ask: "What does this communicate?"
- A shadow communicates elevation → acceptable if intentional and graduated
- A gradient communicates... what exactly? If you can't answer, don't use it
- A border radius communicates friendliness → but not when applied uniformly to everything
- An icon communicates function → but not when it's purely decorative filler

### Typography Is the Design

The typeface IS the personality. The type scale IS the hierarchy. The line-height IS the breathing
room. Get typography right and the rest follows. Get it wrong and nothing else matters.

**Extreme contrast is mandatory:**
- Weight: 200-300 body vs 800-900 headlines (not 400 vs 600)
- Size: 3x+ jump between heading and body (not timid 1.5x)
- Tracking: -0.04em at display sizes, 0 at body, +0.08em at caption/label sizes
- Line-height: 0.95-1.1 for display, 1.6 for body

Read `references/typography.md` before choosing any font.

### Color Is an OKLCH Decision

Use `oklch()` for ALL color definitions. OKLCH produces perceptually uniform scales where
lightness changes look consistent. Hex and HSL produce muddy mid-tones.

One dominant palette. One accent color that means something. Neutrals that have warmth or
coolness — never pure grey (`oklch(50% 0 0)` is banned; always add chroma and hue).

Read `references/color-system.md` before choosing any color.

### Texture Separates Human from AI

AI output is perfectly flat. Human-designed sites have depth through:
- Grain/noise overlays at 3-5% opacity
- Subtle background gradients (tonal, not decorative)
- Material-inspired surfaces (not glassmorphism)

Consult `references/texture-and-depth.md` for implementation.

### Layout Is Composition

Stop stacking sections in a predictable sequence. Think about what the user sees FIRST, what
they see SECOND, and how their eye travels. Use asymmetry, scale contrast, and negative space
to create visual rhythm.

Read `references/layout-patterns.md` before structuring any page.

### Components Must Not Look Like Defaults

If you're using shadcn/ui or any component library, you MUST customize:
- Border-radius (choose based on product tone, not `rounded-md` for everything)
- Shadow depth (graduated system: sm/md/lg with different offset and blur)
- Button styles (not the default primary/secondary pattern)
- Card styles (not uniform rounded cards with identical shadows)

Consult `references/component-distinctiveness.md` for specific overrides.

## How to Ask the User Questions

When you need the user's input on a design direction, you MUST follow the rules
in `references/question-templates.md`. The key principle: explain every term, every option,
every trade-off in language a non-designer can instantly understand. Use visual analogies.
Never use jargon without explaining it.

## Hard Rules

1. ALWAYS run Phase 0 (Research) — it is MANDATORY, not optional.
2. NEVER implement before the user approves the design direction via AskUserQuestion in Phase 2.
3. NEVER use plain text questions like "Does this look good?" — ALWAYS use AskUserQuestion.
4. Print `[Phase N/5]` progress headers before each phase — NEVER skip progress indicators.
5. ALWAYS run the anti-pattern check AND the Three-Second Test in Phase 4 before presenting.
6. No animation unless the user explicitly requests it.
7. Every visual choice must be intentional — if you can't explain WHY, change it.
8. ALWAYS use OKLCH for color definitions, not hex or HSL.
9. ALWAYS define colors as CSS custom properties, not inline Tailwind palette names.
10. If Phase 4 reveals AI design tells, fix them BEFORE presenting to the user.
11. "Major revision" in Phase 5 loops back to Phase 2 — it does NOT restart from Phase 0.
12. The "non-web constraint" chosen in Phase 0 MUST visibly influence the final design.

## DO NOT

- Start coding before the user approves the design direction in Phase 2.
- Skip Phase 0 research — this is the #1 cause of generic output.
- Use generic AI aesthetics: neon glow, purple gradients, gratuitous animations.
- Use Tailwind palette names directly (`bg-indigo-500`) — always define custom properties.
- Use `rounded-md` as default border-radius — choose intentionally per product tone.
- Leave pure greys — always tint with warmth or coolness using OKLCH chroma.
- Skip the Three-Second Test in Phase 4.
- Ask the user about implementation details — only ask about visual/feeling decisions.
- Present design options using abstract jargon — always use visual analogies.
- Add animation of any kind unless explicitly requested by the user.
- Use the same shadow depth for all elements.
- Leave body text at full container width — always constrain with `max-width: 65ch`.

## References

- [Research Methodology](references/research-methodology.md) — structured queries for Phase 0
- [Industry Archetypes](references/industry-archetypes.md) — design approach by product type
- [Design Philosophy](references/design-philosophy.md) — the deeper "why" behind every rule
- [Typography System](references/typography.md) — font selection, type scale, OKLCH, extreme contrast
- [Color System](references/color-system.md) — OKLCH palette construction, warm neutrals
- [Layout Patterns](references/layout-patterns.md) — editorial composition, magazine techniques
- [Texture & Depth](references/texture-and-depth.md) — grain, noise, material surfaces
- [Component Distinctiveness](references/component-distinctiveness.md) — breaking shadcn defaults
- [Anti-Patterns](references/anti-patterns.md) — complete catalog of AI design tells
- [Animation Policy](references/animation-policy.md) — when and how animation is acceptable
- [Question Templates](references/question-templates.md) — how to communicate in plain language

## Done When

- [ ] Phase 0 (Research) completed with 3 reference sites and non-web constraint identified
- [ ] Design direction approved by user via AskUserQuestion (Phase 2)
- [ ] Implementation complete with all design tokens applied
- [ ] Anti-pattern audit passed (Phase 4) — zero AI design tells remaining
- [ ] Three-Second Test passed — design has a distinctive, non-generic identity
- [ ] User confirms satisfaction via AskUserQuestion (Phase 5)

## Constraints (Three-Tier)

### ALWAYS
- Run Phase 0 (Research) — it is mandatory, not optional
- Use OKLCH for all color definitions — never hex or HSL
- Define colors as CSS custom properties — never inline Tailwind palette names
- Run anti-pattern check AND Three-Second Test before presenting to user
- Print progress headers before each phase

### ASK FIRST
- Implement design (require user approval of direction via AskUserQuestion in Phase 2)
- Add animation of any kind (only if user explicitly requests it)

### NEVER
- Start coding before user approves design direction in Phase 2
- Use generic AI aesthetics: neon glow, purple gradients, gratuitous animations
- Use `rounded-md` as default border-radius — choose intentionally per product tone
- Leave pure greys — always tint with warmth or coolness using OKLCH chroma
- Add animation unless explicitly requested by the user
