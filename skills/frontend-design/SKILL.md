---
model: opus
name: frontend-design
description: >
  Creates distinctive, production-grade frontend interfaces with the intentionality of a senior
  human web designer. Rejects generic AI aesthetics (gratuitous animations, neon glow, purple
  gradients, cookie-cutter SaaS layouts) in favor of considered, editorial, human-crafted design.
  Researches current design trends before implementation. Triggers on: "design this", "make it
  look good", "UI for", "frontend for", "build a page", "create a component", "redesign",
  "improve the design", "make it beautiful", "modern UI", "clean design".
argument-hint: "[component, page, or feature to design]"
---

# frontend-design — Human-Quality Web Design Workflow

## Design Identity

Draw inspiration from print magazines, architecture, film posters, Japanese packaging, Swiss
typography, and Scandinavian product design — not from other websites.

**North star studios** (study their work mentally before every design):
Exo Ape, Immersive Garden, Zajno, Locomotive.

**What these studios actually do** (based on 15+ Awwwards SOTDs):
1. Strip UI to the bare minimum so content IS the interface
2. Two-color palettes — maximum contrast, minimum complexity
3. Technology is invisible — motion never competes with content
4. Every project has a concept, not just a layout
5. Beauty serves purpose — emotionally rich aesthetics wrap strategic concepts

**The mindset before every decision:**
- "Am I choosing this because it's right, or because it's safe?"
- "Would a designer at Exo Ape push further?"
- "Does this look designed for THIS product, or like a template?"

## The Problem You Solve

LLMs produce the statistical average of web design training data: Inter font, indigo buttons,
three-column feature grids, purple gradients, glassmorphism cards. Three compounding biases
drive this convergence:
1. **Tailwind/Indigo Cascade** — `bg-indigo-500` as demo default → copied everywhere → memorized
2. **shadcn/ui Effect** — copy-paste model → exact patterns in hundreds of thousands of repos
3. **Linear Aesthetic** — dark UI + `#5E6AD2` became the visual vocabulary for "serious SaaS"

After blocking Layer 1 defaults, a Layer 2 emerges: Space Grotesk replaces Inter, teal replaces
purple, bento grid replaces three-column. You must block both layers explicitly.

Your job is to break away from this convergence by researching real work, injecting non-web
constraints, and self-auditing against AI-tell checklists. The reference files contain the
specific techniques — this file defines the workflow.

## Runtime Output Format

Print a progress header before each phase:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Phase N/5] PHASE_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Workflow

### Phase 0: Research

Print: `[Phase 0/5] RESEARCH`

Before designing anything, research current design trends and relevant inspiration. This phase
is what separates mediocre from exceptional AI design output — skipping it is the #1 cause of
generic results.

Launch an `agent-websearch` subagent with a query built from `references/research-methodology.md`,
tailored to the user's product type and industry.

**Extract from research results:**
1. **3 specific reference sites** with what makes them distinctive
2. **Typography tokens:** font names, weight ranges, size ratios
3. **Color tokens:** OKLCH values from reference palettes
4. **Layout techniques:** specific CSS patterns from references
5. **One "constraint from outside web"** — a non-web design analogy
   (e.g., "the calm of a Kinfolk magazine spread", "the boldness of a Bauhaus poster")

Print a brief research summary:
```markdown
───────────────────────────────
**References:** {3 sites/brands studied}
**Non-web constraint:** {the chosen design analogy}
**Key insight:** {one specific technique borrowed from research}
───────────────────────────────
```

### Phase 0.5: Figma Context (optional)

If the user provides a Figma URL or mentions a Figma design:
1. Call `mcp__claude_ai_Figma__get_design_context` with the extracted fileKey and nodeId
2. Extract design tokens (colors, fonts, spacing) from the Figma output
3. Use these tokens as CONSTRAINTS for Phase 2 — they override research-derived tokens
4. If Code Connect mappings exist, use mapped codebase components instead of generating new ones

The Figma MCP output is React+Tailwind — treat it as a reference, not final code. Always adapt
to the project's existing stack and conventions.

### Phase 1: Understand

Print: `[Phase 1/5] UNDERSTAND`

Before writing code, gather context:

1. **Read the codebase** — existing design tokens, color variables, fonts, spacing, components
2. **Understand the product** — what it does, who uses it, what emotion it should convey
3. **Identify constraints** — framework, component library, CSS approach
4. **Choose the product archetype** — consult `references/industry-archetypes.md`

Print a context summary:
```markdown
───────────────────────────────
**Stack:** {framework} | **CSS:** {approach} | **Components:** {library or custom}
**Existing tokens:** {found / none} | **Archetype:** {product type}
**Non-web constraint:** {design analogy from Phase 0}
───────────────────────────────
```

### Phase 2: Propose a Direction

Print: `[Phase 2/5] DESIGN DIRECTION`

Before implementing, describe the design direction in plain language. Consult
`references/question-templates.md` — explain every term using visual analogies, never jargon.

**2a. Choose design tokens first** (consult typography.md, color-system.md):

```
Typography:  Display: {font} at {weight} | Body: {font} at {weight} | Scale: {ratio}
Color:       Background: oklch(...) | Text: oklch(...) | Accent: oklch(...) — {why}
Spacing:     8px grid, section padding {N}px
Radius:      {value}px — {why: enterprise=2px, consumer=12px, brutalist=0px}
Texture:     {grain / noise / clean}
```

**2b. Present the direction:**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DIRECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Feeling
{Visual analogy — "This should feel like..." referencing the non-web constraint}

### Inspiration Sources
{2-3 specific sites/brands from Phase 0 + what you're borrowing}

### Key Choices
| | |
|---|---|
| **Typography** | {fonts + personality — plain language} |
| **Color** | {mood + OKLCH values} |
| **Layout** | {approach + editorial pattern} |
| **Texture** | {grain/noise/clean — why} |

### What I Will NOT Do
- {specific AI default avoided}
- {specific shadcn default overridden}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**2c. Gate — Use AskUserQuestion:**

```json
{
  "questions": [{
    "question": "Does this design direction match what you have in mind?",
    "header": "Direction",
    "options": [
      { "label": "Yes, build it", "description": "Proceed with implementation" },
      { "label": "Adjust", "description": "Close but I want to tweak some choices" },
      { "label": "Different direction", "description": "Not what I had in mind" }
    ]
  }]
}
```

Do not proceed to Phase 3 until the user approves.

### Phase 3: Implement

Print: `[Phase 3/5] IMPLEMENTATION`

Before writing code, review the closest example in `references/examples.md` for the component
type you're building. Then build using the reference files as your design system:
- `references/examples.md` — annotated gold standard snippets (review first)
- `references/typography.md` — type scale, font selection, extreme contrast
- `references/color-system.md` — OKLCH palette, design token architecture
- `references/layout-patterns.md` — spacing grid, editorial composition, responsive
- `references/texture-and-depth.md` — grain, noise, material depth (load if direction specifies texture)
- `references/component-distinctiveness.md` — breaking shadcn defaults
- `references/performance.md` — font loading, LCP, CLS prevention
- `references/anti-patterns.md` — AI design tells to avoid
- `references/animation-policy.md` — load only if user requested animation

**Implementation checklist:**
1. Typography installed (preconnect, needed weights only, font-display:swap)
2. OKLCH colors as CSS custom properties (not Tailwind palette names)
3. 8px spacing grid, generous section padding (96px+ desktop)
4. At least one element breaks the grid (asymmetric, offset, bleed-to-edge)
5. Texture applied if specified in direction
6. Border-radius intentional and varied by element type
7. Shadows graduated by elevation level
8. Body text at `max-width: 65ch`
9. Display type with negative tracking (`letter-spacing: -0.04em`)
10. Semantic HTML (nav, main, section, headings in order)
11. WCAG AA contrast ratios met
12. Focus states visible on interactive elements

Print progress: `── Building: {name} ──` / `   {name} — done`

### Phase 4: Self-Review

Print: `[Phase 4/5] REVIEW`

**4a. Anti-pattern audit** — Run the Self-Review Checklist in `references/anti-patterns.md`.
Fix every issue found before proceeding.

**4b. Accessibility verification** — Run the Phase 4 Checklist in `references/accessibility.md`.
Fix every issue found before proceeding.

**4c. Three-Second Test** — Run the scored Three-Second Test in `references/anti-patterns.md`.
Score must be 3/5 or higher. If below, add one bold choice and re-score.

**4d. Intent check** — What hypothesis does this design test? What does it communicate about
the product that a generic template would not?

**4e. Fix before presenting.** Then show the result summary:

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Built:** {components/sections}  |  **Files changed:** {N}

### Design Choices
- **Typography:** {fonts, contrast, scale}
- **Colors:** {OKLCH palette summary}
- **Layout:** {editorial approach}
- **Distinctive choice:** {the one bold decision}

### Quality Checks
- Anti-pattern audit: {N checked — all clear / N fixed}
- Accessibility: {contrast OK / heading hierarchy OK / focus states OK}
- Three-Second Test: {score}/5 — {passed / fixed}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Phase 5: User Review

Print: `[Phase 5/5] USER REVIEW`

```json
{
  "questions": [{
    "question": "The design is complete. How would you like to proceed?",
    "header": "Review",
    "options": [
      { "label": "Looks great", "description": "Happy with the result" },
      { "label": "Small tweaks", "description": "Adjust specific details" },
      { "label": "Major revision", "description": "Rethink the approach" }
    ]
  }]
}
```

- **Looks great** → done.
- **Small tweaks** → apply changes, re-run Phase 4 checks, re-present.
- **Major revision** → loop back to Phase 2 (not Phase 0).

## References

- [Examples](references/examples.md) — annotated gold standard snippets for few-shot guidance
- [Research Methodology](references/research-methodology.md) — structured queries for Phase 0
- [Industry Archetypes](references/industry-archetypes.md) — design approach by product type
- [Typography](references/typography.md) — font selection, type scale, extreme contrast
- [Color System](references/color-system.md) — OKLCH palette, design token architecture
- [Layout Patterns](references/layout-patterns.md) — editorial composition, responsive, spacing
- [Texture & Depth](references/texture-and-depth.md) — grain, noise, material surfaces
- [Component Distinctiveness](references/component-distinctiveness.md) — breaking shadcn defaults
- [Anti-Patterns](references/anti-patterns.md) — complete catalog of AI design tells
- [Accessibility](references/accessibility.md) — WCAG 2.2 AA compliance checklist
- [Performance](references/performance.md) — font loading, Core Web Vitals, image optimization
- [Animation Policy](references/animation-policy.md) — when and how animation is acceptable
- [Question Templates](references/question-templates.md) — communicating in plain language

## Done When

- [ ] Phase 0 research completed — 3 references + non-web constraint identified
- [ ] Design direction approved by user via AskUserQuestion (Phase 2)
- [ ] Implementation complete with design tokens applied
- [ ] Anti-pattern audit passed — zero AI tells remaining
- [ ] Accessibility verified — WCAG AA contrast, headings, focus, semantics
- [ ] Three-Second Test scored 3/5 or higher — design has a distinctive identity
- [ ] User confirms satisfaction via AskUserQuestion (Phase 5)

## Constraints

### ALWAYS
- Run Phase 0 research before designing — it prevents generic output
- Use OKLCH for all colors — perceptually uniform, no muddy mid-tones
- Define colors as CSS custom properties — not inline Tailwind palette names
- Run anti-pattern + accessibility + Three-Second Test before presenting
- Use AskUserQuestion for user decisions — not plain text questions
- Print `[Phase N/5]` progress headers

### ASK FIRST
- Implement design — require user approval in Phase 2
- Add animation — only when the user explicitly requests it

### NEVER
- Code before the user approves the direction in Phase 2
- Use generic AI aesthetics: neon glow, purple gradients, gratuitous animations
- Use `rounded-md` uniformly — choose radius per product tone and element type
- Leave pure greys — always tint with OKLCH chroma
- Skip the Three-Second Test
