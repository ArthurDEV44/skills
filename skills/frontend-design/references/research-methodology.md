# Research Methodology — Structured Queries for Phase 0

Phase 0 research is MANDATORY. The quality of the final design depends directly on the quality
of research conducted before any code is written.

## How to Research

Launch an `agent-websearch` subagent with a query built from the templates below.
The query MUST be specific to the user's product type, not generic.

## Query Templates by Product Type

### SaaS / Dashboard / Internal Tool
```
Research the best-designed SaaS dashboards and data-heavy web applications in 2025-2026.
Look at:
1. Awwwards SOTD winners in the SaaS/dashboard category
2. Linear, Raycast, Arc Browser, Notion — what specific design techniques do they use?
3. Creative agency redesigns of enterprise dashboards
4. Current typography trends for data-dense UIs
5. Color palettes used by top dashboards (NOT Tailwind defaults)
Focus on: information density, typographic hierarchy in tables, sidebar navigation patterns,
data visualization styling, dark mode approaches.
Extract specific: font names, hex/OKLCH color values, border-radius values, spacing systems.
```

### Landing Page / Marketing Site
```
Research the most distinctive landing pages and marketing sites from 2025-2026.
Look at:
1. Awwwards SOTD winners in the corporate/startup category
2. Framer template best-sellers — what visual patterns make them premium?
3. Creative studios' own sites: Exo Ape, Zajno, Lusion, Immersive Garden
4. Startup launches on Product Hunt with exceptional design
5. Current hero section trends — what replaces the centered-text-plus-screenshot pattern?
Focus on: hero composition, typography as layout element, scroll-driven storytelling,
asymmetric layouts, full-bleed imagery, editorial grid systems.
Extract specific: font pairings, color values, section padding values, grid structures.
```

### E-commerce / Product Page
```
Research the best-designed e-commerce sites and product pages in 2025-2026.
Look at:
1. Luxury brand sites: Cartier, Aesop, COS, Acne Studios, Maison Margiela
2. DTC brands with exceptional web presence
3. Product detail page innovations — what goes beyond the standard left-image right-details?
4. Japanese retail and packaging design influence on web
5. Editorial product photography styling and how it integrates with layout
Focus on: product photography treatment, grid systems for product grids, typography for
price and product details, color palettes that don't compete with products.
Extract specific: image treatment CSS, grid column structures, card styling approaches.
```

### Portfolio / Personal Site / Creative
```
Research the most creative personal portfolios and creative sites from 2025-2026.
Look at:
1. Awwwards SOTD winners in the portfolio category
2. Bruno Simon, Lynn Fisher, Gianluca Gradogna — what makes their work feel human?
3. Design agency portfolio patterns that stand out
4. Experimental typography sites on typewolf.com
5. Interactive experiences that use scroll or cursor as a design element
Focus on: single-concept design ideas, experimental typography, one-page navigation,
project showcase layouts that aren't just grids of cards.
Extract specific: animation approaches (if requested), typographic treatments, color moods.
```

### Blog / Editorial / Content Site
```
Research the best-designed editorial and content websites in 2025-2026.
Look at:
1. Award-winning magazine sites: Kinfolk, Monocle, The Outline (archived), Bloomberg
2. Long-form journalism sites with exceptional reading experience
3. Swiss/International Style applied to web — what does that look like in 2025?
4. Drop caps, pull quotes, margin notes — editorial micro-patterns on the web
5. Typography-forward sites on typewolf.com
Focus on: reading experience, article layout variations, typographic rhythm, image placement
within long text, sidebar annotations, progressive disclosure of content.
Extract specific: text column widths, line-height values, paragraph spacing, font choices.
```

## What to Extract from Research

After the research agent returns, extract these concrete values:

### 1. Reference Sites (pick 3)
For each reference:
- Site name and URL
- One specific technique to borrow
- What makes it feel "not AI-generated"

### 2. Typography Tokens
- Display font name + weight range
- Body font name + weight
- Scale ratio observed (measure: heading size / body size)
- Letter-spacing at display size

### 3. Color Tokens (convert to OKLCH if given hex)
- Background color
- Surface/card color
- Primary text color
- Muted text color
- Accent color
- Border color

### 4. Layout Techniques
- Hero section approach (centered / asymmetric / full-bleed / typography-driven)
- Grid structure (12-col / asymmetric / editorial)
- Section spacing scale

### 5. Non-Web Constraint
Choose ONE analogy from outside web design that could inform this project:
- "The editorial calm of a Kinfolk magazine"
- "The information density of a Bloomberg terminal"
- "The material warmth of Japanese ceramic packaging"
- "The bold typography of a Bauhaus poster"
- "The stark minimalism of a Scandinavian furniture showroom"
- "The dramatic lighting of a Martin Scorsese film"
- "The structured chaos of a jazz album cover"
- "The precise geometry of Swiss railway timetables"
- "The tactile roughness of a letterpress print"
- "The contemplative space of a Tadao Ando building"

This constraint MUST visibly influence at least one major design decision.

## Research Quality Check

Before proceeding to Phase 1, verify:
- [ ] You have 3 specific reference sites (not just "awwwards in general")
- [ ] You have concrete typography tokens (font names, not "a modern sans-serif")
- [ ] You have OKLCH color values (not "a warm palette")
- [ ] You have identified one non-web constraint
- [ ] You can describe one specific technique you will borrow from a reference

If any of these are missing, the research was too vague. Run a more specific search query.
