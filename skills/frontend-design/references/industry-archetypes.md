# Industry Archetypes — Design Approach by Product Type

Every product category has a visual language. The mistake AI makes is applying the same
"clean modern SaaS" aesthetic to everything. A funeral home website should not look like
a startup dashboard. A luxury brand site should not look like a developer tool.

Choose the archetype that matches the product, then use its constraints.

## Archetypes

### 1. SaaS / Developer Tool
**Feeling:** Precise, efficient, trustworthy, focused
**Typography:** One geometric sans (Geist, IBM Plex Sans) or monospace for labels (JetBrains Mono)
**Color:** Cool dark palette (near-black with blue tint) OR warm light (cream/stone), single accent
**Border-radius:** 2-4px (sharp = serious)
**Shadow:** Minimal, layered backgrounds instead of shadows
**Layout:** Dense information hierarchy, sidebar + content, data tables
**Texture:** Clean — no grain, precision matters
**Reference mood:** "A well-organized Notion workspace" or "Terminal with good typography"
**DO NOT:** Make it look like Linear (dark + indigo). That's the AI default for this category.

### 2. Consumer App / Social
**Feeling:** Friendly, approachable, fun, personal
**Typography:** Rounded grotesque (Bricolage Grotesque, Nunito), friendly weight contrast
**Color:** Bright, saturated accent on warm neutral background
**Border-radius:** 12-24px (soft = friendly)
**Shadow:** Soft, diffuse (`0 8px 32px rgba(...)`)
**Layout:** Card-based, generous spacing, focus on user content
**Texture:** Subtle grain on backgrounds — warm, tactile
**Reference mood:** "Unboxing a well-designed product" or "A bright café"
**DO NOT:** Use dark mode by default. Consumer apps should feel inviting.

### 3. Luxury / Fashion / Premium
**Feeling:** Refined, exclusive, quiet confidence, aspirational
**Typography:** High-contrast serif for display (Cormorant, Bodoni Moda, Libre Caslon),
thin sans for body. Large display sizes. Generous spacing.
**Color:** Muted, desaturated palette. Blacks that are near-black with warm tint.
Whites that are cream. Accent color used extremely sparingly (one element per screen).
**Border-radius:** 0px (luxury = sharp precision)
**Shadow:** None or barely visible. Elevation through background contrast only.
**Layout:** Magazine editorial — asymmetric, large images, generous whitespace.
Single-column narrative. Content bleeds to edges.
**Texture:** Subtle paper grain, noise at 2-3% opacity
**Reference mood:** "Aesop packaging" or "A Kinfolk magazine spread"
**DO NOT:** Use bright colors. Luxury whispers; it doesn't shout.

### 4. Creative Agency / Studio
**Feeling:** Bold, experimental, confident, distinctive
**Typography:** One unusual display face that IS the identity. Variable weight extremes.
Display type as layout structure, not just labels. Kinetic if animation is requested.
**Color:** Two-color restricted palette with maximum contrast. OR a single unexpected
color (terracotta, chartreuse, rust) against neutral.
**Border-radius:** 0px (brutalist) or exaggerated (pill shapes). Never safe middle ground.
**Shadow:** Hard offset (`4px 4px 0 #000`) or none
**Layout:** Broken grids, overlapping elements, text-as-hero, experimental scroll
**Texture:** Heavy grain, noise overlays, raw/imperfect textures
**Reference mood:** "A concrete brutalist building" or "An indie film festival poster"
**DO NOT:** Make it clean and safe. Agency sites MUST have a point of view.

### 5. Editorial / Blog / Content
**Feeling:** Thoughtful, readable, long-form, considered
**Typography:** Serif for body (Newsreader, Libre Caslon Text, Literata), editorial display serif
or contrasting grotesque for headlines. Reading experience is everything.
**Color:** Warm off-white background, warm near-black text. Accent only for links and
one call-to-action. Nothing else colored.
**Border-radius:** 2-4px or 0px (editorial = precise)
**Shadow:** None
**Layout:** Single-column narrow (max-width: 65ch), with pull quotes, drop caps, large
margin notes. Images full-bleed or offset from text column.
**Texture:** Paper-like warmth on background, subtle grain
**Reference mood:** "Reading a well-typeset book" or "The New York Times Magazine"
**DO NOT:** Add sidebar widgets, tags clouds, or feature lists. Let the content breathe.

### 6. E-commerce / Product
**Feeling:** Clean, product-focused, trustworthy, conversion-oriented
**Typography:** Clean sans for navigation and details, slightly warmer face for product descriptions.
Price typography large and confident.
**Color:** Neutral palette that doesn't compete with product photography. Background should
disappear. Accent only on "Add to Cart" — one color, one purpose.
**Border-radius:** 4-8px (friendly but not childish)
**Shadow:** Subtle product card shadows, larger on hover (the one context where hover shadows
are appropriate — they signal clickability on product cards)
**Layout:** Grid for product listings, asymmetric for product detail. Large product photography.
**Texture:** Clean — products are the texture
**Reference mood:** "A COS or Aesop store — the shelves disappear, the product remains"
**DO NOT:** Use colored backgrounds behind products. The product IS the color.

### 7. Fintech / Finance / Enterprise
**Feeling:** Authoritative, stable, trustworthy, data-rich
**Typography:** IBM Plex family, or a confident grotesque (Geist, Untitled Sans). Monospace
for numbers and financial data. Weight contrast for hierarchy.
**Color:** Deep navy or forest green as primary. Conservative accent (gold, warm grey).
NEVER use red prominently (except for losses — it's semantic in finance).
**Border-radius:** 2-4px (sharp = trust)
**Shadow:** Subtle elevation for cards holding financial data
**Layout:** Dashboard-style information density. Tables with clear headers. Charts that
use the same color tokens as the UI.
**Texture:** Clean — data precision demands visual clarity
**Reference mood:** "The reassuring weight of a bank statement" or "A well-designed annual report"
**DO NOT:** Use playful design. Finance demands seriousness.

### 8. Health / Wellness / Mindfulness
**Feeling:** Calm, breathing, natural, gentle
**Typography:** Rounded sans (Satoshi, Nunito) or gentle serif (Gambetta). Light weights.
Generous line-height (1.8+). Nothing compressed or aggressive.
**Color:** Nature-inspired — sage greens, soft blues, warm earth tones. Desaturated.
Background should feel like natural light.
**Border-radius:** 12-16px (soft, organic)
**Shadow:** Very soft, large blur radius (`0 12px 48px rgba(...)` at very low opacity)
**Layout:** Generous spacing — double what you think you need. Let every section breathe.
Single-column preferred. Images integrated gently.
**Texture:** Organic grain, natural material references
**Reference mood:** "A spa waiting room" or "Morning light through sheer curtains"
**DO NOT:** Use bold colors, aggressive typography, or dense layouts.

## How to Use This

1. In Phase 1, identify which archetype matches the product
2. Use the archetype's constraints as your starting point
3. Then PERSONALIZE within those constraints — the archetype prevents generic design,
   but the non-web constraint from Phase 0 adds the distinctive character
4. If the product spans two archetypes (e.g., a fintech app with consumer-friendly goals),
   pick the PRIMARY archetype and borrow ONE element from the secondary
