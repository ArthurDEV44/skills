# Catalog Profile — Karag Furniture (Bathroom Furniture)

Supplier: **Karag** (Italy/Greece)
Product domain: **Bathroom furniture** (vanity cabinets, countertops, mirrors, storage columns)
Catalog language: Italian/English bilingual

---

## Page Layout Patterns

### Lifestyle page
- Full-page or half-page photo of complete bathroom set
- Set name prominent (e.g., "KEA 80 Smooth Cream")
- Shows all components installed together
- No extractable data — skip to data pages

### Dimension/spec page (MAIN EXTRACTION SOURCE)
- **Top:** Product photo (front view of vanity or set)
- **Center:** Technical drawing with dimensions (front view + side view)
- **Bottom:** Component list with references, colors, and prices

### Component list structure
```
KEA 80 SMOOTH CREAM
Component          | Reference      | Color          | Price (€)
Meuble vasque      | KEA-80-SC-MV   | Smooth Cream   | 890,00
Plan vasque        | KEA-80-SC-PV   | Bianco Lucido  | 320,00
Miroir LED         | KEA-80-SC-MI   | —              | 280,00
Colonne            | KEA-80-SC-CO   | Smooth Cream   | 450,00
```

### Separate price page (some catalogs)
- Grid of all models × colors × components with prices
- May be at the end of the catalog section

---

## Ensemble (Set) Architecture

Karag furniture is organized as **ensembles** (sets). One ensemble = multiple products:

| Component | `pieceType` | Notes |
|-----------|-------------|-------|
| Meuble vasque (vanity cabinet) | `meuble` | Main product, includes sink basin |
| Plan vasque (countertop) | `plan-vasque` | May be ceramic or stone |
| Miroir LED (LED mirror) | `miroir` | With integrated LED lighting |
| Colonne (storage column) | `colonne` | Tall storage unit |

**Product count rule:** Each component = 1 separate product. They share a `seriesName` and are linked via `productRelationships`.

---

## Color Handling

Furniture colors are typically described in English/Italian marketing names:

| Catalog name | `relational.color` | Notes |
|-------------|-------------------|-------|
| Smooth Cream | `creme` | Cabinet color |
| Brown | `marron` | Cabinet color |
| White Glossy / Bianco Lucido | `blanc` | Countertop typically |
| Grey Matt / Grigio Opaco | `gris` | |
| Anthracite | `anthracite` | |

**Color belongs to variant level** for furniture (same model available in multiple colors).

Furniture base schema has `color` marked as `variantField: true`, but since `color` is already a relational column on `productVariants`, it goes in `relational.color` (not JSONB attributes).

---

## Extraction Rules

### Product-level fields (per component)

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Set name + component type | `relational.name` | e.g., "KEA 80 Meuble Vasque" |
| Slugified | `relational.slug` | e.g., `kea-80-meuble-vasque` |
| Component material | `relational.material` | → `bois-mdf` (cabinet), `ceramique` (countertop), `verre` (mirror) |
| Always bathroom | `relational.room` | → `["salle-de-bain"]` |
| Not applicable | `relational.style` | → `null` |
| Set base name | `seriesName` | → "KEA 80" |

### Product-level JSONB attributes

| Catalog data | Target field | Notes |
|-------------|-------------|-------|
| Cabinet material | `attributes.furnitureMaterial` | → `MDF`, `solid-wood`, or `melamine` |
| Number of basins | `attributes.numberOfBasins` | Vanity only: 1 or 2 |
| Countertop material type | `attributes.countertopType` | Countertop only: `ceramic`, `stone`, `solid-surface` |

### Variant-level fields (per color of each component)

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Component + color | `relational.name` | e.g., "KEA 80 Meuble Vasque Smooth Cream" |
| Reference code | `relational.supplierRef` | As-is |
| EAN (if present) | `variant.attributes.ean` | Validate: 13 digits |
| Price | `relational.price` | Convert comma decimal |
| Always per unit | `relational.priceUnit` | → `"unit"` |
| Color (mapped) | `relational.color` | → `creme`, `marron`, etc. |
| Finish (if distinguishable) | `relational.finish` | → `mate` or `brillante` |
| Component type | `relational.pieceType` | → `meuble`, `plan-vasque`, `miroir`, `colonne` |
| Width from drawing | `relational.width` | In mm |
| Height from drawing | `relational.height` | In mm |
| Depth from drawing | `relational.thickness` | In mm (or use a separate depth field in JSONB) |

---

## Dimension Extraction

Technical drawings show:
- **Width:** Cabinet/countertop width (e.g., 800mm for KEA 80)
- **Height:** Cabinet height (e.g., 500mm) or mirror height
- **Depth:** Cabinet depth (e.g., 460mm)

The model number often encodes the width: **KEA 80** = 80cm = 800mm width.

All dimensions in **millimeters** (mm).

---

## Product Relationships

Within an ensemble, link components:

| Relationship | Type |
|-------------|------|
| Vanity ↔ Countertop | `accessory` |
| Vanity ↔ Mirror | `complement` |
| Vanity ↔ Column | `complement` |
| Countertop ↔ Mirror | `complement` |

---

## Multi-color Sets

A single furniture model often comes in 2–3 color options:
- KEA 80 Smooth Cream
- KEA 80 Brown

Each color option has the same components at the same (or different) prices.

**Variant decomposition:** Each color = 1 variant per component product.

Example for "KEA 80 Meuble Vasque":
- Variant 1: Smooth Cream → `color: "creme"`, price: 890.00
- Variant 2: Brown → `color: "marron"`, price: 890.00

---

## Separate Price Pages

Some Karag furniture catalogs put all prices on a separate "Price List" page at the end. This page has a grid:

```
Model     | Component     | Cream  | Brown  | Grey
KEA 80    | Meuble vasque | 890    | 890    | 920
KEA 80    | Plan vasque   | 320    | 320    | 320
KEA 80    | Miroir LED    | 280    | 280    | 280
KEA 80    | Colonne       | 450    | 450    | 480
KEA 100   | Meuble vasque | 1090   | 1090   | 1120
...
```

**IMPORTANT:** Cross-reference lifestyle/spec pages with the price page. Don't extract prices from the spec page if a dedicated price page exists (it may be more accurate/complete).

---

## Common Pitfalls

1. **One product per component:** Don't merge the vanity + countertop + mirror into a single product. Each is a separate product with its own pieceType.
2. **Series name vs. product name:** "KEA 80 Smooth Cream" is the SET. Individual products are "KEA 80 Meuble Vasque", "KEA 80 Plan Vasque", etc.
3. **Color on variants:** The color belongs to the variant level. The product "KEA 80 Meuble Vasque" has variants for each available color.
4. **Countertop material vs. cabinet material:** The countertop may be ceramic while the cabinet is MDF. They're separate products with different `material` values.
5. **Mirror has no color:** Mirrors typically don't have a color variant (they're glass). Set `color: null` unless the frame has a color option.
6. **Width encoding in model name:** "KEA 80" = 800mm, "KEA 100" = 1000mm. But always verify against technical drawings.
7. **Missing EAN codes:** Furniture products often lack EAN codes. Set to `null` if absent.
8. **Depth vs. thickness:** For furniture, depth is a major dimension. Use `relational.thickness` if depth isn't a relational column, or store in JSONB attributes as `depth`.
