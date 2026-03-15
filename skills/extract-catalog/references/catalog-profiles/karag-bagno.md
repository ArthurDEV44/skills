# Catalog Profile — Karag Bagno (Sanitary Ware)

Supplier: **Karag** (Italy/Greece)
Product domain: **Sanitary ware** (WC, bidets, toilet seats, reservoirs, technical curves)
Catalog language: Italian/English bilingual

---

## Page Layout Patterns

### Lifestyle page
- Full-page photo of installed bathroom, model name in corner
- Series name prominent (e.g., "MILOS", "NEW LEGEND", "ZINA", "NEW PAROS")
- No extractable data — skip to data pages

### Model data page (MAIN EXTRACTION SOURCE)
- **Model block:** One block per product model
- **Block structure:**
  ```
  MODEL NAME (e.g., MILOS 2141 D)
  ┌─────────────────────────────────────────┐
  │ Photo    │ Specs column  │ Dimensions   │
  │          │ - Material    │ drawing      │
  │          │ - Colors      │ W × D × H   │
  │          │ - Features    │              │
  │          │ - EAN         │              │
  │          │ - Price       │              │
  └─────────────────────────────────────────┘
  ```

### Technical dimensions drawing
- Side-view or top-view technical drawing
- Dimensions in **millimeters** (mm)
- Extract: width, depth (as `height` in DB), seat height for WC

### Price/reference table (may be on separate page)
- Grid format: Model | Color | Reference | EAN | Price
- Or inline in the model block

---

## Product Type Detection

Each model block indicates the product type through visual and textual cues:

| Cue | Product type | `pieceType` |
|-----|-------------|-------------|
| Toilet shape, "WC" label | WC | `wc` |
| Bidet shape, "BIDET" label | Bidet | `bidet` |
| "Sedile" / "Seat" label | Toilet seat | `siege-wc` |
| "Cassetta" / "Reservoir" / "Cistern" | Reservoir | `reservoir` (use mapped value) |
| "Curva Tecnica" / "Technical Curve" | Technical drain | `courbe-technique` |

---

## Badge/Feature Decoding

Model blocks often have feature badges or icons:

| Badge | Meaning | Extraction |
|-------|---------|------------|
| RIMLESS | Rimless WC (no rim) | `attributes.rimless: true` |
| TORNADO | Tornado flush technology | `attributes.flushType: "tornado"` (JSONB permissive) |
| SOFT CLOSE | Soft-close seat | Note in description |
| QUICK RELEASE | Quick-release seat | Note in description |
| DUROPLAST / TERMODUR | Seat material | `relational.material: "thermodur"` (on seat variant) |

---

## Color Handling (Italian names)

Karag uses Italian color names that encode BOTH color AND finish:

| Italian name | `relational.color` | `relational.finish` |
|-------------|-------------------|---------------------|
| Bianco Lucido | `blanc` | `brillante` |
| Bianco Opaco | `blanc` | `mate` |
| Nero Opaco | `noir` | `mate` |
| Nero Lucido | `noir` | `brillante` |
| Grigio Opaco | `gris` | `mate` |
| Crema Opaco | `creme` | `mate` |
| Cappuccino Opaco | `taupe` | `mate` |

**Rule:** Always split Italian color names into separate `color` and `finish` values. `Lucido` = `brillante`, `Opaco` = `mate`.

---

## Extraction Rules

### Product-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Model name (MILOS) | `seriesName` | Series grouping |
| Full model name (MILOS 2141 D) | `relational.name` | Product display name |
| Slugified name | `relational.slug` | e.g., `milos-2141-d` |
| "Ceramica sanitaria" | `relational.material` | → `ceramique` |
| Always bathroom | `relational.room` | → `["salle-de-bain"]` |
| Not applicable | `relational.style` | → `null` |

### Product-level JSONB attributes

| Catalog data | Target field | Notes |
|-------------|-------------|-------|
| Wall-hung / Floor-standing | `attributes.mountType` | → `wall-hung` or `floor-standing` |
| Material type | `attributes.material` | → `ceramic` (most sanitary) |
| Horizontal/Vertical/Universal drain | `attributes.evacuationType` | WC only |
| Seat height from drawing (mm) | `attributes.seatHeight` | WC only, number |
| Rimless badge | `attributes.rimless` | WC only, boolean |
| Flush volume (e.g., "3/6L") | `attributes.flushVolume` | Reservoir only |

### Variant-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Color variant name | `relational.name` | e.g., "MILOS 2141 D Bianco Lucido" |
| Color (mapped) | `relational.color` | → `blanc` |
| Finish (mapped) | `relational.finish` | → `brillante` |
| Reference code | `relational.supplierRef` | As-is |
| EAN (13 digits) | Store in `variant.attributes.ean` | Validate: exactly 13 digits |
| Price | `relational.price` | Convert comma decimal |
| Always per unit | `relational.priceUnit` | → `"unit"` |
| Width from drawing (mm) | `relational.width` | In mm |
| Depth from drawing (mm) | `relational.height` | In mm (depth = height in DB) |
| pieceType | `relational.pieceType` | → `wc`, `bidet`, `siege-wc`, etc. |

---

## Series Grouping

Karag sanitary products are grouped by series (e.g., MILOS, NEW LEGEND). Within a series, related products are:

- **WC** (main product)
- **Bidet** (same series, different pieceType)
- **Toilet seat** (accessory to WC)
- **Technical curve** (accessory to WC)

Use `productRelationships` to link:
- WC ↔ Bidet → `type: "complement"`
- WC ↔ Toilet seat → `type: "accessory"`
- WC ↔ Technical curve → `type: "accessory"`

---

## Multi-page Extraction

Karag catalogs often split a series across 2–3 pages:
1. **Page 1:** Lifestyle photo + series name
2. **Page 2:** WC and bidet models with specs + dimensions
3. **Page 3:** Accessories (seats, curves) + full price table

**IMPORTANT:** Read ALL pages for a series before extracting. Prices may only appear on the last page.

---

## Built-in Reservoir Section

Some catalogs have a separate "Cassette da Incasso" (built-in reservoirs) section:
- These are standalone products, not part of a series
- `pieceType: "reservoir"` (mapped from schema)
- Each model has: reference, dimensions (width × height × depth), flush volume, price
- They link to WCs via `productRelationships` → `type: "accessory"`

---

## Common Pitfalls

1. **Color = Color + Finish:** Never store "Bianco Lucido" as a single color value. Always split.
2. **EAN validation:** EAN codes must be exactly 13 digits. If shorter/longer, it's likely a model reference, not an EAN.
3. **Dimensions are in mm:** Unlike tiles (cm), sanitary dimensions are always in millimeters.
4. **Toilet seats are separate products:** Don't merge the seat into the WC product. They have different prices, references, and materials.
5. **Technical curves:** Small PVC drain accessories. Easy to miss. They're listed at the bottom of model blocks.
6. **Price per unit:** Sanitary products are always priced per unit, never per m².
7. **Inclusive vs. exclusive prices:** Check if prices include the seat or not. Some models say "WC con sedile" (WC with seat) — the seat may still be a separate SKU.
