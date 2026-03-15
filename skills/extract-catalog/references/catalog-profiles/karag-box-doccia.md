# Catalog Profile — Karag Box Doccia (Shower Enclosures)

Supplier: **Karag** (Italy/Greece)
Product domain: **Shower enclosures** (cabins, walk-in panels, shower trays, profiles)
Catalog language: Italian/English bilingual

---

## Page Layout Patterns

### Lifestyle page
- Full-page photo of installed shower enclosure
- Model name and profile color in large type (e.g., "TINOS 100 Cromo")
- No extractable data — skip to data pages

### Model data page (MAIN EXTRACTION SOURCE)
- **Header:** Model name + profile color
- **Left section:** Product photo (front view)
- **Center section:** Technical drawing with dimensions
- **Right section:** Specifications + features list
- **Bottom:** Price table with size variants

### Price table structure
```
TINOS 100 CROMO
Size (mm) | Reference      | EAN             | Price (€)
800       | TNS-100-CR-800 | 5206836052663   | 859,00
900       | TNS-100-CR-900 | 5206836052670   | 919,00
1000      | TNS-100-CR-100 | 5206836052687   | 979,00
1200      | TNS-100-CR-120 | 5206836052694   | 1.099,00
```

**Variant decomposition:** Each row = 1 variant (different size). Profile color is fixed per page.

---

## Product Type Detection

| Product description | `pieceType` |
|--------------------|-------------|
| Shower cabin / Box doccia | `standard` |
| Walk-in panel | `standard` |
| Shower tray / Piatto doccia | `receveur-douche` |
| Wall profile / Profilo | `accessoire` |

---

## Feature Decoding

| Feature text | Extraction |
|-------------|------------|
| "Vetro temperato 8mm" | `attributes.glassThickness: 8` |
| "Nano Technology" / "Easy Clean" | `attributes.glassTreatment: "nano-technology"` |
| "Porta scorrevole" / "Sliding door" | `attributes.doorType: "sliding"` |
| "Porta a battente" / "Pivot door" | `attributes.doorType: "pivot"` |
| "Porta pieghevole" / "Folding door" | `attributes.doorType: "folding"` |
| "Rettangolare" / "Rectangular" | `attributes.shape: "rectangular"` |
| "Quarto di cerchio" / "Quarter circle" | `attributes.shape: "quarter-circle"` |
| "Walk-in" | `attributes.shape: "walk-in"` |
| "Profilo in alluminio" | Profile material: `aluminium` |

---

## Profile Color Handling

Shower enclosures come in different profile (frame) colors. Each profile color is typically presented on a separate page with its own price table.

| Italian/English name | `relational.color` |
|---------------------|-------------------|
| Cromo / Chrome | `chrome` |
| Gun Metal | `gun-metal` |
| Bianco / White | `blanc` |
| Nero / Black | `noir` |
| Oro / Gold | `dore` |

**Product naming convention:** `{MODEL} {SIZE} {PROFILE_COLOR}` → e.g., "TINOS 100 Cromo 900mm"

**Product grouping:** One model with one profile color = 1 product. Different profile colors = different products in the same series.

Example:
- Product 1: "TINOS 100 Cromo" (series: TINOS) → 4 size variants (800, 900, 1000, 1200)
- Product 2: "TINOS 100 Gun Metal" (series: TINOS) → 4 size variants
- Product 3: "TINOS 100 Bianco" (series: TINOS) → 4 size variants

---

## Extraction Rules

### Product-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Model + profile color | `relational.name` | e.g., "TINOS 100 Cromo" |
| Slugified | `relational.slug` | e.g., `tinos-100-cromo` |
| "Vetro temperato" | `relational.material` | → `verre-trempe` |
| Always bathroom | `relational.room` | → `["salle-de-bain"]` |
| Not applicable | `relational.style` | → `null` |
| Model base name | `seriesName` | → "TINOS" |

### Product-level JSONB attributes

| Catalog data | Target field |
|-------------|-------------|
| Door type | `attributes.doorType` |
| Glass thickness (mm) | `attributes.glassThickness` |
| Glass treatment | `attributes.glassTreatment` |
| Shape | `attributes.shape` |

### Variant-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Model + size | `relational.name` | e.g., "TINOS 100 Cromo 900mm" |
| Reference code | `relational.supplierRef` | As-is |
| EAN (13 digits) | `variant.attributes.ean` | Validate format |
| Price | `relational.price` | Convert: `1.099,00` → `1099.00` |
| Always per unit | `relational.priceUnit` | → `"unit"` |
| Size (opening width) | `relational.format` | e.g., `"900"` |
| Width from dimensions | `relational.width` | In mm |
| Height from dimensions | `relational.height` | In mm (standard: 2000mm) |
| Profile color | `relational.color` | Mapped per color table |
| Glass finish | `relational.finish` | → `transparent` or `fume` |
| Always standard for cabins | `relational.pieceType` | → `"standard"` |

---

## Dimension Extraction

Technical drawings show:
- **Width:** Opening dimension (corresponds to size variant: 800, 900, 1000, 1200mm)
- **Height:** Panel height (typically 2000mm, fixed across variants)
- **Depth:** For corner/rectangular models only
- **Glass thickness:** From specs (8mm standard)

All dimensions in **millimeters** (mm).

---

## Italian Price Format

Karag uses Italian number formatting:
- Thousands separator: **dot** (`.`)
- Decimal separator: **comma** (`,`)

Example: `1.099,00 €` → `1099.00`

**Always convert:** Remove dots from thousands, replace comma with dot for decimals.

---

## Multi-page Models

Some models span multiple pages:
1. **Page 1:** Lifestyle photo (skip)
2. **Page 2:** First profile color (Cromo) — specs + dimensions + price table
3. **Page 3:** Second profile color (Gun Metal) — price table only (reuses specs from page 2)
4. **Page 4:** Third profile color (Bianco) — price table only

**IMPORTANT:** Specs (glass thickness, door type, shape) are shared across profile colors. Only prices and references change.

---

## Shower Trays and Accessories

Some Box Doccia catalogs include:
- **Shower trays (Piatti doccia):** Separate products, `pieceType: "receveur-douche"`, priced per unit
- **Wall profiles:** Mounting accessories, `pieceType: "accessoire"`

Link shower trays to cabin products via `productRelationships` → `type: "complement"`.

---

## Common Pitfalls

1. **Italian price format:** `1.099,00` is one thousand ninety-nine euros, NOT one-point-zero-nine-nine. Always remove the dot separator.
2. **Profile color vs. glass color:** Profile color (Cromo, Nero) is the FRAME color → `relational.color`. Glass type (transparent, fumé) is the GLASS finish → `relational.finish`.
3. **Size is the opening width:** The "900" in the size column means 900mm opening width, not overall width. Use dimension drawing for actual width/height.
4. **Height is usually fixed:** Most shower enclosures are 2000mm tall regardless of width variant.
5. **Missing EAN on some rows:** Not all size variants have EAN codes. Set to `null` if absent.
6. **Reused specs across colors:** Don't re-read specs for each profile color page. Copy from the first page.
