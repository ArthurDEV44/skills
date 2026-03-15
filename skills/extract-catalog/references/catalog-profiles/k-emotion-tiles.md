# Catalog Profile — K Emotion Tiles

Supplier: **K Emotion** (Spain)
Product domain: **Ceramic tiles** (floor, wall, mosaic, outdoor, pool)
Catalog language: Spanish/English bilingual

---

## Page Layout Patterns

### Series intro page (lifestyle)
- Full-page ambiance photo, series name in large type
- Style category indicated in header: MARBLES, CEMENTS, WOODS, STONES, MONOCHROME
- No extractable data — skip to data pages

### Series data page (MAIN EXTRACTION SOURCE)
- **Top strip:** Series name + technical specification line
- **Technical spec line format:** `Porcelánico Esmaltado | Rectificado | 60x120 | 60x60 | V3 | Antislip`
  - Extract: material, rectified, available formats, variation code, finish
- **Icons bar** (horizontal row of small icons): variation (V1–V4), wear resistance (GR/Grupo), anti-slip (R9–R13), frost resistance, chemical resistance
- **Color swatches:** Row of named color squares (e.g., "White", "Sand", "Grey")
- **Format diagram:** Visual representation of available tile sizes with dimensions
- **Reference table:** Grid with columns: Format | Color | Reference | Price

### Reference table structure
```
FORMAT    | COLOR    | REF.        | PVP €/m²
120x120   | White    | MNL-W-120   | 42,90
120x120   | Sand     | MNL-S-120   | 42,90
60x120    | White    | MNL-W-6012  | 36,90
60x120    | Sand     | MNL-S-6012  | 36,90
60x60     | White    | MNL-W-60    | 29,90
```

**Variant decomposition:** Each row = 1 variant (format × color combination).

### Complementary products (on same or next page)
- **Mosaics** (`mosaique`): mesh-mounted, smaller format, separate price
- **Skirting/Rodapié** (`plinthe`): narrow strip format (e.g., 9x60), separate price
- **Décor/Relief** (`decor`): textured variant with 3D surface, separate price
- **Bullnose** (`plinthe`): rounded-edge wall trim

---

## Icons Bar Decoding

Read left to right. Each icon encodes a specific technical property:

| Icon | Property | Extraction |
|------|----------|------------|
| V1/V2/V3/V4 | Shade variation | → `attributes.variation` |
| GR 1–5 / Grupo 1–5 | Wear resistance (PEI) | → `attributes.wearResistance` as `PEI-1`..`PEI-5` |
| R9–R13 | Anti-slip rating | → `attributes.antiSlipRating` |
| Snowflake icon | Frost resistant | → `attributes.frostResistant: true` (if schema supports) |
| UNE EN 16165 Clase 1 Anexo C | Anti-slip norm | Confirms R9 rating for Silk finish |

**IMPORTANT:** V-codes map directly: `V2` → `"V2"`. Never translate to French labels.
**IMPORTANT:** GR/Grupo maps to PEI: `GR 3` → `"PEI-3"`. Never use `"gr3"`.

---

## Extraction Rules

### Product-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Series name | `relational.name` | e.g., "Moonlight" |
| Series name slugified | `relational.slug` | e.g., "moonlight" |
| "Porcelánico Esmaltado" | `relational.material` | → `gres-cerame-emaille` |
| Style category (MARBLES) | `relational.style` | → `marbre` |
| Floor/Wall/Both indicators | `relational.room` | See room mapping |
| "Rectificado" in spec line | `attributes.rectified` | → `true` |
| "Masa Coloreada" in spec line | `attributes.fullBody` | → `true` |
| V-code from icons | `attributes.variation` | → `"V2"` |
| Sol/Mur indicators | `attributes.productType` | → `sol`, `mur`, or `sol-et-mur` |

### Variant-level fields

| Catalog data | Target field | Placement |
|-------------|-------------|-----------|
| Format (120x120) | `relational.format` | As-is: `"120x120"` |
| Width from format | `relational.width` | In cm: `120` |
| Height from format | `relational.height` | In cm: `120` |
| Color name | `relational.color` | Mapped: "White" → `blanc` |
| Reference code | `relational.supplierRef` | As-is: `"MNL-W-120"` |
| Price (PVP €/m²) | `relational.price` | Convert comma: `42.90` |
| Always m² for tiles | `relational.priceUnit` | → `"m2"` |
| Finish from spec line | `relational.finish` | Mapped per value-mappings.md |
| Piece type | `relational.pieceType` | Default `"standard"`, or `"mosaique"`, `"plinthe"`, `"decor"` |
| Wear resistance (floor only) | `variant.attributes.wearResistance` | → `"PEI-3"` etc. (if variantField) |
| Anti-slip (floor only) | `variant.attributes.antiSlipRating` | → `"R10"` etc. (if variantField) |
| Pièces par boîte | `variant.attributes.piecesPerBox` | Integer |
| m² par boîte | `variant.attributes.boxCoverage` | Decimal |

### Thickness
- Standard floor tiles: **typically 9–10mm** (check spec line or fine print)
- Wall tiles: **typically 7–8mm**
- Outdoor/anti-slip: **typically 10–20mm**
- Extract from spec line if present; otherwise `null`

---

## Multi-format Series Handling

A single K Emotion series often has 3–5 formats (e.g., 120x120, 60x120, 60x60, 30x60). Each format × color combination = 1 variant.

**Example:** Series "Moonlight" with 3 formats × 5 colors = 15 variants on a single product.

**Product count rule:** 1 series = 1 product (unless there are separate wall/floor/mosaic variants, in which case each `pieceType` is a separate product within the same series).

---

## Complementary Product Rules

When a series page shows mosaics, plinths, or décor variants:

1. **Mosaic** → Separate product, same series, `pieceType: "mosaique"`
2. **Skirting (Rodapié)** → Separate product, same series, `pieceType: "plinthe"`
3. **Décor/Relief** → Separate product, same series, `pieceType: "decor"`
4. **Bullnose** → Separate product, same series, `pieceType: "plinthe"`

Each gets its own product entry with `seriesName` linking them. The `priceUnit` for mosaics may be `"unit"` (per sheet) instead of `"m2"`.

---

## Common Pitfalls

1. **Missing box coverage/pieces**: Often in fine print below the reference table or on a separate technical page. If not visible, set to `null`.
2. **Silk finish anti-slip**: Silk finishes have inherent anti-slip (R9 per UNE EN 16165). Don't confuse with the explicit "Antislip" finish which is R10+.
3. **Outdoor variants**: Same series may have an outdoor "Antislip" version at a different price point. This is a separate variant (different finish), not a separate product.
4. **Coordinations**: Some series pages show "coordination" tiles from other series. These are NOT variants of the current series — they're cross-references. Ignore them or note as product relationships.
5. **Price tiers**: Some formats (especially large 120x120) are "premium" pricing. Note `supplierPriceTier: "premium"` if distinguishable.
