---
model: opus
name: extract-catalog
description: "Extract product metadata from any supplier catalog PDF and output structured JSON aligned with the project's hybrid DB schema (relational columns + JSONB attributes). Schema-driven: auto-discovers DB columns and Zod attribute registry before extraction. Supports tiles, sanitary ware, shower enclosures, furniture, and any future category. Data only — NEVER add images. Use when the user asks to extract catalog data, import a supplier catalog, parse product catalog PDF, or says extract-catalog. Triggers on (1) extracting product data from any catalog PDF, (2) importing supplier products into the database, (3) parsing tile/sanitary/shower/furniture catalog, (4) generating catalog-metadata.json. Do NOT use for image management, pricing updates without a catalog source, or non-product data."
argument-hint: "[path/to/catalog.pdf]"
---

# Extract Catalog — Universal Schema-Driven Extraction

Extract product metadata from **any** supplier catalog PDF using Claude Code's native multimodal PDF reading. Outputs structured JSON aligned with the project's hybrid Drizzle DB schema: **relational columns** for shared fields + **JSONB `attributes`** for category-specific fields.

**Architecture:** Four-phase workflow — Schema Discovery → Extraction → Validation → Seed.

## CRITICAL RULE — No images

**NEVER** add, upload, fetch, or generate any image during catalog extraction. Leave **all** image fields as `null`. The user adds images manually after import. **This rule is absolute.**

## CRITICAL RULE — Null over guessing

If a value is **not explicitly present** in the catalog, set it to `null`. Never infer, estimate, or fabricate values. Extraction confidence must be grounded in visible catalog data.

## Input

Always **PDF files** (catalog pages, typically split by category/series). Use the `Read` tool to read each PDF page visually.

---

## Phase 0 — Schema Discovery (MANDATORY first step)

Before reading any catalog PDF, build the extraction context by reading the live codebase.

### 0.1 — Ask the user for target info

Ask the user (or determine from catalog content):
- **Target category slug** — e.g., `imitation-marbre`, `wc`, `cabine-de-douche`, `meuble-vasque`
- **Supplier name** — e.g., "K Emotion", "Karag"
- **Catalog year**

### 0.2 — Read the DB schema

Read `src/db/schema/products.ts` to extract the current **relational columns** for `products` and `productVariants` tables. Note column names, types, and constraints.

### 0.3 — Resolve the Zod attribute schema

Read `src/lib/validations/attributes/index.ts` to understand the registry. Then resolve the target category:

```
getCategoryAttributeSchema("wc") → wcAttributeSchema
  → fields: material, mountType, evacuationType, seatHeight, rimless
```

Read the resolved Zod file (e.g., `src/lib/validations/attributes/sanitary/wc.ts`). Extract:
- Every field key, its Zod type, and allowed enum values
- FieldMeta: `label`, `fieldType`, `options`, `unit`, `variantField`

### 0.4 — Build the Field Mapping Guide

Construct this mental model before any extraction:

```
=== RELATIONAL COLUMNS (SQL) — product level ===
name: string (required)
slug: string (required, unique)
description: string (required)
shortDesc: string | null
style: string | null — allowed: "marbre", "bois", "beton", "pierre", "uni", ...
material: string | null
room: string[] | null
indoorOutdoor: boolean | null
categoryId: string (required)
seriesId: string | null

=== RELATIONAL COLUMNS (SQL) — variant level ===
name: string (required)
sku: string (required, unique)
supplierRef: string | null
price: decimal (required)
priceUnit: string — "m2" | "unit" | "lot"
format: string | null
width: decimal | null (cm for tiles, mm for sanitary/shower)
height: decimal | null
thickness: decimal | null
color: string | null
finish: string | null
pieceType: string — "standard", "mosaique", "plinthe", "decor", "wc", "bidet", ...
material: string | null (per-variant override)

=== JSONB ATTRIBUTES — product.attributes (from Zod: [CATEGORY]) ===
[list each field with type and allowed values]

=== JSONB ATTRIBUTES — variant.attributes (variantField: true only) ===
[list each field with type and allowed values]
```

### 0.5 — Load catalog profile (if available)

Check if a catalog-specific profile exists in this skill's references:
- `references/catalog-profiles/k-emotion-tiles.md` — K Emotion ceramic tiles
- `references/catalog-profiles/karag-bagno.md` — Karag sanitary ware
- `references/catalog-profiles/karag-box-doccia.md` — Karag shower enclosures
- `references/catalog-profiles/karag-furniture.md` — Karag bathroom furniture

If a matching profile exists, read it for catalog-specific parsing hints (page layout, icons, table formats, value mappings). If no profile exists, proceed with generic multimodal extraction.

Also read `references/value-mappings.md` for shared color/finish/material mappings.

---

## Phase A — Product-by-product extraction

### Step 1 — Identify extraction units

Read the first few pages to identify the catalog structure:
- **Series-based** (K Emotion): one series = one product with multiple variants
- **Model-based** (Karag Bagno): one model block = one product with color variants
- **Size-based** (Box Doccia): one model = one product with size variants
- **Ensemble-based** (Furniture): one set = multiple products (cabinet, countertop, mirror, column)

### Step 2 — For each extraction unit (LOOP)

**Process one product at a time.** Complete all sub-steps before moving to the next. This prevents context saturation.

#### 2.1 — Read all pages for this unit

Read ALL pages belonging to this product/series. Do NOT read pages for other products at this point.

#### 2.2 — Analysis scratchpad (MANDATORY)

Before extracting structured data, write a free-form analysis in a thinking block:

```
ANALYSIS for [PRODUCT NAME]:
─────────────────────────────
What I see on this page:
- Product type: _____
- Number of variants visible: _____
- Data available: [list what's present: prices, dimensions, specs, colors...]
- Data missing: [list what's absent]
- Ambiguities: [anything unclear]
- Catalog-specific elements: [icons, badges, special notation]
```

This scratchpad step forces systematic reading before committing to structured output. It catches missed data that would be hard to recover later.

#### 2.3 — Extract relational fields

For the product level, extract into `relational`:

```json
{
  "relational": {
    "name": "TINOS 100 Cromo",
    "slug": "tinos-100-cromo",
    "description": "[2-3 sentences in French, e-commerce friendly]",
    "shortDesc": "[1 sentence, key specs]",
    "style": null,
    "material": "verre-trempe",
    "room": ["salle-de-bain"],
    "indoorOutdoor": null
  }
}
```

**Description rules:**
- 2-3 sentences in French, natural and e-commerce friendly
- At least 3 of 5 elements: visual appearance, key differentiator, range available, suggested use, special characteristics
- **Anti-template test:** if replacing the product name still makes sense → too generic → rewrite
- Forbidden: "Produit de qualité supérieure", "Design moderne et élégant", "Disponible en plusieurs..."

#### 2.4 — Extract JSONB attributes

Map category-specific data to the resolved Zod schema fields:

```json
{
  "attributes": {
    "doorType": "pivot",
    "glassThickness": 8,
    "glassTreatment": "nano-technology",
    "shape": "rectangular"
  }
}
```

**Rules:**
- Only include keys that exist in the resolved Zod schema
- Use exact enum values from the schema (e.g., `"pivot"` not `"pivotante"`)
- Unrecognized catalog-specific data → store as additional keys (JSONB is permissive)
- `null` for any field not explicitly present in the catalog

#### 2.5 — Extract variants

For each variant row (color × size × finish combination), extract:

```json
{
  "variants": [{
    "relational": {
      "name": "TINOS 100 Cromo 900mm",
      "sku": null,
      "supplierRef": "TNS-100-CR-900",
      "ean": "5206836052670",
      "price": 919.00,
      "priceUnit": "unit",
      "format": "900",
      "width": 900,
      "height": 2000,
      "color": "chrome",
      "finish": "transparent",
      "pieceType": "standard",
      "material": null
    },
    "attributes": {}
  }]
}
```

**Rules:**
- `sku` is generated by the seed script — set to `null` in extraction
- `ean` (EAN barcode) is stored in variant `attributes` if not a relational column
- Only populate `variant.attributes` for fields marked `variantField: true` in FieldMeta
- Price: convert comma decimals (189,00 €) to dot (189.00)
- Dimensions: use the unit convention of the category (cm for tiles, mm for sanitary/shower/furniture)

#### 2.6 — Immediate validation (per-product)

Before moving to the next product, verify:

1. **Required fields present**: `name`, `slug`, `description`, `price`, `priceUnit` on every variant
2. **Enum values valid**: every `attributes` value matches its Zod enum (if constrained)
3. **Variant count reasonable**: cross-check against visible rows in catalog
4. **No fabricated data**: every extracted value traces to visible catalog content
5. **EAN format** (if present): exactly 13 digits
6. **Price > 0**: every variant has a positive price
7. **Supplier ref present**: every variant has a reference code

Fix discrepancies NOW while the source pages are still in context.

---

## Phase B — Assembly and global validation

### Step 3 — Merge all products

Combine all product JSON blocks into the final output structure.

### Step 4 — Cross-product validation (MANDATORY)

**4a. Uniqueness checks:**
- Every `slug` must be unique across products
- Every `supplierRef` must be unique across variants
- Every `ean` must be unique (if present)

**4b. Value consistency:**
- Same color name used consistently across products (not "blanc" in one and "white" in another)
- Same finish name used consistently
- Price ranges reasonable within the same category

**4c. Description uniqueness:**
- Compare all descriptions pairwise
- If any pair shares >50% identical text → REWRITE with product-specific details

**4d. Completeness:**
- Every declared color in color swatches has at least one variant
- No orphan products with zero variants

### Step 5 — Check against app constants

Read `src/lib/constants/product-options.ts` and `src/lib/constants/admin-options.ts`. Print warnings for any new values that need to be added (new colors, materials, finishes, piece types, etc.).

### Step 6 — Write output

Write `catalog-metadata.json` to the project root:

```json
{
  "supplier": "...",
  "catalog": "...",
  "catalogYear": 2025,
  "targetCategorySlug": "cabine-de-douche",
  "extractedAt": "2025-...",
  "products": [
    {
      "relational": { ... },
      "attributes": { ... },
      "seriesName": "TINOS",
      "variants": [
        {
          "relational": { ... },
          "attributes": { ... }
        }
      ]
    }
  ],
  "globalArrays": {
    "allColors": [],
    "allFinishes": [],
    "allMaterials": [],
    "allPieceTypes": [],
    "newConstantsNeeded": []
  }
}
```

### Step 7 — Seed database

Run the universal seed script:

```bash
bun scripts/seed-catalog-universal.ts
```

Wait for completion and verify output counts.

### Step 8 — Cleanup

```bash
rm -f catalog-metadata.json
```

### Step 9 — Summary

Display a summary table with:
- Products extracted (count)
- Variants per product
- Price range
- Attribute coverage (% of Zod fields populated)
- New constants needed
- Validation results (pass/fail)

---

## Common extraction errors to avoid

1. **Adding images**: NEVER add image URLs or any image content. All image fields must be `null`.
2. **Guessing missing values**: If not in the catalog, use `null`. Never infer.
3. **Wrong field placement**: Relational fields go in `relational`, category-specific fields go in `attributes`. Consult the Field Mapping Guide from Phase 0.
4. **Stale enum values**: Always use the Zod schema's exact enum values (e.g., `"V2"` not `"legere"`, `"PEI-3"` not `"gr3"`).
5. **Missing variants**: Read the entire variant table/section systematically, row by row.
6. **Generic descriptions**: Write unique descriptions per product. Run the anti-template test.
7. **Wrong dimension units**: Tiles use cm, sanitary/shower/furniture use mm. Check the category convention.
8. **Italian price format**: Always convert comma to dot (189,00 → 189.00).
9. **Ignoring accessories**: Technical curves, seats, profiles are separate products — don't skip them.
10. **Conflating product and variant attributes**: Check `variantField: true` in FieldMeta. Only variant-level fields go in `variant.attributes`.

## Done When

- [ ] Schema Discovery (Phase 0) complete — field mapping guide constructed
- [ ] All products extracted with analysis scratchpad per product
- [ ] Per-product validation passed (required fields, enum values, variant count)
- [ ] Cross-product validation passed (uniqueness, consistency, completeness)
- [ ] `catalog-metadata.json` written and valid
- [ ] Seed script executed successfully
- [ ] Summary table displayed with counts, price ranges, and attribute coverage

## Constraints (Three-Tier)

### ALWAYS
- Run Phase 0 (Schema Discovery) before reading any catalog PDF
- Write an analysis scratchpad for each product before extracting structured data
- Validate per-product immediately while source pages are in context
- Use `null` for any value not explicitly present in the catalog

### ASK FIRST
- Target category slug and supplier name (Phase 0.1)
- Proceed when variant count seems inconsistent with catalog pages

### NEVER
- Add, upload, fetch, or generate any image — all image fields must be `null`
- Infer, estimate, or fabricate values not explicitly in the catalog
- Skip the analysis scratchpad step — it prevents missed data
- Use enum values not in the resolved Zod schema

## Schema and Mapping References

- For the complete DB schema contract (relational vs JSONB split), see [references/schema-contract.md](references/schema-contract.md)
- For shared value mappings (colors, finishes, materials), see [references/value-mappings.md](references/value-mappings.md)
- For catalog-specific parsing rules, see `references/catalog-profiles/`
