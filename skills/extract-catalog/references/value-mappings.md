# Shared Value Mappings

## Contents
- Color mappings (all catalogs)
- Finish mappings
- Material mappings
- Style mappings
- Room mappings
- pieceType values
- SKU generation patterns

---

## Color Mappings

### From K Emotion catalogs (Spanish/English names)

| Catalog name | DB `color` |
|-------------|-----------|
| White / Blanco | `blanc` |
| Cream / Crema | `creme` |
| Beige | `beige` |
| Grey / Gris | `gris` |
| Dark Grey / Gris Oscuro | `anthracite` |
| Black / Negro | `noir` |
| Brown / Marrón | `marron` |
| Taupe | `taupe` |
| Sand / Arena | `sable` |
| Ivory / Marfil | `ivoire` |
| Honey / Miel | `miel` |
| Natural | `naturel` |
| Almond / Almendra | `amande` |
| Pearl / Perla | `perle` |
| Gold / Dorado | `dore` |
| Silver / Plata | `argent` |
| Multicolor | `multicolore` |

### From Karag catalogs (Italian names)

| Italian name | DB `color` | DB `finish` |
|-------------|-----------|-------------|
| Bianco Lucido | `blanc` | `brillante` |
| Bianco Opaco | `blanc` | `mate` |
| Nero Opaco | `noir` | `mate` |
| Nero Lucido | `noir` | `brillante` |
| Grigio Opaco | `gris` | `mate` |
| Grigio Lucido | `gris` | `brillante` |
| Crema Opaco | `creme` | `mate` |
| Cappuccino Opaco | `taupe` | `mate` |

**Rule for Karag:** Italian color names encode BOTH color AND finish. `Lucido` = glossy → `brillante`. `Opaco` = matt → `mate`.

### Shower enclosure profile colors

| Catalog name | DB `color` |
|-------------|-----------|
| Cromo / Chrome | `chrome` |
| Gun Metal | `gun-metal` |
| Bianco / White | `blanc` |
| Nero / Black | `noir` |
| Oro / Gold | `dore` |

---

## Finish Mappings

### Tile finishes (K Emotion)

| Catalog name | DB `finish` |
|-------------|-------------|
| Mate / Matt | `mate` |
| Silk / Acabado Silk | `satinee` |
| Pulido / Polished | `poli-miroir` |
| Antislip / Antideslizante | `antiderapant` |
| Natural / Naturel | `naturel` |
| Lappato | `lappato` |
| Strutturato / Structured | `structure` |

### Tile finish → antiSlipRating (K Emotion)

| Finish | Default antiSlipRating |
|--------|----------------------|
| Mate | null (unless icons bar says otherwise) |
| Silk | R9 (UNE EN 16165 Clase 1 Anexo C) |
| Antislip | R10 or R11 (read from icons bar) |
| Pulido | null |

### Sanitary/shower/furniture finishes

| Catalog name | DB `finish` |
|-------------|-------------|
| Brillante / Lucido / Glossy | `brillante` |
| Mate / Opaco / Matt | `mate` |
| Transparent / Trasparente | `transparent` |
| Satiné / Satinato | `satine` |
| Fumé / Smoked | `fume` |

---

## Material Mappings

### Tiles

| Catalog description | DB `material` |
|--------------------|--------------|
| Porcelánico Esmaltado / Glazed Porcelain Stoneware | `gres-cerame-emaille` |
| Gres Blanco Porcelart | `porcelart` |
| Revestimiento Pasta Blanca | `gres-pate-blanche` |
| Revestimiento Pasta Roja | `gres-pate-rouge` |
| Full Body / Masa Coloreada | `gres-pleine-masse` |

### Sanitary ware

| Product type | DB `material` |
|-------------|--------------|
| WC / Bidet (ceramica sanitaria) | `ceramique` |
| Toilet seat (termodur/duroplast) | `thermodur` |
| Technical curve (PVC) | `pvc` |

### Shower enclosures

| Component | DB `material` |
|-----------|--------------|
| Glass panel (vetro temperato) | `verre-trempe` |
| Profile (alluminio) | `aluminium` |
| Hinges (ottone) | `laiton` |

### Furniture

| Component | DB `material` |
|-----------|--------------|
| Cabinet (MDF/legno) | `bois-mdf` |
| Countertop ceramic | `ceramique` |
| Countertop stone | `pierre-naturelle` |
| Mirror (vetro) | `verre` |

---

## Style Mappings (tiles only)

| K Emotion category | DB `style` |
|-------------------|-----------|
| Marbles | `marbre` |
| Woods | `bois` |
| Cements | `beton` |
| Stones | `pierre` |
| Monochrome | `uni` |

Non-tile products: `style` = `null` (not applicable).

---

## Room Mappings

| Usage context | DB `room[]` values |
|--------------|-------------------|
| Floor indoor tile | `["sol-interieur"]` |
| Wall indoor tile | `["mur-interieur"]` |
| Floor + Wall tile | `["sol-interieur", "mur-interieur"]` |
| Bathroom products | `["salle-de-bain"]` |
| Kitchen backsplash | `["cuisine"]` |
| Outdoor tile | `["exterieur"]` |
| Pool tile | `["piscine"]` |

---

## pieceType Values

### Tile-specific

| Type | DB `pieceType` |
|------|---------------|
| Standard floor/wall tile | `standard` |
| Mosaic (mesh-mounted) | `mosaique` |
| Skirting / Rodapié | `plinthe` |
| Relief / Textured variant | `decor` |
| Bullnose (wall tile trim) | `plinthe` |

### Sanitary ware

| Type | DB `pieceType` |
|------|---------------|
| Toilet | `wc` |
| Bidet | `bidet` |
| Toilet seat | `siege-wc` |
| Technical drain curve | `courbe-technique` |
| Washbasin | `vasque` |

### Shower

| Type | DB `pieceType` |
|------|---------------|
| Shower cabin panel | `standard` |
| Shower tray | `receveur-douche` |
| Wall profile (accessory) | `accessoire` |

### Furniture

| Type | DB `pieceType` |
|------|---------------|
| Vanity cabinet | `meuble` |
| Countertop | `plan-vasque` |
| LED Mirror | `miroir` |
| Storage column | `colonne` |

---

## SKU Generation Patterns

SKU is generated by the seed script, not during extraction. Set `sku: null` in the JSON output.

**Format by supplier:**
- K Emotion tiles: `CAT-{SERIES4}-{FORMAT}-{COLOR3}-{FINISH3}` (e.g., `CAT-MOON-120X120-BLA-SIL`)
- Karag sanitary: `KRG-{SERIES4}-{MODEL}-{COLOR3}-{PIECE}` (e.g., `KRG-MILO-2141D-BLA-WC`)
- Karag shower: `KRG-{MODEL}-{SIZE}-{COLOR3}` (e.g., `KRG-TNS100-900-CHR`)
- Karag furniture: `KRG-{SERIES4}-{PIECE}-{COLOR3}` (e.g., `KRG-KEA8-MEU-CRE`)

---

## Variation Code Mapping (tiles only)

| Icons bar | Zod value | French label |
|-----------|----------|--------------|
| V1 (Slight Variation) | `V1` | Uniforme |
| V2 (Moderate Variation) | `V2` | Légère |
| V3 (High Variation) | `V3` | Modérée |
| V4 (Random Variation) | `V4` | Forte |

**IMPORTANT:** Never use the French words (`"legere"`, `"moderee"`) as values. Always use V-codes (`"V2"`, `"V3"`).

## Wear Resistance Mapping (floor tiles only)

| Icons bar | Zod value |
|-----------|----------|
| GR 1 / Grupo 1 | `PEI-1` |
| GR 2 / Grupo 2 | `PEI-2` |
| GR 3 / Grupo 3 | `PEI-3` |
| GR 4 / Grupo 4 | `PEI-4` |
| GR 5 / Grupo 5 | `PEI-5` |

**IMPORTANT:** Never use `"gr3"` / `"gr4"`. Always use `"PEI-3"` / `"PEI-4"`.
