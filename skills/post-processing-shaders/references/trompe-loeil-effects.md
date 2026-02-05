# Trompe l'Oeil & Optical Illusion Effects

## Table of Contents

- [Staggered LED Cell Panel](#staggered-led-cell-panel)
- [Woven Crochet Effect](#woven-crochet-effect)
- [Lego Bricks Effect](#lego-bricks-effect)
- [Fluted & Frosted Glass](#fluted--frosted-glass)

---

## Staggered LED Cell Panel

Mimics a physical LED panel with staggered sub-pixel columns and visible cell borders.

### Column Staggering

Offset every odd column vertically before pixelation:

```glsl
float maskStagger = 0.5;
vec2 normalizedPixelSize = pixelSize / resolution;
vec2 coord = uv / normalizedPixelSize;

float columnStagger = mod(floor(coord.x), 2.0) * maskStagger;

vec2 offsetUV = uv;
offsetUV.y += columnStagger * normalizedPixelSize.y;

vec2 uvPixel = normalizedPixelSize * floor(offsetUV / normalizedPixelSize);
```

### Sub-Pixel Staggering (RGB sub-cells)

Split each cell into 3 vertical sub-cells (like CRT RGB strips), each staggered independently:

```glsl
vec2 subcoord = coord * vec2(3, 1);
float subPixelIndex = mod(floor(subcoord.x), 3.0);
float subPixelStagger = subPixelIndex * maskStagger;

vec2 offsetUV = uv;
offsetUV.y += (columnStagger + subPixelStagger) * normalizedPixelSize.y;
```

### Cell Border Mask

Draw a dark border around each sub-cell to make the LED structure visible:

```glsl
vec2 cellOffset = vec2(0.0, columnStagger + subPixelStagger);
vec2 subCellUV = fract(subcoord + cellOffset) * 2.0 - 1.0;  // range [-1, 1]

float mask = 1.0;
vec2 border = 1.0 - subCellUV * subCellUV * (MASK_BORDER - luma * 0.25);
mask *= border.x * border.y;
float maskStrength = smoothstep(0.0, 0.95, mask);

color += 0.005;  // make dark cells slightly visible
color.rgb *= 1.0 + (maskStrength - 1.0) * MASK_INTENSITY;
```

**Polish tips:**
- `color += 0.005` keeps black cells slightly visible (like real LEDs in off state)
- Adjust `MASK_BORDER` and `MASK_INTENSITY` uniforms for cell border thickness/darkness

---

## Woven Crochet Effect

Simulates knitted fabric with alternating rotated ellipses per cell.

### Row Offset (organic look)

Unlike LED panel (which moves pixels), crochet only offsets the pattern mask:

```glsl
vec2 normalizedPixelSize = pixelSize / resolution;
vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
vec4 color = texture(inputBuffer, uvPixel);

vec2 cellPosition = floor(uv / normalizedPixelSize);
vec2 cellUV = fract(uv / normalizedPixelSize);

float rowOffset = sin((random(vec2(0.0, uvPixel.y)) - 0.5) * 0.25);
cellUV.x += rowOffset;
```

### Alternating Rotated Ellipses

Even cells: -65 degrees, odd cells: +65 degrees.

```glsl
float isAlternate = mod(cellPosition.x, 2.0);
float angle = isAlternate == 0.0 ? radians(-65.0) : radians(65.0);

vec2 center = cellUV - 0.5;
vec2 rotated = vec2(
  center.x * cos(angle) - center.y * sin(angle),
  center.x * sin(angle) + center.y * cos(angle)
);

float aspectRatio = 1.55;
float ellipse = length(vec2(rotated.x, rotated.y * aspectRatio - 0.075));
color.rgb *= smoothstep(0.2, 1.0, 1.0 - ellipse);
```

### Polish Details

- **Noise on edges:** Apply noise to cell centers for rough/fuzzy ellipse edges
- **Stripe pattern:** Draw thin stripes within each ellipse to mimic thread texture
- **Hue shift:** Slight random hue variation per cell (thread color variation)

---

## Lego Bricks Effect

Each pixelated cell becomes a 1x1 Lego brick with a centered circular stud.

### Stud with Blinn-Phong Lighting

Create the illusion of a 3D stud using diffuse lighting on a circle:

```glsl
uniform vec2 lightPosition;  // e.g. vec2(0.3, -0.5)

// Inside mainImage, after pixelation:
vec2 cellUV = fract(uv / normalizedPixelSize);

float lighting = dot(normalize(cellUV - vec2(0.5)), lightPosition) * 0.7;
float dis = abs(distance(cellUV, vec2(0.5)) * 2.0 - 0.5);
color.rgb *= smoothstep(0.1, 0.0, dis) * lighting + 1.0;
```

**How it works:**
1. Calculate lighting direction from cell center to light position (dot product = diffuse)
2. Calculate distance from cell center, create soft circle edge with `smoothstep`
3. Multiply: pixels near the center get lighting applied, creating a shaded stud

### Color Quantization (limited Lego palette)

```glsl
float levels = 6.0;  // number of discrete color levels
color.rgb = floor(color.rgb * levels + 0.5) / levels;
```

### Cell Border (brick separation)

Reuse the border technique from LED panel:

```glsl
vec2 borderUV = fract(uv / normalizedPixelSize) * 2.0 - 1.0;
float border = 1.0 - max(abs(borderUV.x), abs(borderUV.y));
color.rgb *= smoothstep(0.0, 0.05, border);
```

### Polish Details

- Hue shift for color variety (especially on uniform backgrounds)
- Clamp min/max of color channels to keep studs visible (avoid pure black/white cells)
- Works exceptionally well on photos and paintings, not just 3D scenes

---

## Fluted & Frosted Glass

Physically-inspired distortion: a pane of fluted glass between viewer and scene.

### Core Principle

The glass shape is a sine wave `sin(uv.x * PI)`. The distortion is its derivative `cos(uv.x * PI) * PI` (maximum distortion on slopes, minimum at peaks/valleys).

### Distortion

```glsl
float fluteCount = 25.0;
float flutePosition = fract(uv.x * fluteCount + 0.5);

vec2 distortion = vec2(cos(flutePosition * PI * 2.0) * PI * 0.15, 0.0);
vec2 distortedUV = uv + distortion * distortionAmount;
```

### Surface Normal and Lighting

Convert the derivative into a normal vector for Blinn-Phong lighting:

```glsl
vec3 normal = vec3(0.0);
normal.x = cos(flutePosition * PI * 2.0) * PI * 0.15;
normal.y = 0.0;
normal.z = sqrt(1.0 - normal.x * normal.x);  // from |normal| = 1
normal = normalize(normal);

vec3 lightDir = normalize(vec3(lightPosition, 1.0));
float diffuse = max(dot(normal, lightDir), 0.0);
float specular = pow(max(dot(reflect(-lightDir, normal), vec3(0.0, 0.0, 1.0)), 0.0), 32.0);

// Use normal.xy for distortion
vec2 distortedUV = uv + normal.xy * distortionAmount;
vec4 color = texture2D(inputBuffer, distortedUV);

// Apply glass lighting
color.rgb += diffuse * 0.15 + specular * 0.3;
```

### Polish Details

- **Gaussian blur:** Add slight blur for depth (sample multiple offset UVs and average)
- **Frosting:** Add noise to UVs before sampling for frosted glass texture
- **Chromatic dispersion:** Sample R, G, B channels at slightly different UV offsets:

```glsl
float r = texture2D(inputBuffer, distortedUV + normal.xy * 0.003).r;
float g = texture2D(inputBuffer, distortedUV).g;
float b = texture2D(inputBuffer, distortedUV - normal.xy * 0.003).b;
color = vec4(r, g, b, 1.0);
```

### Variants

- **Glass bubbles:** Use circular SDFs for bubble-shaped distortions
- **Lens effect:** Single circular distortion area following cursor
- **Vertical flutes:** Swap `uv.x` for `uv.y` in the sine wave
