# CRT & Pixelization Effects Reference

## Table of Contents
- Pixelization (UV grid snapping)
- CRT RGB Cells Shadow Mask (Aperture Grill / Schiltzmaske)
- Screen Curvature
- Scanlines
- Chromatic Aberration
- UV Distortion (mainUv)
- Bloom Integration

## Pixelization

Snap UV coordinates to a grid to simulate lower resolution:

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);

  vec4 color = texture2D(inputBuffer, uvPixel);
  color.rgb = dither(uvPixel, color.rgb);

  outputColor = color;
}
```

- `pixelSize` uniform: controls downsampling (higher = more pixelated)
- Balance pixel size with dithering: too large and dithering patterns disappear
- Use `uvPixel` for both texture sampling and dithering coordinate

## CRT RGB Cells Shadow Mask

Emulate the aperture grill (Schiltzmaske) where each column is staggered by half a cell height.

### Implementation

```glsl
// Constants
#define MASK_BORDER 2.0

vec3 applyRGBCellMask(vec2 uv, vec3 sceneColor, float pixelSize, float maskIntensity) {
  vec2 pixel = uv * resolution;
  vec2 coord = pixel / pixelSize;
  vec2 subcoord = coord * vec2(3.0, 1.0);

  // Stagger every other column by half a cell
  vec2 cellOffset = vec2(0.0, mod(floor(coord.x), 3.0) * 0.5);

  // Determine subcell index (0=R, 1=G, 2=B)
  float ind = mod(floor(subcoord.x), 3.0);
  vec3 maskColor = vec3(ind == 0.0, ind == 1.0, ind == 2.0) * 2.0;

  // Draw subcell borders
  vec2 cellUv = fract(subcoord + cellOffset) * 2.0 - 1.0;
  vec2 border = 1.0 - cellUv * cellUv * MASK_BORDER;
  maskColor.rgb *= border.x * border.y;

  // Sample scene at cell-aligned UV
  vec2 rgbCellUV = floor(coord + cellOffset) * pixelSize / resolution;

  // Blend mask with scene color
  vec3 color = texture2D(inputBuffer, rgbCellUV).rgb;
  color *= 1.0 + (maskColor - 1.0) * maskIntensity;

  return color;
}
```

### Key details
- `subcoord = coord * vec2(3, 1)`: subdivide each cell into 3 horizontal subcells (R, G, B)
- `cellOffset`: creates vertical staggering for the Schiltzmaske pattern
- `maskColor`: assigns red/green/blue based on subcell index
- `border`: darkens edges of each subcell for realistic phosphor gaps
- `rgbCellUV`: cell-aligned UV for sampling the scene at cell resolution
- Multiply by `maskIntensity` to control how prominent the RGB mask is
- For Streifermaske (non-staggered): set `cellOffset = vec2(0.0)`

### Mask blending
Without blending, output may appear too dark. Use:
```glsl
color.rgb *= 1.0 + (maskColor - 1.0) * maskIntensity;
```
Adjust `maskIntensity` (0.0-1.0) to taste.

## Screen Curvature

Remap UVs for barrel distortion typical of CRT monitors:

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  // Convert to [-1, 1] range centered on screen
  vec2 curveUV = uv * 2.0 - 1.0;

  // Apply quadratic curvature (stronger at edges)
  vec2 offset = curveUV.yx * curve; // swap x/y for barrel distortion
  curveUV += curveUV * offset * offset;

  // Convert back to [0, 1]
  curveUV = curveUV * 0.5 + 0.5;

  // Draw curved black edges
  vec2 edge = smoothstep(0.0, 0.02, curveUV) * (1.0 - smoothstep(1.0 - 0.02, 1.0, curveUV));
  color.rgb *= edge.x * edge.y;

  // ... use curveUV for subsequent sampling
}
```

- `curve` uniform: 0.0 (flat) to ~0.5 (realistic). Higher values exaggerate the effect.
- Swapping `.yx` makes horizontal offset depend on vertical position and vice versa.
- Using curveUV for RGB cell mask may cause artifacts at low pixel sizes.

## Scanlines

Horizontal lines moving vertically across the screen:

```glsl
float lines = sin(uv.y * 2000.0 + time * 100.0);
color.rgb *= lines + 1.0;
```

- `2000.0`: controls scanline density
- `time * 100.0`: scrolling speed (set to 0 for static scanlines)
- `lines + 1.0`: shifts range from [-1,1] to [0,2], preventing full blackout

## Chromatic Aberration

Slight color channel offset simulating CRT beam misalignment:

```glsl
#define SPREAD vec2(0.002, 0.0) // horizontal offset

vec4 color = vec4(1.0);
color.r = texture2D(inputBuffer, rgbCellUV + SPREAD).r;
color.g = texture2D(inputBuffer, rgbCellUV).g;
color.b = texture2D(inputBuffer, rgbCellUV - SPREAD).b;
```

- Keep SPREAD small (0.001-0.005) for subtle effect
- Apply before dithering for best results

## UV Distortion (mainUv)

Add screen jitter/shake typical of old CRT interference:

```glsl
float random(vec2 c) {
  return fract(sin(dot(c.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(in vec2 st) {
  vec2 i = floor(st);
  vec2 f = fract(st);
  float a = random(i);
  float b = random(i + vec2(1.0, 0.0));
  float c = random(i + vec2(0.0, 1.0));
  float d = random(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void mainUv(inout vec2 uv) {
  float shake = (noise(vec2(uv.y) * sin(time * 400.0) * 100.0) - 0.5) * 0.0025;
  uv.x += shake * 1.5;
}
```

- `mainUv` runs before `mainImage` and modifies the UV coordinates passed to it
- Keep shake amplitude small (0.001-0.005) for subtle analog feel

## Bloom Integration

Add CRT phosphor glow using the Bloom component from @react-three/postprocessing:

```jsx
import { Bloom } from '@react-three/postprocessing';

<EffectComposer>
  <RetroEffect ref={effect} />
  <Bloom
    intensity={0.25}
    luminanceThreshold={0.05}
    luminanceSmoothing={0.9}
  />
</EffectComposer>
```

- Low `luminanceThreshold` (0.05) catches more of the CRT glow
- Moderate `intensity` (0.15-0.35) for subtle phosphor bleed
- Effects in the same EffectComposer get merged into a single pass
