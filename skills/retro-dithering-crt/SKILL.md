---
name: retro-dithering-crt
description: Retro dithering, color quantization, and CRT post-processing shader effects for React Three Fiber and Three.js using GLSL and the pmndrs/postprocessing library. Implements ordered dithering (Bayer matrix 2x2/4x4/8x8), white noise dithering, blue noise dithering, color quantization (grayscale, full-color, custom palettes, hue-lightness based), pixelization, CRT RGB cells shadow mask (aperture grill/Schiltzmaske), screen curvature, scanlines, chromatic aberration, and UV distortion. Use when (1) Adding dithering effects to a React Three Fiber or Three.js scene, (2) Implementing retro/CRT/vintage/pixel art post-processing shaders, (3) Building ordered dithering with Bayer matrices in GLSL, (4) Creating color quantization or palette reduction effects, (5) Emulating CRT monitors with RGB cell masks, curvature, and scanlines, (6) Writing custom postprocessing Effects with pmndrs/postprocessing wrapEffect, (7) Combining dithering with pixelization for retro game aesthetics.
---

# Retro Dithering & CRT Shading

Build retro post-processing effects as custom pmndrs/postprocessing Effects for React Three Fiber and Three.js scenes. Based on techniques from Maxime Heckel's article "The Art of Dithering and Retro Shading for the Web".

## Custom Effect Setup

Extend `Effect` from `postprocessing`, use `wrapEffect` from `@react-three/postprocessing`:

```jsx
import { wrapEffect, EffectComposer } from '@react-three/postprocessing';
import { Effect } from 'postprocessing';
import { Uniform } from 'three';

const fragmentShader = `/* GLSL */`;

class RetroEffectImpl extends Effect {
  constructor({ pixelSize = 4.0, colorNum = 4.0 } = {}) {
    super('RetroEffect', fragmentShader, {
      uniforms: new Map([
        ['pixelSize', new Uniform(pixelSize)],
        ['colorNum', new Uniform(colorNum)],
      ]),
    });
  }

  update(renderer, inputBuffer, deltaTime) {
    this.uniforms.get('pixelSize').value = /* ... */;
  }
}

const RetroEffect = wrapEffect(RetroEffectImpl);

// Usage in JSX:
<EffectComposer>
  <RetroEffect />
</EffectComposer>
```

### Effect shader API (differs from regular shaders)

- `mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor)` — main fragment function, set `outputColor` instead of `gl_FragColor`
- `mainUv(inout vec2 uv)` — modify UVs before they reach `mainImage`
- Built-in variables: `inputBuffer` (scene texture), `time`, `resolution`

## Effect Building Blocks

Combine these techniques to build the desired retro look. Apply in this order:

1. **UV Distortion** (mainUv) — CRT screen shake/jitter
2. **Pixelization** — snap UVs to grid for low-res look
3. **CRT RGB Cells** — shadow mask emulating phosphor dots
4. **Chromatic Aberration** — offset RGB channels when sampling
5. **Dithering + Color Quantization** — reduce colors with dithering patterns
6. **Screen Curvature** — barrel distortion for CRT shape
7. **Scanlines** — horizontal lines across screen
8. **Bloom** — phosphor glow (via `<Bloom />` component)

### Dithering techniques

Choose one based on desired aesthetic:

| Technique | Pattern | Best for |
|-----------|---------|----------|
| White noise | Random, grainy | Film grain, analog noise |
| Ordered (Bayer) | Structured, repeating | Classic retro/game look |
| Blue noise | Even, non-repetitive | Smooth gradients, subtle effect |

See [references/dithering-techniques.md](references/dithering-techniques.md) for complete GLSL implementations of all three techniques including Bayer matrix definitions (2x2, 4x4, 8x8).

### Color quantization

Reduce the color palette to emulate older hardware:

| Method | Result |
|--------|--------|
| Nearest-neighbor | N values per channel (N^3 total colors) |
| Custom palette texture | Arbitrary hand-picked colors |
| Hue-lightness (Charlton) | Artistic palette with lightness variation |

Formula: `floor(color * (n-1) + 0.5) / (n-1)` where n = number of color levels.

See [references/color-quantization.md](references/color-quantization.md) for all quantization methods with GLSL code.

### CRT & display effects

See [references/crt-effects.md](references/crt-effects.md) for complete implementations of:
- Pixelization (UV grid snapping)
- CRT RGB Cells shadow mask (aperture grill / Schiltzmaske)
- Screen curvature (barrel distortion)
- Scanlines, chromatic aberration, UV distortion
- Bloom integration with `@react-three/postprocessing`

## Complete Retro Effect Example

```glsl
// Fragment shader combining all techniques
uniform float pixelSize;
uniform float colorNum;
uniform float curve;
uniform float maskIntensity;

const int bayerMatrix8x8[64] = int[](
  0,  32, 8,  40, 2,  34, 10, 42,
  48, 16, 56, 24, 50, 18, 58, 26,
  12, 44, 4,  36, 14, 46, 6,  38,
  60, 28, 52, 20, 62, 30, 54, 22,
  3,  35, 11, 43, 1,  33, 9,  41,
  51, 19, 59, 27, 49, 17, 57, 25,
  15, 47, 7,  39, 13, 45, 5,  37,
  63, 31, 55, 23, 61, 29, 53, 21
);

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

vec3 dither(vec2 uv, vec3 color) {
  int x = int(uv.x * resolution.x) % 8;
  int y = int(uv.y * resolution.y) % 8;
  float threshold = float(bayerMatrix8x8[y * 8 + x]) / 64.0;
  color += threshold;
  color.r = floor(color.r * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
  color.g = floor(color.g * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
  color.b = floor(color.b * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
  return color;
}

void mainUv(inout vec2 uv) {
  float shake = (noise(vec2(uv.y) * sin(time * 400.0) * 100.0) - 0.5) * 0.0025;
  uv.x += shake * 1.5;
}

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  // 1. Screen curvature
  vec2 curveUV = uv * 2.0 - 1.0;
  vec2 offset = curveUV.yx * curve;
  curveUV += curveUV * offset * offset;
  curveUV = curveUV * 0.5 + 0.5;

  // 2. Pixelization
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(curveUV / normalizedPixelSize);

  // 3. RGB Cell mask
  vec2 pixel = uvPixel * resolution;
  vec2 coord = pixel / pixelSize;
  vec2 subcoord = coord * vec2(3.0, 1.0);
  vec2 cellOffset = vec2(0.0, mod(floor(coord.x), 3.0) * 0.5);
  float ind = mod(floor(subcoord.x), 3.0);
  vec3 maskColor = vec3(ind == 0.0, ind == 1.0, ind == 2.0) * 2.0;
  vec2 cellUv = fract(subcoord + cellOffset) * 2.0 - 1.0;
  vec2 border = 1.0 - cellUv * cellUv * 2.0;
  maskColor *= border.x * border.y;
  vec2 rgbCellUV = floor(coord + cellOffset) * pixelSize / resolution;

  // 4. Chromatic aberration + sample
  vec4 color = vec4(1.0);
  vec2 spread = vec2(0.002, 0.0);
  color.r = texture2D(inputBuffer, rgbCellUV + spread).r;
  color.g = texture2D(inputBuffer, rgbCellUV).g;
  color.b = texture2D(inputBuffer, rgbCellUV - spread).b;

  // 5. Dithering + quantization
  color.rgb = dither(rgbCellUV, color.rgb);

  // 6. Apply mask
  color.rgb *= 1.0 + (maskColor - 1.0) * maskIntensity;

  // 7. Curved edges
  vec2 edge = smoothstep(0.0, 0.02, curveUV) * (1.0 - smoothstep(1.0 - 0.02, 1.0, curveUV));
  color.rgb *= edge.x * edge.y;

  // 8. Scanlines
  float lines = sin(uv.y * 2000.0 + time * 100.0);
  color.rgb *= lines + 1.0;

  outputColor = color;
}
```

## Key Considerations

- **Pixel size vs dithering**: high pixel sizes reduce available pixels for dithering patterns. Find a balance.
- **Bayer matrix size**: larger matrices (8x8) yield finer, more refined dithering. Smaller (2x2) give chunkier patterns.
- **curveUV for RGB cells**: may cause artifacts at low pixel sizes. Use original UVs for mask if needed.
- **Mask blending**: without it, output appears too dark. Use `color *= 1.0 + (maskColor - 1.0) * intensity`.
- **Blue noise textures**: set `wrapS = wrapT = THREE.RepeatWrapping` on the texture.
- **Effect merging**: effects in the same `EffectComposer` merge into a single pass (better perf than passes).
- **Error diffusion** (Floyd-Steinberg): not fragment-shader friendly due to sequential nature.

## Dependencies

```
npm install @react-three/fiber @react-three/drei @react-three/postprocessing postprocessing three
```
