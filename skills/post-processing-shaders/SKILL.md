---
name: post-processing-shaders
description: Creative post-processing shader effects for React Three Fiber and Three.js using GLSL and the pmndrs/postprocessing library. Covers pixelation, cell patterns (receipt bars, halftone, ASCII, SDFs, threshold matrices), trompe l'oeil illusions (LED panels, crochet, lego bricks, frosted glass), and dynamic/interactive effects (progressive depixelation, mouse trail). Use when (1) Creating custom post-processing effects with pmndrs/postprocessing Effect class, (2) Implementing pixelation or cell-based pattern shaders, (3) Building stylized visual effects like halftone, ASCII art, dithering, retro, or pixel art shaders, (4) Creating optical illusion effects (frosted glass, LED panel, lego, crochet, woven), (5) Adding dynamic/interactive post-processing (mouse-driven, time-based, progressive), (6) Working with UV remapping, SDF patterns, threshold matrices, or Blinn-Phong lighting in 2D post-processing, (7) Combining multiple shader techniques into creative stylized effects.
---

# Post-Processing Shaders as a Creative Medium

Creative post-processing effects for React Three Fiber using GLSL fragment shaders and `pmndrs/postprocessing`. Based on techniques from Maxime Heckel's work.

## Two Pillars of Post-Processing Shaders

1. **UV Remapping** - Distort or remap UV coordinates before sampling
2. **Cell Sculpting** - Use `fract()` to work inside individual cells and draw patterns

## Custom Effect Setup (R3F + postprocessing)

```jsx
// Effect class (MyEffect.js)
import { Uniform } from "three";
import { Effect, BlendFunction } from "postprocessing";

const fragmentShader = `
  uniform float pixelSize;

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    // Effect logic here
    outputColor = inputColor;
  }
`;

export class MyEffect extends Effect {
  constructor({ pixelSize = 8.0 } = {}) {
    super("MyEffect", fragmentShader, {
      blendFunction: BlendFunction.Normal,
      uniforms: new Map([
        ["pixelSize", new Uniform(pixelSize)]
      ])
    });
  }

  update(renderer, inputBuffer, deltaTime) {
    this.uniforms.get("pixelSize").value = this._pixelSize;
  }
}
```

```jsx
// React component wrapper
import { forwardRef, useMemo } from "react";
import { MyEffect } from "./MyEffect";

export const MyEffectComponent = forwardRef(({ pixelSize }, ref) => {
  const effect = useMemo(() => new MyEffect({ pixelSize }), [pixelSize]);
  return <primitive ref={ref} object={effect} dispose={null} />;
});
```

```jsx
// Usage in scene
import { EffectComposer } from "@react-three/postprocessing";

<EffectComposer>
  <MyEffectComponent pixelSize={8} />
</EffectComposer>
```

### Available Built-in Uniforms

All fragment shaders access: `resolution`, `texelSize`, `cameraNear`, `cameraFar`, `aspect`, `time`, `inputBuffer` (sampler2D), `depthBuffer` (sampler2D).

### Fragment Shader Signatures

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor);
void mainUv(inout vec2 uv);  // UV-only distortion
// Depth-aware variant:
void mainImage(const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor);
```

## Core Techniques

For detailed GLSL code patterns, recipes, and effect breakdowns, see the reference files:

- **[Pixelation & Cell Patterns](references/pixelation-patterns.md)** - Pixelation formula, cell UV, receipt bars, halftone dots, ASCII, SDFs, threshold matrices
- **[Trompe l'Oeil & Illusions](references/trompe-loeil-effects.md)** - Staggered LED panel, crochet, lego bricks, fluted/frosted glass
- **[Dynamic & Interactive Effects](references/dynamic-effects.md)** - Progressive depixelation, mouse trail pixelation, FBO ping-pong, time-based effects

## Quick Reference: Core Formulas

### Pixelation

```glsl
vec2 normalizedPixelSize = pixelSize / resolution;
vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
vec4 color = texture2D(inputBuffer, uvPixel);
```

### Cell UV (position within a cell)

```glsl
vec2 cellUV = fract(uv / normalizedPixelSize); // ranges [0,1] within each cell
```

### Luma (brightness)

```glsl
float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
```

### 2D Circle SDF

```glsl
float circleSDF(vec2 p) { return length(p - 0.5); }
```

### Blinn-Phong Lighting (2D illusion)

```glsl
float lighting = dot(normalize(cellUV - vec2(0.5)), lightPosition) * 0.7;
float dis = abs(distance(cellUV, vec2(0.5)) * 2.0 - 0.5);
color.rgb *= smoothstep(0.1, 0.0, dis) * lighting + 1.0;
```

### Staggering Columns

```glsl
float columnStagger = mod(floor(coord.x), 2.0) * maskStagger;
offsetUV.y += columnStagger * normalizedPixelSize.y;
```

### Fluted Glass Distortion

```glsl
float flutePosition = fract(uv.x * fluteCount + 0.5);
vec3 normal = vec3(cos(flutePosition * PI * 2.0) * PI * 0.15, 0.0, 0.0);
normal.z = sqrt(1.0 - normal.x * normal.x);
normal = normalize(normal);
vec2 distortedUV = uv + normal.xy * distortionAmount;
```

## Design Philosophy

- Start simple: pixelate, then sculpt cells
- Luma drives pattern selection (darker = more fill, lighter = less)
- Combine techniques: pixelation + SDF + lighting = complex illusion
- Pattern source can be inline GLSL, threshold matrices, or external textures
- Add polish with noise, hue shift, chromatic aberration, Gaussian blur
- Interactivity via uniforms driven by time, mouse position, or FBO textures
