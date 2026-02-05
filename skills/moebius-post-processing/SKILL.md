---
name: moebius-post-processing
description: "Moebius-style (Jean Giraud) NPR post-processing shaders for React Three Fiber and Three.js. Hand-drawn outlines via Sobel filter on depth/normal buffers, crosshatched/tonal/raster shadow patterns, outlined specular highlights with Blinn-Phong. Use when: (1) Adding Moebius, comic, manga, sketch, or hand-drawn outlines to R3F/Three.js scenes, (2) Sobel filter edge detection post-processing on depth and normal buffers, (3) Crosshatched or striped shadow patterns in GLSL, (4) Custom post-processing passes (MoebiusPass + Pass + FullScreenQuad), (5) NPR non-photorealistic rendering stylization, (6) Hand-drawn wiggle/displacement on outlines, (7) Outlined specular via custom Normal material, (8) DepthTexture, scene.overrideMaterial, render-target pipelines in R3F."
---

# Moebius-Style Post-Processing Shaders

Reproduce Jean Giraud's (Moebius) distinctive hand-drawn art style as a GLSL post-processing pass for React Three Fiber / Three.js scenes. Based on techniques from Maxime Heckel's article and @UselessGameDev's video deconstruction.

## Dependencies

```
three, @react-three/fiber, @react-three/drei, postprocessing, three-stdlib
```

## Architecture Overview

The pipeline renders the scene 3 times per frame into separate buffers, then composites in a single post-processing pass:

```
Scene ──┬── Depth Buffer (DepthTexture via useFBO)
        ├── Normal Buffer (scene.overrideMaterial → CustomNormalMaterial)
        └── Color Buffer (standard render → tDiffuse)
                │
        MoebiusPass (fragment shader):
          1. Sobel filter on depth → outer outlines
          2. Sobel filter on normals → inner outlines
          3. Combine gradients → full outline
          4. Apply hand-drawn displacement to outlines
          5. Detect shadow regions via luma thresholds
          6. Apply shadow pattern (crosshatched/tonal/raster)
          7. Output final composited frame
```

## Workflow

### Step 1: Set up render targets

Create two FBOs: one with `DepthTexture` + `depthBuffer: true`, one for normals.

```jsx
const depthTexture = new THREE.DepthTexture(width, height);
const depthRenderTarget = useFBO(width, height, { depthTexture, depthBuffer: true });
const normalRenderTarget = useFBO();
```

### Step 2: Render depth + normals each frame

In `useFrame`, render scene to depth target, then override scene material with `CustomNormalMaterial` and render to normal target.

### Step 3: Create MoebiusPass

Extend `Pass` from `postprocessing`. Use `FullScreenQuad` from `three-stdlib` with a `ShaderMaterial`. Pass `tDiffuse` (readBuffer), `tDepth`, `tNormal`, `cameraNear`, `cameraFar`, `resolution` as uniforms.

### Step 4: Wire into JSX

Use `@react-three/drei`'s `<Effects>` component with `extend({ MoebiusPass })`.

### Step 5: Implement the fragment shader

The fragment shader pipeline:

1. **Read depth** with `readDepth()` using `#include <packing>`, `perspectiveDepthToViewZ`, `viewZToOrthographicDepth`
2. **Sobel filter** (3x3 convolution) on both depth and normal buffers using `Sx` and `Sy` kernels
3. **Combine** depth gradient (weighted x25) + normal gradient for full outline
4. **Displace** all buffer reads with `hash() * sin/cos` for hand-drawn wiggle
5. **Detect shadows** via `luma()` of pixel color + diffuse light from normal alpha
6. **Apply pattern** (crosshatched stripes via `mod()`) with depth guard (`depth <= 0.99`)

## Key GLSL Building Blocks

### Sobel kernels
```glsl
const mat3 Sx = mat3(-1, -2, -1, 0, 0, 0, 1, 2, 1);
const mat3 Sy = mat3(-1, 0, 1, -2, 0, 2, -1, 0, 1);
```

### Luminance
```glsl
float luma(vec3 color) {
  return dot(vec3(0.2125, 0.7154, 0.0721), color);
}
```

### Displacement for hand-drawn feel
```glsl
vec2 displacement = vec2(
  hash(gl_FragCoord.xy) * sin(gl_FragCoord.y * 0.08),
  hash(gl_FragCoord.xy) * cos(gl_FragCoord.x * 0.08)
) * 2.0 / resolution.xy;
```

### Crosshatched shadows (3 levels)
```glsl
// Horizontal (darkest) → Vertical (medium) → Diagonal (lightest)
mod((vUv.y + d.y) * res.y, 8.0) < 1.0  // luma <= 0.35
mod((vUv.x + d.x) * res.x, 8.0) < 1.0  // luma <= 0.55
mod((vUv.x + d.x) * res.y + (vUv.y + d.y) * res.x, 8.0) <= 1.0  // luma <= 0.80
```

### Specular in Normal material
Render specular as `vec3(1.0)` (white) in the custom Normal material so the Sobel filter detects edges around highlights, giving them outlined appearance.

## Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `outlineThickness` | 3.0 | Sobel kernel spread (pixels) |
| `frequency` | 0.08 | Wiggle frequency (lower = longer waves) |
| `amplitude` | 2.0 | Wiggle strength |
| `modVal` | 8.0 | Shadow stripe spacing (pixels) |
| `depthWeight` | 25.0 | Depth gradient multiplier vs normal gradient |
| `shininess` | 600.0 | Specular tightness in Normal material |
| `diffuseness` | 0.5 | Diffuse light factor in Normal material |
| Luma thresholds | 0.35/0.55/0.80 | Shadow pattern density breakpoints |

## References

- **Complete shader code and all patterns**: See [references/shader-patterns.md](references/shader-patterns.md)
- **Workarounds for displaced meshes, per-mesh overrides, style recipes**: See [references/workarounds.md](references/workarounds.md)

## Common Pitfalls

- **Missing inner outlines**: Normal gradient alone may miss edges when surfaces are coplanar from camera's view. Increase `depthWeight`.
- **Background gets shaded**: Always guard shadow patterns with `depth <= 0.99` to exclude background pixels.
- **Displaced mesh outlines missing**: `scene.overrideMaterial` ignores vertex displacement. Use per-mesh traversal with custom Normal material that replicates displacement. See [references/workarounds.md](references/workarounds.md).
- **Specular not outlined**: Specular must be rendered as white in the Normal buffer (not in the post-processing pass) so the Sobel filter can detect its edges.
- **Flat-looking outlines**: Add displacement (`hash + sin/cos`) to all depth/normal buffer reads for hand-drawn feel.
