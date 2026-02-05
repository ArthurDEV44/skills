---
name: painterly-kuwahara-shader
description: Painterly post-processing shaders using the Kuwahara filter for React Three Fiber, Three.js, and WebGL. Transforms 3D scenes into painting-like artwork (watercolor, gouache, oil paint, aquarelle) via edge-preserving smoothing. Covers basic Kuwahara (4-sector box), Papari extension (8-sector circular kernel), Gaussian and polynomial weighting, anisotropic Kuwahara with structure tensor, and final color passes (quantization, tone mapping, paper texture). Use when (1) implementing a painterly or painting post-processing effect in WebGL/Three.js/R3F, (2) adding a Kuwahara filter to a 3D scene or image, (3) creating watercolor, oil paint, gouache, or aquarelle shader effects, (4) building edge-preserving smoothing shaders, (5) implementing anisotropic image filtering with structure tensors, (6) adding stylized NPR (non-photorealistic rendering) post-processing to a WebGL scene.
---

# Painterly Shaders with the Kuwahara Filter

Transform WebGL/Three.js scenes into real-time paintings using the Kuwahara filter as post-processing. Based on techniques from Maxime Heckel, the Papari extension, and Kyprianidis et al.'s anisotropic filtering.

## Choosing an Implementation

| Implementation | Quality | Performance | Best For |
|---------------|---------|-------------|----------|
| Basic Kuwahara (4-sector box) | Low | Fast | Quick prototypes, subtle smoothing |
| Papari + Polynomial (8-sector circular) | Good | Good | **Most projects (recommended)** |
| Anisotropic Kuwahara (multi-pass) | Best | Expensive | High-quality stills, showcase pieces |

**Start with the Papari extension + polynomial weighting.** It balances quality and performance for real-time use. Only go anisotropic if brush strokes need to follow edge contours.

## Pipeline Overview

### Simple (Single Pass)

```
Scene ──> Kuwahara Filter Pass ──> Final Color Pass ──> Output
```

### Anisotropic (Multi-Pass)

```
Scene ──> Structure Tensor Pass ──> Anisotropic Kuwahara Pass ──> Final Color Pass ──> Output
         (Sobel gradients)          (reads tensor + scene)        (quantize, tonemap, texture)
```

## Quick Start: Papari Extension with Polynomial Weighting (Recommended)

Custom `Pass` setup for `pmndrs/postprocessing` + React Three Fiber:

```jsx
import { Pass } from "postprocessing";
import { Uniform, ShaderMaterial, Vector4 } from "three";

const kuwaharaFragment = `
#define SECTOR_COUNT 8

uniform int radius;
uniform sampler2D inputBuffer;
uniform vec4 resolution;

varying vec2 vUv;

vec3 sampleColor(vec2 offset) {
  return texture2D(inputBuffer, vUv + offset / resolution.xy).rgb;
}

float polynomialWeight(float x, float y, float eta, float lambda) {
  float polyValue = (x + eta) - lambda * (y * y);
  return max(0.0, polyValue * polyValue);
}

void getSectorVarianceAndAverageColor(float angle, float rad, out vec3 avgColor, out float variance) {
  vec3 colorSum = vec3(0.0);
  vec3 squaredColorSum = vec3(0.0);
  float weightSum = 0.0;
  float eta = 0.1;
  float lam = 0.5;

  for (float r = 1.0; r <= rad; r += 1.0) {
    for (float a = -0.392699; a <= 0.392699; a += 0.196349) {
      vec2 sampleOffset = vec2(r * cos(angle + a), r * sin(angle + a));
      vec3 color = sampleColor(sampleOffset);
      float weight = polynomialWeight(sampleOffset.x, sampleOffset.y, eta, lam);
      colorSum += color * weight;
      squaredColorSum += color * color * weight;
      weightSum += weight;
    }
  }

  avgColor = colorSum / weightSum;
  vec3 varianceRes = (squaredColorSum / weightSum) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec3 sectorAvgColors[SECTOR_COUNT];
  float sectorVariances[SECTOR_COUNT];

  for (int i = 0; i < SECTOR_COUNT; i++) {
    float angle = float(i) * 6.28318 / float(SECTOR_COUNT);
    getSectorVarianceAndAverageColor(angle, float(radius), sectorAvgColors[i], sectorVariances[i]);
  }

  float minVariance = sectorVariances[0];
  vec3 finalColor = sectorAvgColors[0];

  for (int i = 1; i < SECTOR_COUNT; i++) {
    if (sectorVariances[i] < minVariance) {
      minVariance = sectorVariances[i];
      finalColor = sectorAvgColors[i];
    }
  }

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

export class KuwaharaPass extends Pass {
  constructor({ radius = 4 } = {}) {
    super("KuwaharaPass");
    this.fullscreenMaterial = new ShaderMaterial({
      uniforms: {
        inputBuffer: new Uniform(null),
        resolution: new Uniform(new Vector4()),
        radius: new Uniform(radius),
      },
      fragmentShader: kuwaharaFragment,
    });
  }
}
```

### Key Parameters

| Parameter | Type | Default | Range | Effect |
|-----------|------|---------|-------|--------|
| `radius` | int | 4 | 2-8 | Kernel size. Higher = more painterly, less detail |
| `SECTOR_COUNT` | define | 8 | 4-8 | Number of sampling sectors |
| `eta` | float | 0.1 | - | Polynomial weight offset |
| `lambda` | float | 0.5 | - | Polynomial y-axis fall-off |

## Detailed References

- **[Basic & Papari Kuwahara](references/kuwahara-basic.md)** - Basic 4-sector box kernel, Papari circular extension, Gaussian weighting, polynomial weighting with full GLSL code
- **[Anisotropic Kuwahara](references/anisotropic-kuwahara.md)** - Structure tensor (Sobel), eigenvalue/eigenvector derivation, anisotropy matrix, complete multi-pass pipeline with R3F integration
- **[Final Color Pass](references/final-pass.md)** - Quantization, two-point color interpolation, saturation, ACES tone mapping, paper texture overlay with complete GLSL

## Core Concepts

### Why Kuwahara Preserves Edges

For each pixel, 4-8 surrounding sectors are evaluated. The sector with the lowest color variance is selected. Near edges, the uniform side always has lower variance, so the edge is preserved while details within regions are smoothed.

### Variance to Luminance

Per-channel RGB variance is reduced to a single float via luminance weighting for comparison:

```glsl
variance = dot(varianceVec3, vec3(0.299, 0.587, 0.114));
```

### Sector Angular Math (Circular Kernel)

- Each sector spans `2*PI / SECTOR_COUNT` radians
- Sampling half-angle: `PI / SECTOR_COUNT` = `0.392699` for 8 sectors
- Angular step: `~11.25 deg` = `0.196349` rad (5 samples per ring)

## Performance Tips

- Prefer polynomial weighting over Gaussian (`exp()` is expensive in loops)
- Kernel radius 4-6 is the sweet spot for real-time at 60fps
- For anisotropic: consider rendering post-processing at half resolution
- The structure tensor pass is cheap; the Kuwahara sampling pass is the bottleneck
- Use `#define SECTOR_COUNT` (not a uniform) so the compiler can unroll loops

## Academic References

- Papari, Petkov, Campisi - "Artistic Edge and Corner Enhancing Smoothing"
- Kyprianidis, Semmo, Kang, Dollner - "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions"
- Karpushin - "Anisotropic image segmentation by a gradient structure tensor"
