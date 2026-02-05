---
name: volumetric-lighting
description: "Real-time volumetric lighting with post-processing and raymarching for React Three Fiber and Three.js. Implements god rays, light beams, atmospheric fog, and shadow-casting volumetric effects as custom postprocessing Effects. Use when: (1) Adding volumetric lighting, god rays, or light shafts to a React Three Fiber or Three.js scene, (2) Implementing raymarched light effects as post-processing passes, (3) Creating atmospheric fog or cloud effects with light scattering, (4) Building shadow-mapped volumetric lights with occlusion, (5) Shaping light using SDFs (cone, cylinder, sphere, torus), (6) Setting up multi-light volumetric setups, (7) Optimizing raymarching performance with blue noise dithering, (8) Working with coordinate space transformations between screen space and world space for post-processing effects, (9) Creating custom pmndrs/postprocessing Effect classes with depth buffer access."
---

# Volumetric Lighting

Real-time volumetric lighting via post-processing raymarching in React Three Fiber. Based on techniques from Maxime Heckel's article "On Shaping Light".

## Architecture

The effect operates as a custom `Effect` (pmndrs/postprocessing) that raymarches light through the scene in screen space, using:

1. **Depth buffer** - stop rays at scene geometry (via `EffectAttribute.DEPTH`)
2. **Shadow map** - skip occluded samples (rendered from a light camera into an FBO)
3. **SDF shapes** - constrain light to cones, cylinders, spheres, etc.
4. **Phase functions** - Henyey-Greenstein scattering + Beer's Law transmittance
5. **Blue noise dithering** - fewer steps with comparable quality

## Quick Start

Minimal volumetric spotlight in R3F:

```tsx
import { Canvas } from '@react-three/fiber';
import { EffectComposer } from '@react-three/postprocessing';

<Canvas camera={{ position: [0, 2, 5], fov: 60 }}>
  <Scene />
  <EffectComposer>
    <VolumetricLighting />
  </EffectComposer>
</Canvas>
```

## Implementation Workflow

### 1. Create the Effect class

Extend `Effect` from postprocessing with `EffectAttribute.DEPTH`. Pass uniforms for camera matrices, light position/direction, and shadow map. See [references/effect-class.md](references/effect-class.md).

### 2. Write the fragment shader

Implement `mainImage(inputColor, uv, outputColor)`:
- Reconstruct world position from depth via inverse matrices
- Raymarch from camera toward each pixel
- At each step: check depth, check shadow, evaluate SDF shape, accumulate light

See [references/raymarching-loop.md](references/raymarching-loop.md) for the complete shader with all functions.

### 3. Set up shadow mapping

Create a light camera + FBO with `DepthTexture`, render the scene from the light's POV each frame, pass the depth texture and matrices to the shader. See [references/shadow-mapping.md](references/shadow-mapping.md).

### 4. Shape the light

Use SDFs to constrain light to specific volumes. See [references/sdf-light-shapes.md](references/sdf-light-shapes.md) for cone, cylinder, sphere, and torus implementations.

### 5. Add fog/atmosphere (optional)

Replace hard SDF shape factor with noise-modulated density using FBM for organic cloud-like beams.

### 6. Optimize with blue noise dithering

Offset each ray's start by a blue noise value, shift pattern per frame. Enables reducing NUM_STEPS from 250 to 50 with STEP_SIZE from 0.05 to 0.5 while maintaining quality.

## Key Concepts

### Coordinate Transforms

Screen space (2D post-processing) to world space (3D raymarching) via inverse projection and view matrices. See [references/coordinate-transforms.md](references/coordinate-transforms.md).

### Depth-Based Stopping

Compare ray distance `t` against `sceneDepth = length(worldPosition - cameraPosition)` to prevent light from bleeding through objects.

### Shadow = Skip, Not Break

When a sample is occluded, `continue` the loop (skip). Points further along the ray may still be lit. Only `break` on depth/far limits.

### PerspectiveCamera Only

All transforms assume perspective projection. Orthographic cameras require different math.

## Multi-Light & Omnidirectional

For multiple lights or omnidirectional point lights (CubeCamera), see [references/multi-light.md](references/multi-light.md).

## Performance Guidelines

| Lever | Recommended | Impact |
|-------|------------|--------|
| NUM_STEPS | 50 (with blue noise) | Fewer = faster, more banding |
| STEP_SIZE | 0.5 (with blue noise) | Larger = faster, less depth |
| Shadow map | 512x512 | Larger = fewer artifacts, heavier |
| Blue noise dithering | Always on | Eliminates banding at lower step counts |
| Complementary effects | Bloom + Noise | Masks remaining artifacts |

## References

- **[references/effect-class.md](references/effect-class.md)** - Custom Effect class setup, R3F integration, dependencies
- **[references/raymarching-loop.md](references/raymarching-loop.md)** - Complete GLSL shader: mainImage, HG phase, Beer's Law, FBM noise
- **[references/coordinate-transforms.md](references/coordinate-transforms.md)** - Screen/NDC/clip/view/world space transforms
- **[references/shadow-mapping.md](references/shadow-mapping.md)** - Light camera, FBO, calculateShadow, CubeCamera omnidirectional
- **[references/sdf-light-shapes.md](references/sdf-light-shapes.md)** - Cone, cylinder, sphere, torus SDFs + fog integration
- **[references/multi-light.md](references/multi-light.md)** - Multiple lights, dynamic light-follows-camera pattern
