# Shadow Mapping for Volumetric Lighting

## Overview

Shadow mapping generates a depth texture of the scene from the light's point of view. During raymarching, sample this texture to determine if a point is occluded (in shadow).

## Setup: Light Camera + Shadow FBO

```tsx
import * as THREE from 'three';
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useFBO } from '@react-three/drei';

const SHADOW_MAP_SIZE = 512; // 256-1024 range, balance quality vs performance

const lightCamera = useMemo(() => {
  const cam = new THREE.PerspectiveCamera(90, 1.0, 0.1, 100);
  cam.fov = coneAngle; // match your light's cone angle
  return cam;
}, [coneAngle]);

const shadowFBO = useFBO(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, {
  depth: true,
  depthTexture: new THREE.DepthTexture(
    SHADOW_MAP_SIZE,
    SHADOW_MAP_SIZE,
    THREE.FloatType
  ),
});
```

## Rendering the Shadow Map

```tsx
useFrame((state) => {
  const { gl, camera, scene } = state;

  // Point light camera in the light direction
  const currentLightTargetPos = new THREE.Vector3().addVectors(
    lightPosition.current,
    lightDirection.current
  );
  lightCamera.lookAt(currentLightTargetPos);
  lightCamera.position.copy(lightPosition.current);
  lightCamera.updateMatrixWorld();
  lightCamera.updateProjectionMatrix();

  // Render scene from light's POV into shadow FBO
  const currentRenderTarget = gl.getRenderTarget();
  gl.setRenderTarget(shadowFBO);
  gl.clear(false, true, false);
  gl.render(scene, lightCamera);

  // Restore normal rendering
  gl.setRenderTarget(currentRenderTarget);
  gl.render(scene, camera);

  // Pass matrices and shadow map to the effect
  effect.uniforms.get('shadowMap').value = shadowFBO.depthTexture;
  effect.uniforms.get('lightViewMatrix').value = lightCamera.matrixWorldInverse;
  effect.uniforms.get('lightProjectionMatrix').value = lightCamera.projectionMatrix;
});
```

## GLSL: calculateShadow Function

```glsl
uniform sampler2D shadowMap;
uniform mat4 lightViewMatrix;
uniform mat4 lightProjectionMatrix;
uniform float shadowBias;

float calculateShadow(vec3 worldPosition) {
  vec4 lightClipPos = lightProjectionMatrix * lightViewMatrix * vec4(worldPosition, 1.0);
  vec3 lightNDC = lightClipPos.xyz / lightClipPos.w;

  vec2 shadowCoord = lightNDC.xy * 0.5 + 0.5;
  float lightDepth = lightNDC.z * 0.5 + 0.5;

  // Outside light frustum => not in shadow
  if (
    shadowCoord.x < 0.0 || shadowCoord.x > 1.0 ||
    shadowCoord.y < 0.0 || shadowCoord.y > 1.0 ||
    lightDepth > 1.0
  ) {
    return 1.0;
  }

  float shadowMapDepth = texture2D(shadowMap, shadowCoord).x;

  // If current depth > shadow map depth, point is occluded
  if (lightDepth > shadowMapDepth + shadowBias) {
    return 0.0;
  }

  return 1.0;
}
```

## Integration in Raymarching Loop

Shadows are a **skip** condition, not a break. Points beyond occluded areas may still be lit:

```glsl
for (int i = 0; i < NUM_STEPS; i++) {
  vec3 samplePos = rayOrigin + rayDir * t;

  if (t > sceneDepth || t > cameraFar) break;

  float shadowFactor = calculateShadow(samplePos);
  if (shadowFactor == 0.0) {
    t += STEP_SIZE;
    continue; // skip, don't break
  }

  // ... accumulate light ...
  t += STEP_SIZE;
}
```

## Shadow Map Resolution Guidance

| Resolution | Quality | Performance | Use Case |
|-----------|---------|-------------|----------|
| 128x128 | Blocky | Fast | Prototyping |
| 256x256 | Decent | Good | Mobile/low-end |
| 512x512 | Good | Moderate | Recommended default |
| 1024x1024 | Detailed | Heavier | High-quality scenes |

Higher resolutions reduce shadow flickering artifacts, especially in scenes with many small occluders (e.g., asteroid fields).

## Omnidirectional Shadows with CubeCamera

For point lights that cast shadows in all directions, use a `CubeCamera`:

```tsx
const shadowCubeRenderTarget = useMemo(() => {
  return new THREE.WebGLCubeRenderTarget(SHADOW_MAP_SIZE, {
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    generateMipmaps: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
  });
}, []);

const shadowCubeCamera = useMemo(() => {
  return new THREE.CubeCamera(0.1, 100, shadowCubeRenderTarget);
}, [shadowCubeRenderTarget]);
```

**No CubeDepthTexture exists**, so use a custom `shadowMaterial` that writes normalized distance:

```glsl
// shadowMaterial fragment shader
uniform vec3 lightPosition;
uniform float shadowFar;
varying vec3 vWorldPosition;

void main() {
  float distance = length(vWorldPosition);
  float normalizedDistance = clamp(distance / shadowFar, 0.0, 1.0);
  gl_FragColor = vec4(normalizedDistance, 0.0, 0.0, 1.0);
}
```

Workflow: swap all scene materials with shadowMaterial, render with CubeCamera, restore originals, then sample the cube texture in the volumetric shader.
