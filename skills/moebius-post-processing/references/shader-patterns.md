# Moebius Post-Processing Shader Patterns

Complete GLSL shader code and React Three Fiber patterns for building Moebius-style NPR post-processing.

## Table of Contents

1. [Post-Processing Pass Setup](#1-post-processing-pass-setup)
2. [Depth Buffer GLSL](#2-depth-buffer-glsl)
3. [Normal Buffer GLSL](#3-normal-buffer-glsl)
4. [Sobel Filter Edge Detection](#4-sobel-filter-edge-detection)
5. [Hand-Drawn Outline Displacement](#5-hand-drawn-outline-displacement)
6. [Shadow Patterns](#6-shadow-patterns)
7. [Custom Specular Lighting](#7-custom-specular-lighting)
8. [Complete MoebiusPass Shader](#8-complete-moebiuspass-shader)

---

## 1. Post-Processing Pass Setup

### MoebiusPass class (extends Pass from postprocessing)

```js
import { Pass } from "postprocessing";
import { FullScreenQuad } from "three-stdlib";
import * as THREE from "three";

class MoebiusPass extends Pass {
  constructor({ depthRenderTarget, normalRenderTarget, camera }) {
    super();
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        tDepth: { value: depthRenderTarget.depthTexture },
        tNormal: { value: normalRenderTarget.texture },
        cameraNear: { value: camera.near },
        cameraFar: { value: camera.far },
        resolution: {
          value: new THREE.Vector2(window.innerWidth, window.innerHeight),
        },
      },
      vertexShader: moebiusVertexShader,
      fragmentShader: moebiusFragmentShader,
    });
    this.fsQuad = new FullScreenQuad(this.material);
  }

  dispose() {
    this.material.dispose();
    this.fsQuad.dispose();
  }

  render(renderer, writeBuffer, readBuffer) {
    this.material.uniforms.tDiffuse.value = readBuffer.texture;
    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
      this.fsQuad.render(renderer);
    } else {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
      this.fsQuad.render(renderer);
    }
  }
}
```

### JSX integration with drei Effects

```jsx
import { Effects } from "@react-three/drei";
import { extend } from "@react-three/fiber";

extend({ MoebiusPass });

// Inside scene component:
<Effects>
  <moebiusPass args={[{ depthRenderTarget, normalRenderTarget, camera }]} />
</Effects>
```

### Render target setup in scene component

```jsx
import { useFBO } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";

const depthTexture = new THREE.DepthTexture(
  window.innerWidth,
  window.innerHeight
);
const depthRenderTarget = useFBO(window.innerWidth, window.innerHeight, {
  depthTexture,
  depthBuffer: true,
});
const normalRenderTarget = useFBO();

const { camera } = useThree();

useFrame((state) => {
  const { gl, scene, camera } = state;

  // 1. Render depth
  gl.setRenderTarget(depthRenderTarget);
  gl.render(scene, camera);

  // 2. Render normals
  const originalSceneMaterial = scene.overrideMaterial;
  gl.setRenderTarget(normalRenderTarget);
  scene.matrixWorldNeedsUpdate = true;
  scene.overrideMaterial = CustomNormalMaterial;
  gl.render(scene, camera);
  scene.overrideMaterial = originalSceneMaterial;

  // 3. Reset
  gl.setRenderTarget(null);
});
```

---

## 2. Depth Buffer GLSL

Read depth from `DepthTexture` using Three.js packing utilities:

```glsl
#include <packing>

uniform sampler2D tDepth;
uniform float cameraNear;
uniform float cameraFar;

float readDepth(sampler2D depthTexture, vec2 coord) {
  float fragCoordZ = texture2D(depthTexture, coord).x;
  float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
  return viewZToOrthographicDepth(viewZ, cameraNear, cameraFar);
}
```

- Lighter pixels = further from camera
- Darker pixels = closer to camera
- Background pixels have depth ~1.0 (use `depth <= 0.99` to exclude)

---

## 3. Normal Buffer GLSL

### Custom Normal Material (with specular support)

Vertex shader:

```glsl
varying vec3 vNormal;
varying vec3 eyeVector;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vec4 mvPosition = viewMatrix * worldPos;
  gl_Position = projectionMatrix * mvPosition;

  vec3 transformedNormal = normalMatrix * normal;
  vNormal = normalize(transformedNormal);
  eyeVector = normalize(worldPos.xyz - cameraPosition);
}
```

Fragment shader (outputs normal RGB + diffuse light in alpha):

```glsl
varying vec3 vNormal;
varying vec3 eyeVector;
uniform vec3 lightPosition;

const float shininess = 600.0;
const float diffuseness = 0.5;

vec2 phong() {
  vec3 normal = normalize(vNormal);
  vec3 lightDirection = normalize(lightPosition);
  vec3 halfVector = normalize(eyeVector - lightDirection);

  float NdotL = dot(normal, lightDirection);
  float NdotH = dot(normal, halfVector);
  float NdotH2 = NdotH * NdotH;

  float kDiffuse = max(0.0, NdotL) * diffuseness;
  float kSpecular = pow(NdotH2, shininess);

  return vec2(kSpecular, kDiffuse);
}

void main() {
  vec3 color = vec3(vNormal);
  vec2 phongLighting = phong();

  float specularLight = phongLighting.x;
  float diffuseLight = phongLighting.y;

  // White specular creates edges detectable by Sobel filter
  if (specularLight >= 0.25) {
    color = vec3(1.0, 1.0, 1.0);
  }

  gl_FragColor = vec4(color, diffuseLight);
}
```

Read normal data in post-processing shader:

```glsl
vec3 normal = texture2D(tNormal, vUv).rgb;
```

---

## 4. Sobel Filter Edge Detection

### Sobel kernels

```glsl
const mat3 Sx = mat3(-1, -2, -1, 0, 0, 0, 1, 2, 1);
const mat3 Sy = mat3(-1, 0, 1, -2, 0, 2, -1, 0, 1);
```

### Luminance helper

```glsl
float luma(vec3 color) {
  const vec3 magic = vec3(0.2125, 0.7154, 0.0721);
  return dot(magic, color);
}
```

### Depth edge detection

```glsl
vec2 texel = vec2(1.0 / resolution.x, 1.0 / resolution.y);
float outlineThickness = 3.0;

// Sample 3x3 kernel around current pixel
float depth00 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(-1, 1));
float depth01 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(-1, 0));
float depth02 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(-1, -1));
float depth10 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(0, -1));
float depth11 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(0, 0));
float depth12 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(0, 1));
float depth20 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(1, -1));
float depth21 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(1, 0));
float depth22 = readDepth(tDepth, vUv + outlineThickness * texel * vec2(1, 1));

// Apply Sobel kernels
float xSobelDepth = Sx[0][0] * depth00 + Sx[1][0] * depth01 + Sx[2][0] * depth02 +
                    Sx[0][1] * depth10 + Sx[1][1] * depth11 + Sx[2][1] * depth12 +
                    Sx[0][2] * depth20 + Sx[1][2] * depth21 + Sx[2][2] * depth22;

float ySobelDepth = Sy[0][0] * depth00 + Sy[1][0] * depth01 + Sy[2][0] * depth02 +
                    Sy[0][1] * depth10 + Sy[1][1] * depth11 + Sy[2][1] * depth12 +
                    Sy[0][2] * depth20 + Sy[1][2] * depth21 + Sy[2][2] * depth22;

float gradientDepth = sqrt(pow(xSobelDepth, 2.0) + pow(ySobelDepth, 2.0));
```

### Normal edge detection (same pattern, using luma of normal texture)

```glsl
float normal00 = luma(texture2D(tNormal, vUv + outlineThickness * texel * vec2(-1, -1)).rgb);
// ... sample all 9 pixels same as depth

float xSobelNormal = /* same Sx kernel applied to normal samples */;
float ySobelNormal = /* same Sy kernel applied to normal samples */;

float gradientNormal = sqrt(pow(xSobelNormal, 2.0) + pow(ySobelNormal, 2.0));
```

### Combining depth + normal gradients

```glsl
// Weight depth more heavily to catch outer boundaries
float outline = gradientDepth * 25.0 + gradientNormal;

vec4 outlineColor = vec4(0.0, 0.0, 0.0, 1.0);
vec4 color = mix(pixelColor, outlineColor, outline);
```

---

## 5. Hand-Drawn Outline Displacement

### Hash function (for irregularity)

```glsl
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
```

### Displacement calculation

```glsl
float frequency = 0.08; // Low = elongated waves
float amplitude = 2.0;

vec2 displacement = vec2(
  hash(gl_FragCoord.xy) * sin(gl_FragCoord.y * frequency),
  hash(gl_FragCoord.xy) * cos(gl_FragCoord.x * frequency)
) * amplitude / resolution.xy;
```

### Apply displacement to all buffer reads

```glsl
// Add displacement when sampling depth and normal buffers
float depth00 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(-1, 1));
float normal00 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(-1, -1)).rgb);
// ... for all 9 kernel samples
```

---

## 6. Shadow Patterns

### Luma-based detection with depth guard

```glsl
float pixelLuma = luma(pixelColor.rgb);
float depth = readDepth(tDepth, vUv);

// Use diffuse light from normal alpha to prevent over-shading lit areas
float diffuseLight = texture2D(tNormal, vUv).a;
pixelLuma = luma(pixelColor.rgb + diffuseLight * 0.65);
```

### Pattern A: Tonal shadows

```glsl
if (pixelLuma <= 0.35 && depth <= 0.99) {
  pixelColor = vec4(0.0, 0.0, 0.0, 1.0);
}
if (pixelLuma <= 0.45 && depth <= 0.99) {
  pixelColor = pixelColor * vec4(0.25, 0.25, 0.25, 1.0);
}
if (pixelLuma <= 0.6 && depth <= 0.99) {
  pixelColor = pixelColor * vec4(0.5, 0.5, 0.5, 1.0);
}
if (pixelLuma <= 0.75 && depth <= 0.99) {
  pixelColor = pixelColor * vec4(0.7, 0.7, 0.7, 1.0);
}
```

### Pattern B: Crosshatched shadows (Moebius signature)

Three-level stripe pattern with displacement for hand-drawn feel:

```glsl
float modVal = 8.0; // Stripe spacing in pixels

// Level 1: Horizontal stripes (darkest shadows)
if (pixelLuma <= 0.35 && depth <= 0.99) {
  if (mod((vUv.y + displacement.y) * resolution.y, modVal) < 1.0) {
    pixelColor = outlineColor;
  }
}

// Level 2: Vertical stripes (medium shadows)
if (pixelLuma <= 0.55 && depth <= 0.99) {
  if (mod((vUv.x + displacement.x) * resolution.x, modVal) < 1.0) {
    pixelColor = outlineColor;
  }
}

// Level 3: Diagonal stripes (lightest shadows)
if (pixelLuma <= 0.80 && depth <= 0.99) {
  if (mod((vUv.x + displacement.x) * resolution.y + (vUv.y + displacement.y) * resolution.x, modVal) <= 1.0) {
    pixelColor = outlineColor;
  }
}
```

### Pattern C: Raster / halftone shadows

```glsl
// Circle-based pattern where radius grows with darkness
float rasterSize = 8.0;
vec2 rasterUv = floor(vUv * resolution.xy / rasterSize) * rasterSize;
vec2 rasterCenter = rasterUv + vec2(rasterSize * 0.5);
float dist = distance(gl_FragCoord.xy, rasterCenter);
float rasterRadius = (1.0 - pixelLuma) * rasterSize * 0.5;

if (dist < rasterRadius && depth <= 0.99) {
  pixelColor = outlineColor;
}
```

---

## 7. Custom Specular Lighting

The key insight: render specular as plain white in the Normal buffer so the Sobel filter detects edges around specular highlights, giving them outlined appearance.

In the CustomNormalMaterial fragment shader:

```glsl
if (specularLight >= 0.25) {
  color = vec3(1.0); // Plain white → Sobel detects edges around it
}
```

Adjustable parameters:
- `shininess` (default 600.0): Higher = smaller, sharper specular
- `diffuseness` (default 0.5): Controls diffuse light contribution
- Specular threshold (default 0.25): Lower = larger specular areas

---

## 8. Complete MoebiusPass Shader

### Vertex shader

```glsl
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

### Fragment shader (full pipeline)

```glsl
#include <packing>

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform float cameraNear;
uniform float cameraFar;
uniform vec2 resolution;

const mat3 Sx = mat3(-1, -2, -1, 0, 0, 0, 1, 2, 1);
const mat3 Sy = mat3(-1, 0, 1, -2, 0, 2, -1, 0, 1);

float readDepth(sampler2D depthTexture, vec2 coord) {
  float fragCoordZ = texture2D(depthTexture, coord).x;
  float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
  return viewZToOrthographicDepth(viewZ, cameraNear, cameraFar);
}

float luma(vec3 color) {
  const vec3 magic = vec3(0.2125, 0.7154, 0.0721);
  return dot(magic, color);
}

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 texel = vec2(1.0 / resolution.x, 1.0 / resolution.y);
  float outlineThickness = 3.0;
  vec4 outlineColor = vec4(0.0, 0.0, 0.0, 1.0);

  // Hand-drawn displacement
  float frequency = 0.08;
  float amplitude = 2.0;
  vec2 displacement = vec2(
    hash(gl_FragCoord.xy) * sin(gl_FragCoord.y * frequency),
    hash(gl_FragCoord.xy) * cos(gl_FragCoord.x * frequency)
  ) * amplitude / resolution.xy;

  vec4 pixelColor = texture2D(tDiffuse, vUv);

  // --- Sobel on Depth ---
  float depth00 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(-1, 1));
  float depth01 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(-1, 0));
  float depth02 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(-1, -1));
  float depth10 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(0, -1));
  float depth11 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(0, 0));
  float depth12 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(0, 1));
  float depth20 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(1, -1));
  float depth21 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(1, 0));
  float depth22 = readDepth(tDepth, vUv + displacement + outlineThickness * texel * vec2(1, 1));

  float xSobelDepth = Sx[0][0] * depth00 + Sx[1][0] * depth01 + Sx[2][0] * depth02 +
                      Sx[0][1] * depth10 + Sx[1][1] * depth11 + Sx[2][1] * depth12 +
                      Sx[0][2] * depth20 + Sx[1][2] * depth21 + Sx[2][2] * depth22;
  float ySobelDepth = Sy[0][0] * depth00 + Sy[1][0] * depth01 + Sy[2][0] * depth02 +
                      Sy[0][1] * depth10 + Sy[1][1] * depth11 + Sy[2][1] * depth12 +
                      Sy[0][2] * depth20 + Sy[1][2] * depth21 + Sy[2][2] * depth22;
  float gradientDepth = sqrt(pow(xSobelDepth, 2.0) + pow(ySobelDepth, 2.0));

  // --- Sobel on Normals ---
  float normal00 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(-1, -1)).rgb);
  float normal01 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(-1, 0)).rgb);
  float normal02 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(-1, 1)).rgb);
  float normal10 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(0, -1)).rgb);
  float normal11 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(0, 0)).rgb);
  float normal12 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(0, 1)).rgb);
  float normal20 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(1, -1)).rgb);
  float normal21 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(1, 0)).rgb);
  float normal22 = luma(texture2D(tNormal, vUv + displacement + outlineThickness * texel * vec2(1, 1)).rgb);

  float xSobelNormal = Sx[0][0] * normal00 + Sx[1][0] * normal01 + Sx[2][0] * normal02 +
                        Sx[0][1] * normal10 + Sx[1][1] * normal11 + Sx[2][1] * normal12 +
                        Sx[0][2] * normal20 + Sx[1][2] * normal21 + Sx[2][2] * normal22;
  float ySobelNormal = Sy[0][0] * normal00 + Sy[1][0] * normal01 + Sy[2][0] * normal02 +
                        Sy[0][1] * normal10 + Sy[1][1] * normal11 + Sy[2][1] * normal12 +
                        Sy[0][2] * normal20 + Sy[1][2] * normal21 + Sy[2][2] * normal22;
  float gradientNormal = sqrt(pow(xSobelNormal, 2.0) + pow(ySobelNormal, 2.0));

  // --- Combine outlines ---
  float outline = gradientDepth * 25.0 + gradientNormal;
  vec4 color = mix(pixelColor, outlineColor, outline);

  // --- Crosshatched shadows ---
  float diffuseLight = texture2D(tNormal, vUv).a;
  float pixelLuma = luma(color.rgb + diffuseLight * 0.65);
  float depth = readDepth(tDepth, vUv);
  float modVal = 8.0;

  if (pixelLuma <= 0.35 && depth <= 0.99) {
    if (mod((vUv.y + displacement.y) * resolution.y, modVal) < 1.0) {
      color = outlineColor;
    }
  }
  if (pixelLuma <= 0.55 && depth <= 0.99) {
    if (mod((vUv.x + displacement.x) * resolution.x, modVal) < 1.0) {
      color = outlineColor;
    }
  }
  if (pixelLuma <= 0.80 && depth <= 0.99) {
    if (mod((vUv.x + displacement.x) * resolution.y + (vUv.y + displacement.y) * resolution.x, modVal) <= 1.0) {
      color = outlineColor;
    }
  }

  gl_FragColor = color;
}
```
