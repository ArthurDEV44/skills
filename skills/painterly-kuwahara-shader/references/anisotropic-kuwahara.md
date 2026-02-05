# Anisotropic Kuwahara Filter

## Table of Contents
- [Overview](#overview)
- [Multi-Pass Pipeline](#multi-pass-pipeline)
- [Pass 1: Structure Tensor](#pass-1-structure-tensor)
- [Pass 2: Anisotropic Kuwahara](#pass-2-anisotropic-kuwahara)
- [Full Pipeline Integration](#full-pipeline-integration)

## Overview

The anisotropic Kuwahara filter adapts the circular kernel to the local structure of the image by squeezing and rotating it into an ellipse. This aligns brush-stroke artifacts with edge directions, mimicking how a painter follows contours.

Based on "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions" by Kyprianidis, Semmo, Kang, and Dollner.

### Isotropic vs Anisotropic

- **Isotropic**: uniform circular kernel, same in all directions
- **Anisotropic**: elliptical kernel oriented along the dominant gradient direction

## Multi-Pass Pipeline

```
Scene Render ──> Pass 1: Structure Tensor ──> Pass 2: Anisotropic Kuwahara ──> Pass 3: Final Color
                 (Sobel + tensor output)       (adapted elliptical kernel)       (quantize, tonemap, texture)
```

The structure tensor pass reads the scene, the Kuwahara pass reads BOTH the structure tensor AND the original scene.

## Pass 1: Structure Tensor

Compute the gradient structure tensor using Sobel operators to find the local orientation at each pixel.

### Sobel Matrices

```
Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
      [-2, 0, 2],           [ 0,  0,  0],
      [-1, 0, 1]]           [ 1,  2,  1]]
```

### GLSL Implementation

```glsl
varying vec2 vUv;
uniform sampler2D inputBuffer;
uniform vec4 resolution;

const mat3 Gx = mat3(-1, -2, -1, 0, 0, 0, 1, 2, 1);
const mat3 Gy = mat3(-1, 0, 1, -2, 0, 2, -1, 0, 1);

vec4 computeStructureTensor(sampler2D inputTexture, vec2 uv) {
  vec3 tx0y0 = texture2D(inputTexture, uv + vec2(-1, -1) / resolution.xy).rgb;
  vec3 tx0y1 = texture2D(inputTexture, uv + vec2(-1,  0) / resolution.xy).rgb;
  vec3 tx0y2 = texture2D(inputTexture, uv + vec2(-1,  1) / resolution.xy).rgb;
  vec3 tx1y0 = texture2D(inputTexture, uv + vec2( 0, -1) / resolution.xy).rgb;
  vec3 tx1y1 = texture2D(inputTexture, uv + vec2( 0,  0) / resolution.xy).rgb;
  vec3 tx1y2 = texture2D(inputTexture, uv + vec2( 0,  1) / resolution.xy).rgb;
  vec3 tx2y0 = texture2D(inputTexture, uv + vec2( 1, -1) / resolution.xy).rgb;
  vec3 tx2y1 = texture2D(inputTexture, uv + vec2( 1,  0) / resolution.xy).rgb;
  vec3 tx2y2 = texture2D(inputTexture, uv + vec2( 1,  1) / resolution.xy).rgb;

  vec3 Sx = Gx[0][0] * tx0y0 + Gx[1][0] * tx1y0 + Gx[2][0] * tx2y0 +
            Gx[0][1] * tx0y1 + Gx[1][1] * tx1y1 + Gx[2][1] * tx2y1 +
            Gx[0][2] * tx0y2 + Gx[1][2] * tx1y2 + Gx[2][2] * tx2y2;

  vec3 Sy = Gy[0][0] * tx0y0 + Gy[1][0] * tx1y0 + Gy[2][0] * tx2y0 +
            Gy[0][1] * tx0y1 + Gy[1][1] * tx1y1 + Gy[2][1] * tx2y1 +
            Gy[0][2] * tx0y2 + Gy[1][2] * tx1y2 + Gy[2][2] * tx2y2;

  // Structure tensor components: J = [[Jxx, Jxy], [Jxy, Jyy]]
  return vec4(dot(Sx, Sx), dot(Sy, Sy), dot(Sx, Sy), 1.0);
}

void main() {
  gl_FragColor = computeStructureTensor(inputBuffer, vUv);
}
```

### Tensor Output Channels

- **R (Jxx)**: squared x-derivative magnitude (vertical edges)
- **G (Jyy)**: squared y-derivative magnitude (horizontal edges)
- **B (Jxy)**: product of x and y derivatives (diagonal alignment)

## Pass 2: Anisotropic Kuwahara

### Step 1: Eigenvalues from Structure Tensor

The eigenvalues represent gradient strength along the dominant direction (lambda1) and orthogonal direction (lambda2).

Derived from: `det(A - lambda*I) = 0` leading to `lambda^2 - trace*lambda + det = 0`

```glsl
float Jxx = structureTensor.r;
float Jyy = structureTensor.g;
float Jxy = structureTensor.b;

float trace = Jxx + Jyy;
float determinant = Jxx * Jyy - Jxy * Jxy;

float lambda1 = trace * 0.5 + sqrt(trace * trace * 0.25 - determinant);
float lambda2 = trace * 0.5 - sqrt(trace * trace * 0.25 - determinant);
```

### Step 2: Eigenvector (Dominant Orientation)

The eigenvector for lambda1 gives the dominant gradient direction.

Derived from `(A - lambda1*I) * v = 0`:
- `(Jxx - lambda1) * vx + Jxy * vy = 0`
- Setting `vy = 1`: `vx = -Jxy / (Jxx - lambda1)`
- Multiply through: `v = (-Jxy, Jxx - lambda1)`

```glsl
vec4 getDominantOrientation(vec4 structureTensor) {
  float Jxx = structureTensor.r;
  float Jyy = structureTensor.g;
  float Jxy = structureTensor.b;

  float trace = Jxx + Jyy;
  float determinant = Jxx * Jyy - Jxy * Jxy;

  float lambda1 = trace * 0.5 + sqrt(trace * trace * 0.25 - determinant);
  float lambda2 = trace * 0.5 - sqrt(trace * trace * 0.25 - determinant);

  float jxyStrength = abs(Jxy) / (abs(Jxx) + abs(Jyy) + abs(Jxy) + 1e-7);

  vec2 v;
  if (jxyStrength > 0.0) {
    v = normalize(vec2(-Jxy, Jxx - lambda1));
  } else {
    v = vec2(0.0, 1.0); // default when no dominant direction
  }

  return vec4(normalize(v), lambda1, lambda2);
}
```

### Step 3: Anisotropy and Kernel Adaptation

```
A = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-7)
```

- A = 0: isotropic (uniform), kernel stays circular
- A -> 1: strong anisotropy, kernel becomes elongated ellipse

```glsl
float anisotropy = (orientationAndAnisotropy.z - orientationAndAnisotropy.w)
                 / (orientationAndAnisotropy.z + orientationAndAnisotropy.w + 1e-7);

float alpha = 25.0; // anisotropy intensity (paper suggests 1.0, but 10-25 works better in practice)
float scaleX = alpha / (anisotropy + alpha);
float scaleY = (anisotropy + alpha) / alpha;
```

### Step 4: Build Anisotropy Matrix

Combine the rotation (from eigenvector) with the scaling (from anisotropy) into a single 2x2 matrix applied to sample offsets:

```glsl
mat2 anisotropyMat = mat2(orientation.x, -orientation.y,
                          orientation.y,  orientation.x)
                   * mat2(scaleX, 0.0,
                          0.0,    scaleY);
```

### Complete Pass 2 Fragment Shader

```glsl
#define SECTOR_COUNT 8

uniform sampler2D inputBuffer;   // structure tensor from pass 1
uniform sampler2D originalTexture; // original scene
uniform int radius;
uniform vec4 resolution;

varying vec2 vUv;

vec3 sampleColor(vec2 offset) {
  return texture2D(originalTexture, vUv + offset / resolution.xy).rgb;
}

float polynomialWeight(float x, float y, float eta, float lambda) {
  float polyValue = (x + eta) - lambda * (y * y);
  return max(0.0, polyValue * polyValue);
}

vec4 getDominantOrientation(vec4 structureTensor) {
  float Jxx = structureTensor.r;
  float Jyy = structureTensor.g;
  float Jxy = structureTensor.b;

  float trace = Jxx + Jyy;
  float determinant = Jxx * Jyy - Jxy * Jxy;

  float lambda1 = trace * 0.5 + sqrt(trace * trace * 0.25 - determinant);
  float lambda2 = trace * 0.5 - sqrt(trace * trace * 0.25 - determinant);

  float jxyStrength = abs(Jxy) / (abs(Jxx) + abs(Jyy) + abs(Jxy) + 1e-7);

  vec2 v;
  if (jxyStrength > 0.0) {
    v = normalize(vec2(-Jxy, Jxx - lambda1));
  } else {
    v = vec2(0.0, 1.0);
  }

  return vec4(normalize(v), lambda1, lambda2);
}

void getSectorVarianceAndAverageColor(mat2 anisotropyMat, float angle, float rad, out vec3 avgColor, out float variance) {
  vec3 weightedColorSum = vec3(0.0);
  vec3 weightedSquaredColorSum = vec3(0.0);
  float totalWeight = 0.0;

  float eta = 0.1;
  float lam = 0.5;

  for (float r = 1.0; r <= rad; r += 1.0) {
    for (float a = -0.392699; a <= 0.392699; a += 0.196349) {
      vec2 sampleOffset = r * vec2(cos(angle + a), sin(angle + a));
      sampleOffset *= anisotropyMat;

      vec3 color = sampleColor(sampleOffset);
      float weight = polynomialWeight(sampleOffset.x, sampleOffset.y, eta, lam);

      weightedColorSum += color * weight;
      weightedSquaredColorSum += color * color * weight;
      totalWeight += weight;
    }
  }

  avgColor = weightedColorSum / totalWeight;
  vec3 varianceRes = (weightedSquaredColorSum / totalWeight) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec4 structureTensor = texture2D(inputBuffer, vUv);

  vec3 sectorAvgColors[SECTOR_COUNT];
  float sectorVariances[SECTOR_COUNT];

  vec4 orientationAndAnisotropy = getDominantOrientation(structureTensor);
  vec2 orientation = orientationAndAnisotropy.xy;

  float anisotropy = (orientationAndAnisotropy.z - orientationAndAnisotropy.w)
                   / (orientationAndAnisotropy.z + orientationAndAnisotropy.w + 1e-7);

  float alpha = 25.0;
  float scaleX = alpha / (anisotropy + alpha);
  float scaleY = (anisotropy + alpha) / alpha;

  mat2 anisotropyMat = mat2(orientation.x, -orientation.y,
                            orientation.y,  orientation.x)
                     * mat2(scaleX, 0.0, 0.0, scaleY);

  for (int i = 0; i < SECTOR_COUNT; i++) {
    float angle = float(i) * 6.28318 / float(SECTOR_COUNT);
    getSectorVarianceAndAverageColor(anisotropyMat, angle, float(radius), sectorAvgColors[i], sectorVariances[i]);
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
```

### Parameters

| Parameter | Type | Recommended | Notes |
|-----------|------|-------------|-------|
| `radius` | int | 4-8 | Kernel radius |
| `alpha` | float | 10-25 | Anisotropy intensity. Paper says 1.0 but 25.0 produces cleaner output |
| `eta` | float | 0.1 | Polynomial weight offset |
| `lambda` | float | 0.5 | Polynomial y-axis fall-off |

## Full Pipeline Integration

### React Three Fiber Multi-Pass Setup

Use the `Pass` class from the `postprocessing` library to create each pass:

```jsx
import { Pass } from "postprocessing";
import * as THREE from "three";

class StructureTensorPass extends Pass {
  constructor(resolution) {
    super("StructureTensorPass");
    this.renderTarget = new THREE.WebGLRenderTarget(resolution.x, resolution.y, {
      type: THREE.FloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    this.fullscreenMaterial = new THREE.ShaderMaterial({
      uniforms: {
        inputBuffer: { value: null },
        resolution: { value: new THREE.Vector4() },
      },
      vertexShader: /* standard fullscreen quad vertex shader */,
      fragmentShader: structureTensorFragmentShader,
    });
  }

  render(renderer, inputBuffer, outputBuffer) {
    this.fullscreenMaterial.uniforms.inputBuffer.value = inputBuffer.texture;
    // render to this.renderTarget
    renderer.setRenderTarget(this.renderTarget);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(this.renderToScreen ? null : outputBuffer);
  }
}

class AnisotropicKuwaharaPass extends Pass {
  constructor(structureTensorPass) {
    super("AnisotropicKuwaharaPass");
    this.fullscreenMaterial = new THREE.ShaderMaterial({
      uniforms: {
        inputBuffer: { value: null },             // structure tensor
        originalTexture: { value: null },          // original scene
        resolution: { value: new THREE.Vector4() },
        radius: { value: 4 },
      },
      vertexShader: /* standard fullscreen quad vertex shader */,
      fragmentShader: anisotropicKuwaharaFragmentShader,
    });
    this.structureTensorPass = structureTensorPass;
  }

  render(renderer, inputBuffer, outputBuffer) {
    this.fullscreenMaterial.uniforms.inputBuffer.value =
      this.structureTensorPass.renderTarget.texture;
    this.fullscreenMaterial.uniforms.originalTexture.value = inputBuffer.texture;
    // render to outputBuffer
  }
}
```

### Pass Ordering in EffectComposer

```
1. StructureTensorPass  (reads scene -> writes tensor to its own renderTarget)
2. AnisotropicKuwaharaPass (reads tensor + scene -> writes filtered image)
3. FinalColorPass (reads filtered image -> outputs with color corrections)
```

### Performance Notes

- The anisotropic version is significantly more expensive than the isotropic Papari extension
- For real-time 60fps, prefer the Papari extension with polynomial weighting at kernel size 4-6
- The anisotropic version is best for higher quality stills or lower resolution scenes
- Consider lowering the post-processing resolution (e.g., render at 0.5x) for real-time use
