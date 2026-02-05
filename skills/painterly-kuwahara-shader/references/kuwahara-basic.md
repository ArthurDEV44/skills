# Kuwahara Filter: Basic & Papari Extension

## Table of Contents
- [Basic Kuwahara Filter](#basic-kuwahara-filter)
- [Papari Circular Kernel Extension](#papari-circular-kernel-extension)
- [Gaussian Weighting](#gaussian-weighting)
- [Polynomial Weighting (Performance)](#polynomial-weighting)

## Basic Kuwahara Filter

Edge-preserving smoothing filter: for each pixel, divide a surrounding box into 4 sectors, compute mean color and variance per sector, output the mean of the sector with lowest variance.

### Principle

- Lower variance = colors in sector are close to the average (uniform region)
- Pixels near edges: the sector on the uniform side always wins
- Result: smoothing without blurring edges

### GLSL Implementation (4-Sector Box Kernel)

```glsl
#define SECTOR_COUNT 4

uniform int kernelSize;
uniform sampler2D inputBuffer;
uniform vec4 resolution;

varying vec2 vUv;

vec3 sampleColor(vec2 offset) {
  return texture2D(inputBuffer, vUv + offset / resolution.xy).rgb;
}

void getSectorVarianceAndAverageColor(vec2 offset, int boxSize, out vec3 avgColor, out float variance) {
  vec3 colorSum = vec3(0.0);
  vec3 squaredColorSum = vec3(0.0);
  float sampleCount = 0.0;

  for (int y = 0; y < boxSize; y++) {
    for (int x = 0; x < boxSize; x++) {
      vec2 sampleOffset = offset + vec2(float(x), float(y));
      vec3 color = sampleColor(sampleOffset);
      colorSum += color;
      squaredColorSum += color * color;
      sampleCount += 1.0;
    }
  }

  avgColor = colorSum / sampleCount;
  vec3 varianceRes = (squaredColorSum / sampleCount) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114)); // luminance-weighted
}

void main() {
  vec3 boxAvgColors[SECTOR_COUNT];
  float boxVariances[SECTOR_COUNT];

  getSectorVarianceAndAverageColor(vec2(-kernelSize, -kernelSize), kernelSize, boxAvgColors[0], boxVariances[0]);
  getSectorVarianceAndAverageColor(vec2(0, -kernelSize), kernelSize, boxAvgColors[1], boxVariances[1]);
  getSectorVarianceAndAverageColor(vec2(-kernelSize, 0), kernelSize, boxAvgColors[2], boxVariances[2]);
  getSectorVarianceAndAverageColor(vec2(0, 0), kernelSize, boxAvgColors[3], boxVariances[3]);

  float minVariance = boxVariances[0];
  vec3 finalColor = boxAvgColors[0];

  for (int i = 1; i < SECTOR_COUNT; i++) {
    if (boxVariances[i] < minVariance) {
      minVariance = boxVariances[i];
      finalColor = boxAvgColors[i];
    }
  }

  gl_FragColor = vec4(finalColor, 1.0);
}
```

### Parameters

| Uniform | Type | Recommended | Notes |
|---------|------|-------------|-------|
| `kernelSize` | int | 3-6 | Higher = more painterly but loses detail |

### Limitations

- Boxy artifacts at high kernel sizes
- Only 4 sectors = poor edge preservation at complex angles
- No weight differentiation within sectors

---

## Papari Circular Kernel Extension

From "Artistic Edge and Corner Enhancing Smoothing" by Papari, Petkov, and Campisi. Replace box kernel with circular kernel and increase to 8 sectors.

### Advantages Over Basic

- 8 sectors preserve edges at more angles
- Circular shape avoids boxy artifacts
- Handles higher kernel sizes gracefully

### GLSL Implementation (8-Sector Circular Kernel)

```glsl
#define SECTOR_COUNT 8

void getSectorVarianceAndAverageColor(float angle, float radius, out vec3 avgColor, out float variance) {
  vec3 colorSum = vec3(0.0);
  vec3 squaredColorSum = vec3(0.0);
  float sampleCount = 0.0;

  // Each sector spans pi/4 (0.785398 rad), sampled in steps of ~11.25 deg (0.196349 rad)
  for (float r = 1.0; r <= radius; r += 1.0) {
    for (float a = -0.392699; a <= 0.392699; a += 0.196349) {
      vec2 sampleOffset = r * vec2(cos(angle + a), sin(angle + a));
      vec3 color = sampleColor(sampleOffset);
      colorSum += color;
      squaredColorSum += color * color;
      sampleCount += 1.0;
    }
  }

  avgColor = colorSum / sampleCount;
  vec3 varianceRes = (squaredColorSum / sampleCount) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec3 sectorAvgColors[SECTOR_COUNT];
  float sectorVariances[SECTOR_COUNT];

  for (int i = 0; i < SECTOR_COUNT; i++) {
    float angle = float(i) * 6.28318 / float(SECTOR_COUNT); // 2*PI / 8
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
```

### Angular Constants Explained

- `-0.392699` to `0.392699` = half-sector span of pi/8 on each side (~22.5 deg)
- `0.196349` = angular step (~11.25 deg), 5 samples per radial ring per sector
- `6.28318 / SECTOR_COUNT` = sector angular offset (2*PI / 8 = pi/4)

---

## Gaussian Weighting

Pixels closer to the sector center contribute more to the average. This produces smoother transitions between sectors and reduces visible artifacts.

### Gaussian Function

```
G(x,y) = exp(-(x^2 + y^2) / (2 * sigma^2))
```

### GLSL Implementation

```glsl
float gaussianWeight(float distance, float sigma) {
  return exp(-(distance * distance) / (2.0 * sigma * sigma));
}

void getSectorVarianceAndAverageColor(float angle, float radius, out vec3 avgColor, out float variance) {
  vec3 weightedColorSum = vec3(0.0);
  vec3 weightedSquaredColorSum = vec3(0.0);
  float totalWeight = 0.0;
  float sigma = radius / 3.0;

  for (float r = 1.0; r <= radius; r += 1.0) {
    for (float a = -0.392699; a <= 0.392699; a += 0.196349) {
      vec2 sampleOffset = r * vec2(cos(angle + a), sin(angle + a));
      vec3 color = sampleColor(sampleOffset);
      float weight = gaussianWeight(length(sampleOffset), sigma);

      weightedColorSum += color * weight;
      weightedSquaredColorSum += color * color * weight;
      totalWeight += weight;
    }
  }

  avgColor = weightedColorSum / totalWeight;
  vec3 varianceRes = (weightedSquaredColorSum / totalWeight) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114));
}
```

### Drawback

The `exp()` call is expensive in nested loops. See Polynomial Weighting below.

---

## Polynomial Weighting

From Kyprianidis et al. "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions". Replace Gaussian with a polynomial approximation for similar quality at better performance.

### Formula

```
w(x, y) = max(0, ((x + eta) - lambda * y^2))^2
```

### GLSL Implementation

```glsl
float polynomialWeight(float x, float y, float eta, float lambda) {
  float polyValue = (x + eta) - lambda * (y * y);
  return max(0.0, polyValue * polyValue);
}

void getSectorVarianceAndAverageColor(float angle, float radius, out vec3 avgColor, out float variance) {
  vec3 colorSum = vec3(0.0);
  vec3 squaredColorSum = vec3(0.0);
  float weightSum = 0.0;

  float eta = 0.1;
  float lambda = 0.5;

  for (float r = 1.0; r <= radius; r += 1.0) {
    for (float a = -0.392699; a <= 0.392699; a += 0.196349) {
      vec2 sampleOffset = vec2(r * cos(angle + a), r * sin(angle + a));
      vec3 color = sampleColor(sampleOffset);
      float weight = polynomialWeight(sampleOffset.x, sampleOffset.y, eta, lambda);

      colorSum += color * weight;
      squaredColorSum += color * color * weight;
      weightSum += weight;
    }
  }

  avgColor = colorSum / weightSum;
  vec3 varianceRes = (squaredColorSum / weightSum) - (avgColor * avgColor);
  variance = dot(varianceRes, vec3(0.299, 0.587, 0.114));
}
```

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `eta` | 0.1 | Controls weight offset |
| `lambda` | 0.5 | Controls y-axis fall-off |

### Recommendation

The Papari extension with polynomial weighting is sufficient for most use cases. It balances quality and performance well, and produces a satisfying painterly look even at smaller kernel sizes. The anisotropic version (see [anisotropic-kuwahara.md](anisotropic-kuwahara.md)) adds directional adaptation but is more complex and computationally expensive.
