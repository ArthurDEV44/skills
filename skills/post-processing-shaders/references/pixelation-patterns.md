# Pixelation & Cell Patterns

## Table of Contents

- [Pixelation Formula](#pixelation-formula)
- [Cell UV Coordinates](#cell-uv-coordinates)
- [Receipt Bar Effect](#receipt-bar-effect)
- [Halftone / Dot Pattern](#halftone--dot-pattern)
- [ASCII Effect](#ascii-effect)
- [SDF-Based Patterns](#sdf-based-patterns)
- [Threshold Matrix Patterns](#threshold-matrix-patterns)

---

## Pixelation Formula

The foundation of all cell-based post-processing effects.

```glsl
vec2 normalizedPixelSize = pixelSize / resolution;
vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
vec4 color = texture2D(inputBuffer, uvPixel);
```

**How it works:**
1. `pixelSize` = desired block size in screen pixels (use powers of 2: 2, 4, 8, 16...)
2. Normalize by `resolution` to get cell size in UV space [0,1]
3. `floor(uv / normalizedPixelSize)` snaps UVs to a grid
4. Multiply back to get the UV of the cell's origin
5. Sample the texture at this snapped UV

**Example breakdown:**
```
resolution = vec2(800, 600), pixelSize = vec2(8, 8), uv = vec2(0.374, 0.567)
normalizedPixelSize = (0.01, 0.0133)
floor((0.374, 0.567) / (0.01, 0.0133)) = floor(37.4, 42.6) = (37, 42)
uvPixel = (0.37, 0.559)
```

## Cell UV Coordinates

Get the relative position [0,1] of a pixel within its cell:

```glsl
vec2 cellUV = fract(uv / normalizedPixelSize);
```

This is the key to sculpting patterns inside each cell. `cellUV` ranges from (0,0) at the cell's top-left to (1,1) at the bottom-right.

To get the cell's grid position (integer index):

```glsl
vec2 cellPosition = floor(uv / normalizedPixelSize);
```

## Receipt Bar Effect

Horizontal black bars per cell whose width is proportional to brightness. Inspired by Japanese receipt typography.

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
  vec4 color = texture2D(inputBuffer, uvPixel);

  float luma = dot(vec3(0.2126, 0.7152, 0.0722), color.rgb);
  vec2 cellUV = fract(uv / normalizedPixelSize);

  float lineWidth = 0.0;
  if (luma > 0.0)  lineWidth = 1.0;
  if (luma > 0.3)  lineWidth = 0.7;
  if (luma > 0.5)  lineWidth = 0.5;
  if (luma > 0.7)  lineWidth = 0.3;
  if (luma > 0.9)  lineWidth = 0.1;
  if (luma > 0.99) lineWidth = 0.0;

  float yStart = 0.05;
  float yEnd = 0.95;

  if (cellUV.y > yStart && cellUV.y < yEnd && cellUV.x > 0.0 && cellUV.x < lineWidth) {
    color = vec4(0.0, 0.0, 0.0, 1.0);  // black bar
  } else {
    color = vec4(0.70, 0.74, 0.73, 1.0);  // receipt paper color
  }

  outputColor = color;
}
```

**Key pattern:** Luma thresholds map to bar widths. `cellUV.x < lineWidth` draws the bar from the left. `cellUV.y` constrains vertical extent.

## Halftone / Dot Pattern

Circles in each cell, size varies with brightness.

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
  vec4 color = texture2D(inputBuffer, uvPixel);

  float luma = dot(vec3(0.2126, 0.7152, 0.0722), color.rgb);
  vec2 cellUV = fract(uv / normalizedPixelSize);

  float dist = length(cellUV - 0.5);  // distance from cell center

  // Bright cells: large white circle centered
  if (luma > 0.5) {
    float radius = mix(0.3, 0.45, (luma - 0.5) * 2.0);
    outputColor = dist < radius ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
  }
  // Dark cells: small circle at bottom-left corner
  else {
    float radius = mix(0.05, 0.3, luma * 2.0);
    float cornerDist = length(cellUV);
    outputColor = cornerDist < radius ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
  }
}
```

## ASCII Effect

Map luma to characters from an external texture atlas.

### R3F Setup (create ASCII palette texture)

```jsx
const ASCII_CHARS = './nohamelama';

useEffect(() => {
  const CHAR_SIZE = pixelSize;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = CHAR_SIZE * ASCII_CHARS.length;
  canvas.height = CHAR_SIZE;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.font = `${CHAR_SIZE}px monospace`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';

  ASCII_CHARS.split('').forEach((char, i) => {
    ctx.fillText(char, (i + 0.5) * CHAR_SIZE, CHAR_SIZE / 2);
  });

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;

  effectRef.current.asciiTexture = texture;
  effectRef.current.charCount = [ASCII_CHARS.length, 1];
}, [pixelSize]);
```

### GLSL (sample ASCII character)

```glsl
uniform sampler2D asciiTexture;
uniform vec2 charCount;

// ...after pixelation and luma calculation:
float charIndex = clamp(floor(luma * (charCount.x - 1.0)), 0.0, charCount.x - 1.0);

vec2 asciiUV = vec2(
  (charIndex + cellUV.x) / charCount.x,
  cellUV.y
);

float character = texture2D(asciiTexture, asciiUV).r;
outputColor = vec4(vec3(character), 1.0);
```

**Pattern source flexibility:** The pattern can come from inline GLSL math, from a threshold matrix, or from an external texture (like this ASCII atlas).

## SDF-Based Patterns

Use 2D signed distance functions to draw shapes inside cells. The shape rendered is driven by luma.

### Common 2D SDFs

```glsl
float circleSDF(vec2 p) {
  return length(p - 0.5);
}

float squareSDF(vec2 p) {
  vec2 d = abs(p - 0.5) - 0.3;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float crossSDF(vec2 p) {
  vec2 d = abs(p - 0.5);
  return min(d.x, d.y) - 0.1;
}

float triangleSDF(vec2 p) {
  p -= 0.5;
  float k = sqrt(3.0);
  p.x = abs(p.x) - 0.3;
  p.y = p.y + 0.3 / k;
  if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
  p.x -= clamp(p.x, -0.6, 0.0);
  return -length(p) * sign(p.y);
}

float diamondSDF(vec2 p) {
  p = abs(p - 0.5);
  return (p.x + p.y - 0.35) * 0.707;
}
```

### SDF Pattern Rendering

```glsl
float d = circleSDF(cellUV);

if (luma > 0.2) {
  // Dark-to-mid: filled circle on white
  color = (d < 0.3) ? vec4(0.0, 0.31, 0.933, 1.0) : vec4(1.0);
}

if (luma > 0.75) {
  // Bright: inverted - white circle on blue
  color = (d < 0.3) ? vec4(1.0) : vec4(0.0, 0.31, 0.933, 1.0);
}
```

## Threshold Matrix Patterns

Define a matrix of luma thresholds. Each cell pixel is turned on/off by comparing its luma against the matrix value at that position. Enables complex patterns impossible with SDFs.

### Stripes Pattern (two matrices, switch by luma)

```glsl
const float stripesMatrix[64] = float[64](
  0.2, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 0.2,
  0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0,
  1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 1.0,
  1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2,
  0.2, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 0.2,
  0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0,
  1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 1.0,
  1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2
);

const float crossStripeMatrix[64] = float[64](
  1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0,
  0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 1.0, 0.2,
  0.2, 0.2, 1.0, 0.2, 0.2, 1.0, 0.2, 0.2,
  0.2, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 0.2,
  0.2, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 0.2,
  0.2, 0.2, 1.0, 0.2, 0.2, 1.0, 0.2, 0.2,
  0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 1.0, 0.2,
  1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0
);

int x = int(cellUV.x * 8.0);
int y = int(cellUV.y * 8.0);
int index = y * 8 + x;

if (luma < 0.6) {
  color = (stripesMatrix[index] > luma) ? vec4(1.0) : vec4(0.0, 0.31, 0.933, 1.0);
} else {
  color = (crossStripeMatrix[index] > luma) ? vec4(1.0) : vec4(0.0, 0.31, 0.933, 1.0);
}
```

### Weave / Sine Pattern (single matrix)

```glsl
const float sineMatrix[64] = float[64](
  0.99, 0.75, 0.2,  0.2,  0.2,  0.2,  0.99, 0.99,
  0.99, 0.99, 0.75, 0.2,  0.2,  0.99, 0.99, 0.75,
  0.2,  0.99, 0.99, 0.75, 0.99, 0.99, 0.2,  0.2,
  0.2,  0.2,  0.99, 0.99, 0.99, 0.2,  0.2,  0.2,
  0.2,  0.2,  0.2,  0.99, 0.99, 0.99, 0.2,  0.2,
  0.2,  0.2,  0.99, 0.99, 0.75, 0.99, 0.99, 0.2,
  0.75, 0.99, 0.99, 0.2,  0.2,  0.75, 0.99, 0.99,
  0.99, 0.99, 0.2,  0.2,  0.2,  0.2,  0.75, 0.99
);

int x = int(cellUV.x * 8.0);
int y = int(cellUV.y * 8.0);
int index = y * 8 + x;
color = (sineMatrix[index] > luma) ? vec4(1.0) : vec4(0.0, 0.31, 0.933, 1.0);
```

### Custom Threshold Matrices

Design your own 8x8 (or any size) matrix:
- Values near 0.0: pixels turn on even for very dark areas
- Values near 1.0: pixels only turn on for very bright areas
- Use multiple matrices and switch between them based on luma for richer patterns
