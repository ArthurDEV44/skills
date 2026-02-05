# Dithering Techniques Reference

## Table of Contents
- White Noise Dithering
- Ordered Dithering (Bayer Matrix)
- Blue Noise Dithering
- Bayer Matrix Definitions

## White Noise Dithering

Compare pixel luminance to a random value. Simple but noisy.

```glsl
float random(vec2 c) {
  return fract(sin(dot(c.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 whiteNoiseDither(vec2 uv, float lum) {
  vec3 color = vec3(0.0);
  if (lum < random(uv)) {
    color = vec3(0.0);
  } else {
    color = vec3(1.0);
  }
  return color;
}
```

Usage in mainImage:
```glsl
float lum = dot(vec3(0.2126, 0.7152, 0.0722), color.rgb);
color.rgb = whiteNoiseDither(uv, lum);
```

## Ordered Dithering (Bayer Matrix)

Replace random threshold with a structured Bayer Matrix for repeating patterns.

### 4x4 Bayer Matrix

```glsl
const mat4 bayerMatrix4x4 = mat4(
  0.0,  8.0,  2.0, 10.0,
  12.0, 4.0,  14.0, 6.0,
  3.0,  11.0, 1.0, 9.0,
  15.0, 7.0,  13.0, 5.0
) / 16.0;

vec3 orderedDither4x4(vec2 uv, float lum, float bias) {
  vec3 color = vec3(0.0);
  int x = int(uv.x * resolution.x) % 4;
  int y = int(uv.y * resolution.y) % 4;
  float threshold = bayerMatrix4x4[y][x];

  if (lum < threshold + bias) {
    color = vec3(0.0);
  } else {
    color = vec3(1.0);
  }
  return color;
}
```

### 8x8 Bayer Matrix (flat array)

```glsl
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

float getBayerThreshold8x8(vec2 uv) {
  int x = int(mod(uv.x * resolution.x, 8.0));
  int y = int(mod(uv.y * resolution.y, 8.0));
  return float(bayerMatrix8x8[y * 8 + x]) / 64.0;
}
```

### Generic ordered dithering with 8x8 matrix

```glsl
vec3 orderedDither(vec2 uv, vec3 color, float colorNum) {
  int x = int(uv.x * resolution.x) % 8;
  int y = int(uv.y * resolution.y) % 8;
  float threshold = float(bayerMatrix8x8[y * 8 + x]) / 64.0;

  color.rgb += threshold;
  color.r = floor(color.r * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
  color.g = floor(color.g * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
  color.b = floor(color.b * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);

  return color;
}
```

### Luminance formula

Standard luminance (Rec. 709):
```glsl
float lum = dot(vec3(0.2126, 0.7152, 0.0722), color.rgb);
```

## Blue Noise Dithering

Use a blue noise texture instead of Bayer matrix for less repetitive patterns.

```glsl
// Uniform: sampler2D uNoise (128x128 blue noise texture)
vec4 noise = texture2D(uNoise, gl_FragCoord.xy / 128.0);
float threshold = noise.r;
```

Texture setup in React:
```jsx
const texture = useTexture('/path/to/blue-noise.png');
texture.wrapS = THREE.RepeatWrapping;
texture.wrapT = THREE.RepeatWrapping;
```

Then use `threshold` the same way as the Bayer matrix value.

## Bayer Matrix Formulas

Generic Bayer Matrix formula:
```
M(2n) = 1/(2n)^2 * [[(2n)^2 * M(n), (2n)^2 * M(n) + 2],
                      [(2n)^2 * M(n) + 3, (2n)^2 * M(n) + 1]]
```

Base case: M(1) = [0]

Pre-computed values:
- **2x2**: `1/4 * [[0, 2], [3, 1]]`
- **4x4**: `1/16 * [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]`
- **8x8**: Values 0-63, divided by 64.0 (see flat array above)

Larger matrices yield finer dithering patterns. 8x8 is the most commonly used.
