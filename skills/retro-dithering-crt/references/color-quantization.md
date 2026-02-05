# Color Quantization Reference

## Table of Contents
- Nearest-neighbor quantization formula
- Grayscale quantization
- Full-color quantization
- Custom color palettes via texture sampling
- Hue-lightness based quantization (Alex Charlton method)

## Nearest-Neighbor Quantization Formula

Reduce a color channel to `n` levels:
```
floor(color * (n - 1) + 0.5) / (n - 1)
```

- n=2: black/white (2 values per channel, 2^3=8 total colors)
- n=4: 4 values per channel, 4^3=64 total colors
- n=8: 8 values per channel, 8^3=512 total colors

## Grayscale Quantization with Dithering

```glsl
vec3 ditherGrayscale(vec2 uv, float lum, float colorNum) {
  vec3 color = vec3(lum);

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

## Full-Color Quantization with Dithering

Apply quantization directly to RGB channels:

```glsl
vec3 ditherColor(vec2 uv, vec3 color, float colorNum) {
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

## Custom Color Palette via Texture Sampling

Use grayscale quantized values to sample a 1D palette texture:

```glsl
// uniform sampler2D palette; — horizontal strip of N color blocks
vec3 ditherWithPalette(vec2 uv, float lum, float numColors) {
  vec3 color = vec3(lum);

  int x = int(uv.x * resolution.x) % 8;
  int y = int(uv.y * resolution.y) % 8;
  float threshold = float(bayerMatrix8x8[y * 8 + x]) / 64.0;

  color.rgb += threshold * 0.2; // scale threshold intensity
  color.r = floor(color.r * (numColors - 1.0) + 0.5) / (numColors - 1.0);
  color.g = floor(color.g * (numColors - 1.0) + 0.5) / (numColors - 1.0);
  color.b = floor(color.b * (numColors - 1.0) + 0.5) / (numColors - 1.0);

  // Sample palette texture using quantized grayscale as U coordinate
  vec3 paletteColor = texture2D(palette, vec2(color.r, 0.5)).rgb;
  return paletteColor;
}
```

Keep the number of color blocks in the palette texture equal to `numColors`.

## Hue-Lightness Based Quantization (Alex Charlton)

Alternative technique using HSL color space for more artistic results.

### Helper functions

```glsl
// HSL <-> RGB conversions (standard implementations)
vec3 rgbToHsl(vec3 color);
vec3 hslToRgb(vec3 hsl);

// Circular hue distance (hue wraps at 0/1)
float hueDistance(float h1, float h2) {
  float diff = abs(h1 - h2);
  return min(diff, 1.0 - diff);
}
```

### Index value from 4x4 Bayer matrix

```glsl
const int indexMatrix4x4[16] = int[](
  0,  8,  2, 10,
  12, 4, 14,  6,
  3, 11,  1,  9,
  15, 7, 13,  5
);

float indexValue() {
  int x = int(mod(gl_FragCoord.x, 4.0));
  int y = int(mod(gl_FragCoord.y, 4.0));
  return float(indexMatrix4x4[x + y * 4]) / 16.0;
}
```

### Lightness quantization

```glsl
const float lightnessSteps = 4.0;

float lightnessStep(float l) {
  return floor((0.5 + l * lightnessSteps)) / lightnessSteps;
}
```

### Algorithm

1. Convert pixel color to HSL
2. Find the two closest hues from the palette
3. Compute normalized hue distance: `hueDiff = hueDistance(hsl.x, c1.x) / hueDistance(c2.x, c1.x)`
4. If `hueDiff < indexValue()`, pick closest color; else pick second closest
5. Find two closest lightness steps for the chosen color
6. Compute lightness distance: `lightnessDiff = (hsl.z - l1) / (l2 - l1)`
7. If `lightnessDiff < indexValue()`, use darker lightness; else use lighter
8. Convert back to RGB

This yields approximately `paletteSize x lightnessSteps` total perceived colors.
