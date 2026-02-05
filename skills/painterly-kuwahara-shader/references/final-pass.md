# Final Color Pass: Quantization, Tone Mapping & Texture

## Table of Contents
- [Color Quantization](#color-quantization)
- [Two-Point Color Interpolation](#two-point-color-interpolation)
- [Saturation Boost](#saturation-boost)
- [ACES Tone Mapping](#aces-tone-mapping)
- [Paper Texture Overlay](#paper-texture-overlay)
- [Complete Final Pass Shader](#complete-final-pass-shader)

## Color Quantization

Reduce the number of distinct color values to simulate paint's limited palette.

### Formula

```
quantized = floor(color * (n - 1) + 0.5) / (n - 1)
```

### GLSL

```glsl
vec3 grayscale = vec3(dot(color, vec3(0.299, 0.587, 0.114)));

int n = 16; // number of color levels
float x = grayscale.r;
float qn = floor(x * float(n - 1) + 0.5) / float(n - 1);
qn = clamp(qn, 0.2, 0.7); // reduce spread for painterly feel
```

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n` | 16 | Number of quantization levels. Lower = more stylized |
| clamp min | 0.2 | Avoids pure black |
| clamp max | 0.7 | Avoids pure white, keeps painterly contrast |

## Two-Point Color Interpolation

Blend original colors with grayscale quantization for painterly contrast. Dark areas get more saturated paint color, light areas tend toward white (paper showing through).

```glsl
if (qn < 0.5) {
  color = mix(vec3(0.1), color.rgb, qn * 2.0);
} else {
  color = mix(color.rgb, vec3(1.0), (qn - 0.5) * 2.0);
}
```

- `qn < 0.5`: interpolate from near-black (0.1) to the color
- `qn >= 0.5`: interpolate from the color to white (1.0)
- Emphasizes contrast between painted (dark) and unpainted (light/white) areas

## Saturation Boost

Increase color vibrancy to compensate for the smoothing/quantization wash-out:

```glsl
vec3 sat(vec3 rgb, float adjustment) {
  vec3 W = vec3(0.2125, 0.7154, 0.0721); // luminance weights
  vec3 intensity = vec3(dot(rgb, W));
  return mix(intensity, rgb, adjustment);
}

color = sat(color, 1.5); // 1.0 = no change, >1.0 = more saturated
```

## ACES Tone Mapping

The ACES filmic tone mapping curve provides a pleasing color balance and prevents blown-out highlights:

```glsl
vec3 ACESFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

color = ACESFilm(color);
```

## Paper Texture Overlay

Blend a paper/canvas texture with the filtered output to add physicality. Use a tileable paper texture sampled at UV coordinates.

```glsl
uniform sampler2D paperTexture;

vec3 paper = texture2D(paperTexture, vUv * paperScale).rgb;
// Multiply blend: darkens the image where paper has grain
color = color * mix(vec3(1.0), paper, paperStrength);
```

### Parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `paperScale` | float | 3.0-5.0 | Tiling scale of paper texture |
| `paperStrength` | float | 0.3-0.5 | How visible the paper grain is |

### Paper Texture Tips

- Use a high-res tileable watercolor paper or canvas texture
- Multiply blend preserves the color while adding grain
- Subtle values (0.2-0.4 strength) look most realistic
- Scale should match the "zoom level" of the painting effect

## Complete Final Pass Shader

```glsl
uniform sampler2D inputBuffer;
uniform sampler2D paperTexture;
uniform vec4 resolution;
uniform float saturationAmount;  // default: 1.5
uniform float paperStrength;     // default: 0.3
uniform float paperScale;        // default: 4.0
uniform int quantizeLevels;      // default: 16

varying vec2 vUv;

vec3 sat(vec3 rgb, float adjustment) {
  vec3 W = vec3(0.2125, 0.7154, 0.0721);
  vec3 intensity = vec3(dot(rgb, W));
  return mix(intensity, rgb, adjustment);
}

vec3 ACESFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 color = texture2D(inputBuffer, vUv).rgb;

  // Quantization
  vec3 grayscale = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
  float x = grayscale.r;
  float qn = floor(x * float(quantizeLevels - 1) + 0.5) / float(quantizeLevels - 1);
  qn = clamp(qn, 0.2, 0.7);

  // Two-point color interpolation
  if (qn < 0.5) {
    color = mix(vec3(0.1), color.rgb, qn * 2.0);
  } else {
    color = mix(color.rgb, vec3(1.0), (qn - 0.5) * 2.0);
  }

  // Saturation boost
  color = sat(color, saturationAmount);

  // Tone mapping
  color = ACESFilm(color);

  // Paper texture overlay
  vec3 paper = texture2D(paperTexture, vUv * paperScale).rgb;
  color = color * mix(vec3(1.0), paper, paperStrength);

  gl_FragColor = vec4(color, 1.0);
}
```

## Recommended Defaults Summary

| Effect | Parameter | Value | Visual Impact |
|--------|-----------|-------|---------------|
| Quantization | levels | 16 | Reduces color palette |
| Quantization | clamp range | 0.2-0.7 | Avoids extremes |
| Interpolation | dark base | vec3(0.1) | Near-black for shadows |
| Saturation | amount | 1.5 | Compensates for wash-out |
| Paper | strength | 0.3 | Subtle grain |
| Paper | scale | 4.0 | Natural paper feel |
