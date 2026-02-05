# Volumetric Raymarching Loop

## Complete Fragment Shader Structure

```glsl
uniform sampler2D depthBuffer;
uniform float cameraFar;
uniform mat4 projectionMatrixInverse;
uniform mat4 viewMatrixInverse;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform float coneAngle;

uniform sampler2D shadowMap;
uniform mat4 lightViewMatrix;
uniform mat4 lightProjectionMatrix;
uniform float shadowBias;

uniform sampler2D blueNoiseTexture;
uniform int frame;
uniform float time;

#define NUM_STEPS 50
#define STEP_SIZE 0.5
#define FOG_DENSITY 0.15
#define LIGHT_INTENSITY 1.0
#define SCATTERING_ANISO 0.3
```

## Core Functions

### World Position Reconstruction

```glsl
vec3 getWorldPosition(vec2 uv, float depth) {
  float clipZ = depth * 2.0 - 1.0;
  vec2 ndc = uv * 2.0 - 1.0;
  vec4 clip = vec4(ndc, clipZ, 1.0);
  vec4 view = projectionMatrixInverse * clip;
  vec4 world = viewMatrixInverse * view;
  return world.xyz / world.w;
}
```

### Henyey-Greenstein Phase Function

Simulates directional scattering of light through a medium:

```glsl
float HGPhase(float mu) {
  float g = SCATTERING_ANISO;  // -1 to 1: negative=back-scatter, 0=isotropic, positive=forward-scatter
  float gg = g * g;
  float denom = 1.0 + gg - 2.0 * g * mu;
  denom = max(denom, 0.0001);
  float scatter = (1.0 - gg) / pow(denom, 1.5);
  return scatter;
}
```

### Beer's Law Transmittance

Models exponential light absorption through a medium:

```glsl
float BeersLaw(float density, float absorptionCoeff) {
  return exp(-density * absorptionCoeff);
}
```

### FBM Noise for Fog/Clouds

```glsl
const float NOISE_FREQUENCY = 0.5;
const float NOISE_AMPLITUDE = 10.0;
const int NOISE_OCTAVES = 3;

float fbm(vec3 p) {
  vec3 q = p + time * 0.5 * vec3(1.0, -0.2, -1.0);
  float g = noise(q);
  float f = 0.0;
  float scale = NOISE_FREQUENCY;
  float factor = NOISE_AMPLITUDE;

  for (int i = 0; i < NOISE_OCTAVES; i++) {
    f += scale * noise(q);
    q *= factor;
    factor += 0.21;
    scale *= 0.5;
  }
  return f;
}
```

## Complete mainImage Function

```glsl
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  float depth = readDepth(depthBuffer, uv);
  vec3 worldPosition = getWorldPosition(uv, depth);
  float sceneDepth = length(worldPosition - cameraPosition);

  vec3 rayOrigin = cameraPosition;
  vec3 rayDir = normalize(worldPosition - rayOrigin);
  vec3 lightPos = lightPosition;
  vec3 lightDir = normalize(lightDirection);

  float coneAngleRad = radians(coneAngle);
  float halfConeAngleRad = coneAngleRad * 0.5;
  float smoothEdgeWidth = 0.1;

  // Blue noise dithering for performance
  float blueNoise = texture2D(blueNoiseTexture, gl_FragCoord.xy / 1024.0).r;
  float offset = fract(blueNoise + float(frame % 32) / sqrt(0.5));
  float t = STEP_SIZE * offset;

  float transmittance = 5.0;
  vec3 accumulatedLight = vec3(0.0);

  for (int i = 0; i < NUM_STEPS; i++) {
    vec3 samplePos = rayOrigin + rayDir * t;

    // Depth-based stopping
    if (t > sceneDepth || t > cameraFar) break;

    // Shadow check (skip if occluded)
    float shadowFactor = calculateShadow(samplePos);
    if (shadowFactor == 0.0) {
      t += STEP_SIZE;
      continue;
    }

    // SDF-based light shaping (cone example)
    float sdfVal = sdCone(samplePos, lightPos, lightDir, halfConeAngleRad);
    float shapeFactor = smoothstep(0.0, -smoothEdgeWidth, sdfVal);
    // For fog: shapeFactor = -sdfVal + fbm(samplePos);

    if (shapeFactor < 0.1) {
      t += STEP_SIZE;
      continue;
    }

    // Light scattering
    float distanceToLight = length(samplePos - lightPos);
    vec3 sampleLightDir = normalize(samplePos - lightPos);
    float attenuation = exp(-0.3 * distanceToLight);
    float scatterPhase = HGPhase(dot(rayDir, -sampleLightDir));
    vec3 luminance = lightColor * LIGHT_INTENSITY * attenuation * scatterPhase;

    // Density and transmittance (Beer's Law)
    float stepDensity = max(FOG_DENSITY * shapeFactor, 0.0);
    float stepTransmittance = BeersLaw(stepDensity * STEP_SIZE, 1.0);
    transmittance *= stepTransmittance;
    accumulatedLight += luminance * transmittance * stepDensity * STEP_SIZE;

    t += STEP_SIZE;
  }

  vec3 finalColor = inputColor.rgb + accumulatedLight;
  outputColor = vec4(finalColor, 1.0);
}
```

## Performance Tuning

| Parameter | Lower Value | Higher Value | Trade-off |
|-----------|------------|--------------|-----------|
| NUM_STEPS | 30-50 | 100-250 | Quality vs speed |
| STEP_SIZE | 0.05 | 0.5 | Detail vs banding |
| FOG_DENSITY | 0.02 | 0.5 | Subtle vs thick fog |
| SCATTERING_ANISO | 0.0 | 0.8 | Uniform vs directional |
| NOISE_OCTAVES | 1-2 | 4-6 | Smooth vs detailed fog |

Blue noise dithering enables using fewer steps (50) with larger step size (0.5) while maintaining quality comparable to 250 steps at 0.05.
