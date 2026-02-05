# Caustics Shader Reference

## Table of Contents
- [1. NormalMaterial](#1-normalmaterial)
- [2. CausticsComputeMaterial](#2-causticscomputematerial)
- [3. CausticsPlaneMaterial (Chromatic Aberration)](#3-causticsplanematerial)
- [4. Dynamic Normals (Vertex Displacement)](#4-dynamic-normals)
- [5. Material Setup Patterns](#5-material-setup-patterns)

---

## 1. NormalMaterial

Renders the mesh normals as color data into a render target. Use `THREE.BackSide` to capture back-face normals (important for transmissive objects like glass).

### Vertex Shader

```glsl
varying vec2 vUv;
varying vec3 vNormal;

void main() {
  vUv = uv;
  vNormal = normal * normalMatrix;

  vec4 modelPosition = modelMatrix * vec4(position, 1.0);
  vec4 viewPosition = viewMatrix * modelPosition;
  gl_Position = projectionMatrix * viewPosition;
}
```

### Fragment Shader

```glsl
varying vec3 vNormal;

void main() {
  gl_FragColor = vec4(vNormal, 1.0);
}
```

### Three.js Class

```js
import * as THREE from 'three'

class NormalMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: `/* vertex shader above */`,
      fragmentShader: `/* fragment shader above */`,
    })
  }
}
```

---

## 2. CausticsComputeMaterial

Core of the effect. Uses the Evan Wallace technique: refract a light ray through the normal, compute area before/after refraction with `dFdx`/`dFdy`, derive intensity from the ratio.

### Fragment Shader

```glsl
uniform sampler2D uTexture;
uniform vec3 uLight;
uniform float uIntensity;

varying vec2 vUv;
varying vec3 vPosition;

void main() {
  vec2 uv = vUv;

  vec3 normalTexture = texture2D(uTexture, uv).rgb;
  vec3 normal = normalize(normalTexture);
  vec3 lightDir = normalize(uLight);

  // Refract light through surface. 1.0/1.25 = IOR ratio (air to glass ~1.5 range)
  vec3 ray = refract(lightDir, normal, 1.0 / 1.25);

  vec3 newPos = vPosition.xyz + ray;
  vec3 oldPos = vPosition.xyz;

  // Approximate surface area via partial derivatives
  float lightArea = length(dFdx(oldPos)) * length(dFdy(oldPos));
  float newLightArea = length(dFdx(newPos)) * length(dFdy(newPos));

  // Ratio: converging rays -> brighter, diverging -> dimmer
  float value = lightArea / newLightArea;

  // Clamp to [0,1] to avoid artifacts with MeshTransmissionMaterial
  float scale = clamp(value, 0.0, 1.0) * uIntensity;

  // Square for more contrast (bright brighter, dim dimmer)
  scale *= scale;

  gl_FragColor = vec4(vec3(scale), 1.0);
}
```

### Vertex Shader

```glsl
varying vec2 vUv;
varying vec3 vPosition;

void main() {
  vUv = uv;
  vPosition = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

### Three.js Class

```js
class CausticsComputeMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        uTexture: { value: null },
        uLight: { value: new THREE.Vector3(-10, 13, -10) },
        uIntensity: { value: 0.5 },
      },
      vertexShader: `/* vertex shader above */`,
      fragmentShader: `/* fragment shader above */`,
    })
  }
}
```

### Key Parameters

- **IOR ratio** (`1.0 / 1.25`): Adjust for different materials. Glass ~1.5, water ~1.33. Lower denominator = subtler refraction.
- **uIntensity**: Multiplier for brightness. Start at 0.5, tune with leva.
- **Clamping**: Always clamp to `[0,1]` before squaring. Without clamping, viewing the caustic plane through `MeshTransmissionMaterial` produces visual glitches.

---

## 3. CausticsPlaneMaterial

Applies chromatic aberration to the computed caustics texture for a natural, colorful look with slight blur.

### Fragment Shader

```glsl
uniform sampler2D uTexture;
uniform float uAberration;

varying vec2 vUv;

const int SAMPLES = 16;

float random(vec2 p) {
  return fract(sin(dot(p.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 sat(vec3 rgb, float adjustment) {
  const vec3 W = vec3(0.2125, 0.7154, 0.0721);
  vec3 intensity = vec3(dot(rgb, W));
  return mix(intensity, rgb, adjustment);
}

void main() {
  vec2 uv = vUv;
  vec3 refractCol = vec3(0.0);

  float flip = -0.5;

  for (int i = 0; i < SAMPLES; i++) {
    float noiseIntensity = 0.01;
    float noise = random(uv) * noiseIntensity;
    float slide = float(i) / float(SAMPLES) * 0.1 + noise;

    // Flip direction each iteration to avoid stripe artifacts and add blur
    float mult = i % 2 == 0 ? 1.0 : -1.0;
    flip *= mult;

    vec2 dir = i % 2 == 0 ? vec2(flip, 0.0) : vec2(0.0, flip);

    // Shift each channel by different amounts for chromatic aberration
    refractCol.r += texture2D(uTexture, uv + (uAberration * slide * dir * 1.0)).r;
    refractCol.g += texture2D(uTexture, uv + (uAberration * slide * dir * 2.0)).g;
    refractCol.b += texture2D(uTexture, uv + (uAberration * slide * dir * 3.0)).b;
  }

  refractCol /= float(SAMPLES);
  refractCol = sat(refractCol, 1.265);

  gl_FragColor = vec4(refractCol, 1.0);
}
```

### Three.js Class

```js
class CausticsPlaneMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        uTexture: { value: null },
        uAberration: { value: 0.03 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `/* fragment shader above */`,
    })
  }
}
```

### Chromatic Aberration Tips

- The `flip` technique alternates the direction of color channel offsets each loop iteration, which: (a) eliminates visible stripe artifacts, (b) distributes blur evenly, (c) makes the effect look more natural.
- `uAberration` controls strength. Start at `0.03`, increase for more colorful spread.
- `SAMPLES = 16` is a good balance between quality and performance. Reduce for mobile.
- The `sat()` function boosts saturation slightly (`1.265`) for more vivid caustics.

### Blending Setup (Critical)

```js
const mat = new CausticsPlaneMaterial()
mat.transparent = true
mat.blending = THREE.CustomBlending
mat.blendSrc = THREE.OneFactor
mat.blendDst = THREE.SrcAlphaFactor
```

Without this, the caustic plane shows a black frame around the pattern instead of blending with the ground.

---

## 4. Dynamic Normals

For animated/dynamic caustics, apply vertex displacement (noise) and recompute normals on the fly using Marco Fugaro's neighboring-vertex technique.

### Updated Normal Material Vertex Shader

```glsl
uniform float uFrequency;
uniform float uAmplitude;
uniform float time;
uniform bool uDisplace;

varying vec2 vUv;
varying vec3 vNormal;

// Include cnoise (Perlin 3D noise) definition here...

vec3 orthogonal(vec3 v) {
  return normalize(
    abs(v.x) > abs(v.z)
      ? vec3(-v.y, v.x, 0.0)
      : vec3(0.0, -v.z, v.y)
  );
}

float displace(vec3 point) {
  if (uDisplace) {
    return cnoise(point * uFrequency + vec3(time)) * uAmplitude;
  }
  return 0.0;
}

void main() {
  vUv = uv;

  vec3 displacedPosition = position + normal * displace(position);
  vec4 modelPosition = modelMatrix * vec4(displacedPosition, 1.0);
  vec4 viewPosition = viewMatrix * modelPosition;
  gl_Position = projectionMatrix * viewPosition;

  // Recompute normals from displaced neighbors
  float offset = 4.0 / 256.0;
  vec3 tangent = orthogonal(normal);
  vec3 bitangent = normalize(cross(normal, tangent));
  vec3 neighbour1 = position + tangent * offset;
  vec3 neighbour2 = position + bitangent * offset;
  vec3 displacedNeighbour1 = neighbour1 + normal * displace(neighbour1);
  vec3 displacedNeighbour2 = neighbour2 + normal * displace(neighbour2);

  vec3 displacedTangent = displacedNeighbour1 - displacedPosition;
  vec3 displacedBitangent = displacedNeighbour2 - displacedPosition;

  vec3 displacedNormal = normalize(cross(displacedTangent, displacedBitangent));

  vNormal = displacedNormal * normalMatrix;
}
```

### Wiring Dynamic Uniforms in useFrame

```js
useFrame((state) => {
  const { gl, clock } = state

  // Normal material: apply displacement for normal capture
  mesh.current.material = normalMaterial
  mesh.current.material.side = THREE.BackSide
  mesh.current.material.uniforms.time.value = clock.elapsedTime
  mesh.current.material.uniforms.uDisplace.value = true
  mesh.current.material.uniforms.uAmplitude.value = amplitude
  mesh.current.material.uniforms.uFrequency.value = frequency

  gl.setRenderTarget(normalRenderTarget)
  gl.render(mesh.current, normalCamera)

  // Restore original material and apply same displacement
  mesh.current.material = originalMaterial
  mesh.current.material.uniforms.time.value = clock.elapsedTime
  mesh.current.material.uniforms.uDisplace.value = true
  mesh.current.material.uniforms.uAmplitude.value = amplitude
  mesh.current.material.uniforms.uFrequency.value = frequency

  // ... rest of pipeline
})
```

### Why Recompute Normals?

When vertices are displaced in a vertex shader, the original normals remain unchanged. The normals must be recalculated based on the new vertex positions for the caustic pattern to reflect the displacement. The technique samples two neighboring points, displaces them the same way, then computes the cross product of the displaced tangent and bitangent to get the new normal.

---

## 5. Material Setup Patterns

### Using FullScreenQuad for Compute Pass

```js
import { FullScreenQuad } from '@react-three/drei'

const [causticsQuad] = useState(() => new FullScreenQuad())

// In useFrame:
causticsQuad.material = causticsComputeMaterial
causticsQuad.material.uniforms.uTexture.value = normalRenderTarget.texture
causticsQuad.material.uniforms.uLight.value = lightVec
causticsQuad.material.uniforms.uIntensity.value = intensity

gl.setRenderTarget(causticsComputeRenderTarget)
causticsQuad.render(gl)
```

The FullScreenQuad is not added to the scene. It is instantiated standalone and rendered directly into a render target inside `useFrame`.

### Render Target Setup

```js
import { useFBO } from '@react-three/drei'

const normalRenderTarget = useFBO(2000, 2000, {})
const causticsComputeRenderTarget = useFBO(2000, 2000, {})
```

Resolution of 2000x2000 provides good quality. Reduce for performance on mobile.

### Dedicated Camera for Normal Capture

```js
const [normalCamera] = useState(
  () => new THREE.PerspectiveCamera(65, 1, 0.1, 1000)
)

// In useFrame:
const bounds = new THREE.Box3().setFromObject(mesh.current, true)
normalCamera.position.set(light.x, light.y, light.z)
normalCamera.lookAt(bounds.getCenter(new THREE.Vector3()))
normalCamera.up.set(0, 1, 0)  // Lock up vector to prevent rotation issues
```
