# Coordinate System Transformations

## Coordinate Systems in 3D Rendering

1. **Object/Model space** - coordinates relative to the object's origin
2. **World space** - shared coordinate system between all objects
3. **View space** - coordinates relative to camera (camera at origin, looking down z-axis)
4. **Clip space** - camera-relative coordinates transformed for clipping (discarding geometry outside frustum)
5. **NDC (Normalized Device Coordinates)** - normalized clip space after perspective divide
6. **Screen space** - final 2D coordinates of the rendered frame buffer

## Transformation Chain

```
Object → (modelMatrix) → World → (viewMatrix) → View → (projectionMatrix) → Clip → (perspective divide) → NDC → Screen
```

## Screen Space to World Space (Inverse Transform)

Required for reconstructing 3D rays from 2D post-processing passes:

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

**Critical**: Matrix multiplication is NOT commutative. Always apply `projectionMatrixInverse` first, then `viewMatrixInverse`.

## World Space to Light Clip Space (Shadow Mapping)

Required for projecting a world-space point into the light camera's view for shadow lookups:

```glsl
vec4 lightClipPos = lightProjectionMatrix * lightViewMatrix * vec4(worldPosition, 1.0);
vec3 lightNDC = lightClipPos.xyz / lightClipPos.w;

vec2 shadowCoord = lightNDC.xy * 0.5 + 0.5;  // UV for shadow map sampling
float lightDepth = lightNDC.z * 0.5 + 0.5;   // depth from light's POV
```

## Required Uniforms

```glsl
uniform mat4 projectionMatrixInverse;  // camera.projectionMatrixInverse
uniform mat4 viewMatrixInverse;        // camera.matrixWorld
uniform vec3 cameraPosition;           // camera.position
uniform float cameraFar;               // camera.far
```

For shadow mapping, additionally:
```glsl
uniform mat4 lightViewMatrix;          // lightCamera.matrixWorldInverse
uniform mat4 lightProjectionMatrix;    // lightCamera.projectionMatrix
```

## PerspectiveCamera Only

These transformations apply to `THREE.PerspectiveCamera`. Orthographic cameras require different projection math.
