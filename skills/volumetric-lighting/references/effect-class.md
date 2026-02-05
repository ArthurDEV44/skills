# Custom Postprocessing Effect Class

## pmndrs/postprocessing Effect Pattern

Volumetric lighting runs as a custom `Effect` in the pmndrs/postprocessing library, compatible with `@react-three/postprocessing`.

## Effect Class Implementation

```tsx
import { Effect, EffectAttribute } from 'postprocessing';
import * as THREE from 'three';
import fragmentShader from './volumetricLighting.frag';

class VolumetricLightingEffectImpl extends Effect {
  constructor(
    cameraFar = 500,
    projectionMatrixInverse = new THREE.Matrix4(),
    viewMatrixInverse = new THREE.Matrix4(),
    cameraPosition = new THREE.Vector3(),
    lightDirection = new THREE.Vector3(),
    lightPosition = new THREE.Vector3(),
    coneAngle = 40.0
  ) {
    const uniforms = new Map([
      ['cameraFar', new THREE.Uniform(cameraFar)],
      ['projectionMatrixInverse', new THREE.Uniform(projectionMatrixInverse)],
      ['viewMatrixInverse', new THREE.Uniform(viewMatrixInverse)],
      ['cameraPosition', new THREE.Uniform(cameraPosition)],
      ['lightDirection', new THREE.Uniform(lightDirection)],
      ['lightPosition', new THREE.Uniform(lightPosition)],
      ['coneAngle', new THREE.Uniform(coneAngle)],
      // Shadow uniforms
      ['shadowMap', new THREE.Uniform(null)],
      ['lightViewMatrix', new THREE.Uniform(new THREE.Matrix4())],
      ['lightProjectionMatrix', new THREE.Uniform(new THREE.Matrix4())],
      ['shadowBias', new THREE.Uniform(0.005)],
      // Noise + performance
      ['blueNoiseTexture', new THREE.Uniform(null)],
      ['frame', new THREE.Uniform(0)],
      ['time', new THREE.Uniform(0)],
      // Light appearance
      ['lightColor', new THREE.Uniform(new THREE.Vector3(1.0, 1.0, 1.0))],
    ]);

    super('VolumetricLightingEffect', fragmentShader, {
      attributes: EffectAttribute.DEPTH,
      uniforms,
    });

    this.uniforms = uniforms;
  }

  update(_renderer, _inputBuffer, _deltaTime) {
    this.uniforms.get('projectionMatrixInverse').value = this.projectionMatrixInverse;
    this.uniforms.get('viewMatrixInverse').value = this.viewMatrixInverse;
    this.uniforms.get('cameraPosition').value = this.cameraPosition;
    this.uniforms.get('cameraFar').value = this.cameraFar;
    this.uniforms.get('lightDirection').value = this.lightDirection;
    this.uniforms.get('lightPosition').value = this.lightPosition;
    this.uniforms.get('coneAngle').value = this.coneAngle;
  }
}
```

## Key Details

- `EffectAttribute.DEPTH` exposes the depth texture as a built-in `depthBuffer` uniform in the fragment shader
- Use `readDepth(depthBuffer, uv)` (provided by postprocessing) to sample depth
- Fragment shader must implement: `void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor)`
- Built-in uniforms available: `resolution`, `texelSize`, `cameraNear`, `cameraFar`, `aspect`, `time`, `inputBuffer`, `depthBuffer`

## React Three Fiber Integration

```tsx
import { forwardRef, useMemo } from 'react';
import { EffectComposer } from '@react-three/postprocessing';

const VolumetricLighting = forwardRef((props, ref) => {
  const effect = useMemo(() => new VolumetricLightingEffectImpl(), []);
  return <primitive ref={ref} object={effect} />;
});

// In your scene:
<EffectComposer>
  <VolumetricLighting ref={effectRef} />
  {/* Additional effects: Bloom, Noise, etc. */}
</EffectComposer>
```

## Complementary Effects

Stack with other postprocessing effects for richer results:
- **Bloom** - amplifies light/dark contrast
- **Noise/Grain** - adds texture to final render
- **ToneMapping** - controls overall exposure
- **Vignette** - darkens edges for cinematic look

## Dependencies

```json
{
  "@react-three/fiber": "^8.x || ^9.x",
  "@react-three/drei": "^9.x",
  "@react-three/postprocessing": "^2.x",
  "postprocessing": "^6.x",
  "three": ">=0.150"
}
```
