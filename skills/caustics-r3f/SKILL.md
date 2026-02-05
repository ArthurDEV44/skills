---
name: caustics-r3f
description: >-
  Real-time caustic light effects for React Three Fiber and Three.js using GLSL shaders, render targets (FBO),
  normal extraction, and Evan Wallace area-ratio technique. Simulates light refraction through transmissive/curved
  surfaces producing swirls of light on a catcher plane. Covers normal capture, caustics intensity via dFdx/dFdy,
  chromatic aberration, plane positioning/scaling from projected bounding box, and dynamic caustics with vertex
  displacement and normal recomputation. Use when: (1) Adding caustic light patterns to R3F or Three.js scenes,
  (2) Refraction-based light effects on a ground plane, (3) Custom caustics shaders with GLSL and render targets,
  (4) Dynamic/animated caustics with displacement and noise, (5) Caustics with MeshTransmissionMaterial,
  (6) Normal-based light simulation and area-ratio intensity.
---

# Caustics in React Three Fiber

Caustics are swirls of light visible when rays travel through a transmissive/curved surface (glass, water) and converge on a receiving plane after refraction. This skill implements a real-time, somewhat physically-based caustic effect using a multi-pass render pipeline.

## Architecture Overview

The pipeline has 3 stages rendered per frame in `useFrame`:

1. **Normal Capture** - Render target mesh with a normal material into FBO from the light's viewpoint
2. **Caustics Compute** - FullScreenQuad with a shader that refracts rays through normals and computes intensity via area ratios
3. **Final Plane** - Caustics plane material with chromatic aberration + additive blending

For full shader code and implementation details, see [references/shaders.md](references/shaders.md).
For the plane positioning/scaling math, see [references/plane-math.md](references/plane-math.md).

## Quick Start: Custom Caustics Component

```tsx
import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { useFBO, FullScreenQuad } from '@react-three/drei'
import * as THREE from 'three'
import { NormalMaterial } from './NormalMaterial'
import { CausticsComputeMaterial } from './CausticsComputeMaterial'
import { CausticsPlaneMaterial } from './CausticsPlaneMaterial'

const Caustics = ({ light = [-10, 13, -10], intensity = 0.5, aberration = 0.03 }) => {
  const mesh = useRef()
  const causticsPlane = useRef()

  const normalRenderTarget = useFBO(2000, 2000, {})
  const causticsComputeRenderTarget = useFBO(2000, 2000, {})

  const [normalCamera] = useState(() => new THREE.PerspectiveCamera(65, 1, 0.1, 1000))
  const [normalMaterial] = useState(() => new NormalMaterial())
  const [causticsComputeMaterial] = useState(() => new CausticsComputeMaterial())
  const [causticsQuad] = useState(() => new FullScreenQuad())

  const [causticsPlaneMaterial] = useState(() => {
    const mat = new CausticsPlaneMaterial()
    mat.transparent = true
    mat.blending = THREE.CustomBlending
    mat.blendSrc = THREE.OneFactor
    mat.blendDst = THREE.SrcAlphaFactor
    return mat
  })

  useFrame((state) => {
    const { gl, clock } = state
    const bounds = new THREE.Box3().setFromObject(mesh.current, true)
    const lightVec = new THREE.Vector3(...light)

    // Position camera at light, look at mesh center
    normalCamera.position.copy(lightVec)
    normalCamera.lookAt(bounds.getCenter(new THREE.Vector3()))
    normalCamera.up.set(0, 1, 0)

    // Pass 1: Capture normals
    const originalMaterial = mesh.current.material
    mesh.current.material = normalMaterial
    mesh.current.material.side = THREE.BackSide
    gl.setRenderTarget(normalRenderTarget)
    gl.render(mesh.current, normalCamera)
    mesh.current.material = originalMaterial

    // Pass 2: Compute caustics intensity
    causticsQuad.material = causticsComputeMaterial
    causticsQuad.material.uniforms.uTexture.value = normalRenderTarget.texture
    causticsQuad.material.uniforms.uLight.value = lightVec
    causticsQuad.material.uniforms.uIntensity.value = intensity
    gl.setRenderTarget(causticsComputeRenderTarget)
    causticsQuad.render(gl)

    // Pass 3: Apply to plane with chromatic aberration
    causticsPlane.current.material = causticsPlaneMaterial
    causticsPlane.current.material.uniforms.uTexture.value = causticsComputeRenderTarget.texture
    causticsPlane.current.material.uniforms.uAberration.value = aberration

    // Position and scale the plane (see references/plane-math.md)
    // ...

    gl.setRenderTarget(null)
  })

  return (
    <>
      <mesh ref={mesh} position={[0, 6.5, 0]}>
        <torusKnotGeometry args={[10, 3, 16, 100]} />
        <meshTransmissionMaterial backside />
      </mesh>
      <mesh ref={causticsPlane} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry />
      </mesh>
    </>
  )
}
```

## Key Concepts

### Area-Ratio Intensity (Evan Wallace Technique)

Caustic brightness = `oldArea / newArea` where areas are computed via partial derivatives of vertex positions before/after refraction. Converging rays (smaller newArea) = brighter. Diverging rays (larger newArea) = dimmer.

### Normal Capture Camera

Place at light position, `lookAt` mesh bounds center. Lock `camera.up = (0,1,0)` to prevent unwanted rotations.

### Blending

Use `CustomBlending` with `blendSrc=OneFactor`, `blendDst=SrcAlphaFactor` so the caustic plane blends additively with the ground (no black frame).

### Dynamic Caustics

Apply vertex displacement (e.g. Perlin noise) to the mesh, then recompute normals in the vertex shader using the neighboring-vertex technique (see [references/shaders.md](references/shaders.md) > Dynamic Normals section). Wire `time`, `amplitude`, `frequency` uniforms to both the normal material and the original mesh material.

## Alternative: Drei's Caustics Component

For production use, prefer `@react-three/drei`'s `<Caustics>` component:

```tsx
import { Caustics } from '@react-three/drei'

<Caustics
  lightSource={[5, 5, 5]}
  intensity={0.05}
  ior={1.1}
  backsideIOR={1.1}
  resolution={2048}
  color="white"
  frames={Infinity}
  backside={false}
  debug={false}
  causticsOnly={false}
  worldRadius={0.3125}
>
  <mesh>
    <sphereGeometry />
    <meshTransmissionMaterial />
  </mesh>
</Caustics>
```

Use `debug={true}` and `leva` to tune `intensity`, `worldRadius`, `ior`, `backsideIOR`.
