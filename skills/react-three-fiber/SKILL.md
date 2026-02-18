---
name: react-three-fiber
description: >-
  React Three Fiber (R3F) — the React renderer for Three.js. Build declarative, component-based 3D scenes
  with reusable, state-reactive components. Use when: (1) Setting up a React Three Fiber project with Canvas,
  camera, lighting, and scene configuration, (2) Using R3F hooks — useFrame, useThree, useLoader, useGraph,
  (3) Creating 3D components with JSX primitives, geometries, materials, and lights, (4) Extending Three.js
  with custom objects via extend(), (5) Handling pointer events (onClick, onPointerOver, onPointerMove) on 3D
  objects, (6) Optimizing R3F performance — instancing, disposal, demand rendering, re-render prevention,
  (7) Loading 3D models (GLTF/GLB) with useGLTF or useLoader, (8) Using @react-three/drei helpers —
  OrbitControls, Environment, Stage, Text, Html, Instances, View, (9) Working with portals, views, and
  render textures, (10) Integrating R3F with Next.js, Vite, or other React frameworks.
---

# React Three Fiber

React Three Fiber (R3F) is a React renderer for Three.js. Every Fiber component creates a corresponding `THREE.*` object and composes them into a scene graph. R3F provides a render loop, pointer events via raycasting, automatic resizing, and full access to the Three.js ecosystem — all through declarative JSX.

## Installation

```bash
npm install three @react-three/fiber @react-three/drei
# Optional
npm install @react-three/postprocessing @react-three/rapier
```

**TypeScript types:**

```bash
npm install -D @types/three
```

## Canvas — The Root Component

`Canvas` sets up a `THREE.WebGLRenderer`, scene, camera, and render loop. Everything inside renders into the Three.js scene graph.

```tsx
import { Canvas } from '@react-three/fiber'

function App() {
  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 75 }}
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      frameloop="always"
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} castShadow />
      <MyScene />
    </Canvas>
  )
}
```

### Canvas Defaults

- Renderer: `WebGLRenderer` with antialias + alpha enabled, high-performance power preference
- Camera: `PerspectiveCamera` at `[0, 0, 0]` (use `orthographic` prop for ortho)
- Color: `SRGBColorSpace` output, `ACESFilmicToneMapping` (disable with `flat` prop)
- Shadows: `PCFSoftShadowMap` when `shadows` is `true`
- DPR: `[1, 2]` — clamped to device pixel ratio
- Resize: Automatic via `react-use-measure`

### Key Canvas Props

| Prop | Type | Default | Purpose |
|------|------|---------|---------|
| `camera` | object / Camera | PerspectiveCamera | Camera config or instance |
| `shadows` | boolean / string | false | Enable shadow maps (`'basic'`, `'percentage'`, `'soft'`, `'variance'`) |
| `dpr` | number / [min, max] | [1, 2] | Device pixel ratio |
| `frameloop` | string | `'always'` | `'always'` / `'demand'` / `'never'` |
| `flat` | boolean | false | Disable tone mapping |
| `linear` | boolean | false | Disable sRGB color space |
| `orthographic` | boolean | false | Use orthographic camera |
| `gl` | object / callback | — | Renderer props or factory `gl={defaults => new Renderer(defaults)}` |
| `onCreated` | function | — | Callback after canvas mounts, receives state |
| `onPointerMissed` | function | — | Click that misses all objects |

## JSX & The Object API

R3F maps JSX elements to Three.js constructors. The mapping is **camelCase**:

```tsx
// <mesh /> → new THREE.Mesh()
// <boxGeometry /> → new THREE.BoxGeometry()
// <meshStandardMaterial /> → new THREE.MeshStandardMaterial()

<mesh position={[1, 2, 3]} rotation={[0, Math.PI / 2, 0]} scale={1.5}>
  <boxGeometry args={[1, 1, 1]} />
  <meshStandardMaterial color="hotpink" metalness={0.8} roughness={0.2} />
</mesh>
```

### Rules

1. **`args`** — Constructor arguments as an array: `<boxGeometry args={[2, 2, 2]} />`
2. **`attach`** — How a child attaches to its parent. Auto-detected for geometries (`"geometry"`) and materials (`"material"`). Override for custom attachments: `<texture attach="map" />`
3. **`object` via `<primitive>`** — Injects an existing Three.js object: `<primitive object={gltf.scene} />`
4. **Nested properties** — Use dashes: `<meshStandardMaterial color-r={1} />` or `<directionalLight shadow-mapSize={[2048, 2048]} />`
5. **`set` shortcut** — Set scalar properties on vector types: `<mesh position={[0, 0, 0]} />` calls `position.set(0, 0, 0)`

### Extending Three.js

Register custom classes so R3F recognizes them in JSX:

```tsx
import { extend } from '@react-three/fiber'
import { OrbitControls } from 'three-stdlib'

extend({ OrbitControls })

// Now usable as <orbitControls />
```

For custom materials via drei's `shaderMaterial`:

```tsx
import { shaderMaterial } from '@react-three/drei'
import { extend } from '@react-three/fiber'

const WaveMaterial = shaderMaterial(
  { uTime: 0, uColor: new THREE.Color(0.2, 0.0, 0.1) },
  /* glsl */ `...vertexShader...`,
  /* glsl */ `...fragmentShader...`
)

extend({ WaveMaterial })
// <waveMaterial ref={ref} uTime={0} />
```

## Core Hooks

### useFrame — Per-Frame Logic

Runs every frame **inside the render loop**. Use refs for mutations — never `setState` here.

```tsx
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'

function Spinner() {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame((state, delta) => {
    ref.current.rotation.y += delta               // Consistent speed
    ref.current.position.y = Math.sin(state.clock.elapsedTime)
  })

  return (
    <mesh ref={ref}>
      <boxGeometry />
      <meshStandardMaterial />
    </mesh>
  )
}
```

**State object fields:** `gl`, `scene`, `camera`, `raycaster`, `pointer`, `clock`, `viewport`, `size`, `get`, `set`, `invalidate`, `advance`

**Priority:** `useFrame(callback, priority)` — higher priority runs later. Default is 0. Use priority > 0 to take over the render loop:

```tsx
useFrame(({ gl, scene, camera }) => {
  gl.render(scene, camera)
}, 1)
```

### useThree — Access State

Reactive access to the R3F state. Use **selectors** to avoid unnecessary re-renders:

```tsx
// Bad — re-renders on ANY state change
const state = useThree()

// Good — re-renders only when camera changes
const camera = useThree((s) => s.camera)
const viewport = useThree((s) => s.viewport)
const size = useThree((s) => s.size)

// Non-reactive read
const get = useThree((s) => s.get)
const currentCamera = get().camera

// Trigger render in demand mode
const invalidate = useThree((s) => s.invalidate)
```

### useLoader — Load Assets

Suspense-compatible asset loading:

```tsx
import { useLoader } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import { TextureLoader } from 'three'

function Model() {
  const gltf = useLoader(GLTFLoader, '/model.glb')
  return <primitive object={gltf.scene} />
}

function Textured() {
  const texture = useLoader(TextureLoader, '/texture.png')
  return (
    <mesh>
      <planeGeometry />
      <meshStandardMaterial map={texture} />
    </mesh>
  )
}

// Wrap in Suspense
<Suspense fallback={<FallbackComponent />}>
  <Model />
</Suspense>
```

**Prefer drei's typed hooks** when available: `useGLTF`, `useTexture`, `useFBX`, `useCubeTexture`.

## Pointer Events

Meshes and lines with `onPointer*` props receive raycasted pointer events automatically:

```tsx
<mesh
  onClick={(e) => { e.stopPropagation(); toggle() }}
  onPointerOver={(e) => { e.stopPropagation(); setHovered(true) }}
  onPointerOut={() => setHovered(false)}
  onPointerMove={(e) => console.log('3D point:', e.point)}
  onDoubleClick={(e) => console.log('double click')}
  onContextMenu={(e) => console.log('right click')}
>
```

**Event object fields:** `object`, `eventObject`, `point`, `distance`, `face`, `faceIndex`, `uv`, `ray`, `camera`, `stopPropagation()`, `delta` (click distance)

**Key rules:**
- Call `e.stopPropagation()` to prevent events from reaching objects behind
- Events bubble from the closest intersection outward
- `onPointerMissed` on Canvas fires when no object is hit

## Performance Essentials

1. **Never `setState` in useFrame** — mutate refs directly
2. **Use selectors in useThree** — `useThree(s => s.camera)` not `useThree()`
3. **Memoize** — `useMemo` for computed objects, geometries, materials
4. **Instance repeated meshes** — `<Instances>` / `<InstancedMesh>` for 100+ identical objects
5. **Demand rendering** — `frameloop="demand"` + `invalidate()` for static/infrequent updates
6. **Dispose resources** — R3F auto-disposes on unmount, but manually dispose with `useEffect` cleanup for custom objects
7. **Limit re-renders** — extract animated parts into their own components
8. **DPR control** — `dpr={[1, 1.5]}` for mobile-friendly rendering
9. **Suspense boundaries** — wrap heavy components for progressive loading

For detailed performance patterns, see [references/performance.md](references/performance.md).

## Drei — Essential Helpers

`@react-three/drei` provides 100+ components. Key categories:

| Category | Components |
|----------|-----------|
| **Controls** | `OrbitControls`, `CameraControls`, `ScrollControls`, `FlyControls` |
| **Staging** | `Environment`, `Stage`, `Sky`, `Stars`, `ContactShadows`, `AccumulativeShadows` |
| **Loaders** | `useGLTF`, `useTexture`, `useFBX`, `useCubeTexture` |
| **Abstractions** | `Text`, `Text3D`, `Html`, `Billboard`, `Float`, `MeshWobbleMaterial` |
| **Materials** | `MeshTransmissionMaterial`, `MeshReflectorMaterial`, `shaderMaterial` |
| **Performance** | `Instances`, `Merged`, `Bvh`, `Preload`, `BakeShadows` |
| **Portals** | `View`, `RenderTexture`, `Hud` |
| **Cameras** | `PerspectiveCamera`, `OrthographicCamera`, `CubeCamera` |

For detailed drei component patterns, see [references/drei-components.md](references/drei-components.md).

## Loading 3D Models

```tsx
import { useGLTF } from '@react-three/drei'
import { Suspense } from 'react'

// Preload on module load
useGLTF.preload('/model.glb')

function Model() {
  const { scene, nodes, materials, animations } = useGLTF('/model.glb')

  // Option 1: Render entire scene
  return <primitive object={scene} />
}

function SelectiveModel() {
  const { nodes, materials } = useGLTF('/robot.glb')

  // Option 2: Cherry-pick nodes
  return (
    <group>
      <mesh geometry={nodes.Body.geometry} material={materials.Metal} castShadow />
      <mesh geometry={nodes.Head.geometry} material={materials.Glass} />
    </group>
  )
}

function App() {
  return (
    <Canvas>
      <Suspense fallback={null}>
        <Model />
      </Suspense>
    </Canvas>
  )
}
```

**Tip:** Use [gltf.pmnd.rs](https://gltf.pmnd.rs) to auto-generate R3F JSX from `.glb` files.

## Common Patterns

### Scene Template

```tsx
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import { Suspense } from 'react'

export default function App() {
  return (
    <Canvas shadows camera={{ position: [3, 3, 3], fov: 50 }}>
      <Suspense fallback={null}>
        <Environment preset="city" />
        <MyScene />
      </Suspense>
      <OrbitControls makeDefault />
    </Canvas>
  )
}
```

### Responsive Sizing

```tsx
function ResponsiveMesh() {
  const viewport = useThree((s) => s.viewport)
  return (
    <mesh scale={viewport.width / 5}>
      <planeGeometry />
      <meshBasicMaterial />
    </mesh>
  )
}
```

### Animation with useFrame

```tsx
function AnimatedBox() {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame((state, delta) => {
    ref.current.rotation.x += delta * 0.5
    ref.current.position.y = Math.sin(state.clock.elapsedTime) * 0.5
  })

  return (
    <mesh ref={ref}>
      <boxGeometry />
      <meshNormalMaterial />
    </mesh>
  )
}
```

## Reference Guides

- **Hooks Deep Dive**: See [references/hooks-api.md](references/hooks-api.md) — full useFrame, useThree, useLoader, useGraph API
- **Drei Components**: See [references/drei-components.md](references/drei-components.md) — organized by category with usage patterns
- **Performance**: See [references/performance.md](references/performance.md) — instancing, disposal, demand rendering, re-render isolation
- **Events & Interactivity**: See [references/events-interactivity.md](references/events-interactivity.md) — pointer events, raycasting, drag, cursor
- **Advanced Patterns**: See [references/advanced-patterns.md](references/advanced-patterns.md) — portals, Views, RenderTexture, extending, Next.js integration
