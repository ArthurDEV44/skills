# R3F Hooks — Full API Reference

## useFrame(callback, priority?)

Runs `callback` on every rendered frame. The callback receives the R3F state and `delta` (seconds since last frame).

```tsx
useFrame((state, delta, xrFrame) => {
  // state.clock.elapsedTime — total time
  // state.clock.getDelta()  — same as delta
  // state.pointer           — normalized pointer coordinates [-1, 1]
  // state.camera            — active camera
  // state.gl                — WebGLRenderer
  // state.scene             — root scene
  // state.raycaster         — raycaster
  // state.viewport          — viewport in Three.js units
  // state.size              — canvas size in pixels
})
```

### Priority System

- Default priority is `0` — R3F renders the scene after all priority-0 callbacks
- Priority > 0 runs **after** the default render — use this to take over rendering
- Lower priority runs first

```tsx
// Animation (runs before render)
useFrame((state, delta) => {
  meshRef.current.rotation.y += delta
}, 0)

// Custom render pass (runs after default)
useFrame(({ gl, scene, camera }) => {
  gl.render(scene, camera)
}, 1)

// Post-processing (runs last)
useFrame(() => {
  composer.render()
}, 2)
```

### Rules

- **Never call `setState`** inside useFrame — causes re-renders every frame
- **Mutate refs** for animations: `ref.current.position.x += delta`
- **Use `delta`** for frame-rate-independent animation, not `elapsedTime` directly for increments
- **Return value**: Return `null` to skip the default render pass (when using custom rendering)

## useThree(selector?)

Access the R3F root state. Without a selector, returns the entire state object and re-renders on every state change. Always use a selector in production.

### State Fields

```tsx
interface RootState {
  gl: THREE.WebGLRenderer        // The renderer
  scene: THREE.Scene              // The root scene
  camera: THREE.Camera            // The active camera
  raycaster: THREE.Raycaster      // The raycaster
  pointer: THREE.Vector2          // Normalized pointer coords [-1, 1]
  mouse: THREE.Vector2            // Deprecated alias for pointer
  clock: THREE.Clock              // The clock
  size: { width, height, top, left, updateStyle } // Canvas size in pixels
  viewport: {                     // Viewport in Three.js units
    width, height,                // At current camera distance
    initialDpr, dpr,
    factor,                       // Pixel-to-unit conversion
    distance,                     // Camera distance
    aspect,
    getCurrentViewport(camera?, target?, size?)
  }
  events: EventManager            // Event system
  xr: XRState                     // WebXR state
  controls: THREE.EventDispatcher | null  // Attached controls

  // Methods
  get: () => RootState            // Non-reactive snapshot
  set: (partial) => void          // Merge partial state
  invalidate: () => void          // Request a frame (demand mode)
  advance: (timestamp, runGlobalEffects?) => void  // Advance one frame (never mode)
  setSize: (width, height, updateStyle?, top?, left?) => void
  setDpr: (dpr) => void
  setFrameloop: (frameloop) => void
  setEvents: (events) => void
  onPointerMissed: () => void
  performance: {
    current: number               // 0-1, adaptive performance
    min: number
    max: number
    debounce: number
    regress: () => void           // Signal performance issues
  }
}
```

### Selector Patterns

```tsx
// Single property
const camera = useThree((s) => s.camera)
const gl = useThree((s) => s.gl)

// Multiple via destructuring (still subscribes to all)
const { viewport, size } = useThree()  // Avoid — re-renders too much

// Derived value
const aspect = useThree((s) => s.viewport.aspect)

// Non-reactive access (for event handlers, useFrame)
const get = useThree((s) => s.get)
function handleClick() {
  const { camera, scene } = get()
}
```

## useLoader(Loader, url, extensions?, onProgress?)

Suspense-compatible asset loader. Automatically caches results by URL.

```tsx
import { useLoader } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader'

// Basic
const gltf = useLoader(GLTFLoader, '/model.glb')

// With extensions (e.g., Draco)
const gltf = useLoader(GLTFLoader, '/model.glb', (loader) => {
  const draco = new DRACOLoader()
  draco.setDecoderPath('/draco/')
  loader.setDRACOLoader(draco)
})

// Multiple URLs (returns array)
const [tex1, tex2] = useLoader(TextureLoader, ['/a.png', '/b.png'])

// Preload (call outside component)
useLoader.preload(GLTFLoader, '/model.glb')

// Clear cache
useLoader.clear(GLTFLoader, '/model.glb')
```

### GLTF Scene Extraction

When a loader returns `result.scene`, useLoader auto-extracts `nodes` and `materials`:

```tsx
const { nodes, materials, animations, scene } = useLoader(GLTFLoader, '/model.glb')
// nodes: { MeshName: THREE.Mesh, ... }
// materials: { MatName: THREE.Material, ... }
```

## useGraph(object)

Extracts `nodes` and `materials` from any `THREE.Object3D`:

```tsx
import { useGraph } from '@react-three/fiber'

function Model({ scene }) {
  const { nodes, materials } = useGraph(scene)
  return <mesh geometry={nodes.Body.geometry} material={materials.Skin} />
}
```
