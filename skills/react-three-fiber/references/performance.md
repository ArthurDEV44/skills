# R3F Performance Patterns

## Re-Render Isolation

The most impactful R3F optimization: keep animated/stateful components small and isolated.

```tsx
// BAD — entire scene re-renders when hovered changes
function Scene() {
  const [hovered, setHovered] = useState(false)
  return (
    <group>
      <mesh onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
        <boxGeometry />
        <meshStandardMaterial color={hovered ? 'red' : 'blue'} />
      </mesh>
      <HeavyComponent />  {/* Re-renders unnecessarily */}
      <AnotherHeavy />     {/* Re-renders unnecessarily */}
    </group>
  )
}

// GOOD — only InteractiveMesh re-renders
function InteractiveMesh() {
  const [hovered, setHovered] = useState(false)
  return (
    <mesh onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
      <boxGeometry />
      <meshStandardMaterial color={hovered ? 'red' : 'blue'} />
    </mesh>
  )
}

function Scene() {
  return (
    <group>
      <InteractiveMesh />
      <HeavyComponent />
      <AnotherHeavy />
    </group>
  )
}
```

## useFrame Rules

```tsx
// BAD — sets state 60 times/sec, causes 60 re-renders/sec
useFrame(() => {
  setPosition([x, y, z])
})

// GOOD — direct ref mutation, zero re-renders
useFrame((state, delta) => {
  ref.current.position.x += delta
  ref.current.material.uniforms.uTime.value = state.clock.elapsedTime
})
```

## Instancing

For 100+ identical meshes, use `<Instances>` from drei:

```tsx
import { Instances, Instance } from '@react-three/drei'

function Trees({ count = 1000 }) {
  const positions = useMemo(() =>
    Array.from({ length: count }, () => [
      (Math.random() - 0.5) * 50,
      0,
      (Math.random() - 0.5) * 50,
    ]),
    [count]
  )

  return (
    <Instances limit={count} range={count}>
      <boxGeometry args={[0.2, 2, 0.2]} />
      <meshStandardMaterial color="green" />
      {positions.map((pos, i) => (
        <Instance key={i} position={pos} />
      ))}
    </Instances>
  )
}
```

For raw Three.js instancing:

```tsx
function Particles({ count = 5000 }) {
  const ref = useRef<THREE.InstancedMesh>(null!)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  useEffect(() => {
    for (let i = 0; i < count; i++) {
      dummy.position.set(
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10
      )
      dummy.updateMatrix()
      ref.current.setMatrixAt(i, dummy.matrix)
    }
    ref.current.instanceMatrix.needsUpdate = true
  }, [count])

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, count]}>
      <sphereGeometry args={[0.05, 8, 8]} />
      <meshStandardMaterial />
    </instancedMesh>
  )
}
```

## Merged Geometries

When you have many different meshes that share the same geometry set:

```tsx
import { Merged } from '@react-three/drei'

function Furniture({ items }) {
  const { nodes } = useGLTF('/furniture.glb')

  return (
    <Merged meshes={nodes}>
      {(models) => items.map((item, i) => (
        <models.Chair key={i} position={item.position} />
      ))}
    </Merged>
  )
}
```

## Demand Rendering

For scenes that don't animate constantly:

```tsx
<Canvas frameloop="demand">
  <StaticScene />
</Canvas>

// Inside components — request a frame when needed
function DynamicPart() {
  const invalidate = useThree((s) => s.invalidate)

  function handleChange(newValue) {
    setValue(newValue)
    invalidate()  // Request one frame
  }
}
```

R3F auto-invalidates when:
- Props change on any R3F element
- State changes via `useThree.set()`
- Events fire (pointer, resize)

## Resource Disposal

R3F auto-disposes geometries, materials, and textures when components unmount. For custom objects, clean up manually:

```tsx
function CustomObject() {
  const geometry = useMemo(() => new THREE.BoxGeometry(), [])
  const material = useMemo(() => new THREE.MeshStandardMaterial(), [])

  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])

  return <mesh geometry={geometry} material={material} />
}
```

Disable auto-dispose when sharing resources:

```tsx
<mesh geometry={sharedGeometry} dispose={null}>
```

## BVH Acceleration

Speed up raycasting with Bounding Volume Hierarchies:

```tsx
import { Bvh } from '@react-three/drei'

<Canvas>
  <Bvh firstHitOnly>
    <ComplexScene />
  </Bvh>
</Canvas>
```

## Adaptive Performance

Use `PerformanceMonitor` to auto-adjust quality:

```tsx
import { PerformanceMonitor } from '@react-three/drei'

function Scene() {
  const [dpr, setDpr] = useState(1.5)

  return (
    <Canvas dpr={dpr}>
      <PerformanceMonitor
        onIncline={() => setDpr(2)}
        onDecline={() => setDpr(1)}
        flipflops={3}               // Max quality changes
        onFallback={() => setDpr(0.75)}  // Fallback after too many flips
      />
      <SceneContent />
    </Canvas>
  )
}
```

## Memoization Patterns

```tsx
// Memoize uniforms for shader materials
const uniforms = useMemo(() => ({
  uTime: { value: 0 },
  uTexture: { value: texture },
}), [texture])

// Memoize geometry computations
const points = useMemo(() =>
  new Float32Array(count * 3).map(() => (Math.random() - 0.5) * 10),
  [count]
)

// Memoize materials to prevent recreation
const material = useMemo(() =>
  new THREE.MeshStandardMaterial({ color: 'red' }),
  []
)
```

## Quick Checklist

- [ ] Animated components are isolated (no parent re-renders)
- [ ] useFrame uses refs, not setState
- [ ] useThree uses selectors
- [ ] 100+ identical meshes use Instances/InstancedMesh
- [ ] Heavy assets wrapped in Suspense
- [ ] Static scenes use `frameloop="demand"`
- [ ] Shared resources use `dispose={null}`
- [ ] DPR is capped (`dpr={[1, 1.5]}` for mobile)
- [ ] Uniforms and geometries are memoized
- [ ] Complex scenes use Bvh for raycasting
