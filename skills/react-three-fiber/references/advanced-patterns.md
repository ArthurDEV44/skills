# R3F Advanced Patterns

## Views & Portals

### View (Drei)

Render different scenes or viewpoints into separate DOM containers from a single Canvas:

```tsx
import { Canvas } from '@react-three/fiber'
import { View, OrbitControls } from '@react-three/drei'
import { useRef } from 'react'

function App() {
  const containerRef = useRef<HTMLDivElement>(null!)

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100vh' }}>
      {/* View tracks its DOM container's position/size */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '50%', height: '100%' }}>
        <View style={{ width: '100%', height: '100%' }}>
          <Scene1 />
          <OrbitControls makeDefault />
        </View>
      </div>

      <div style={{ position: 'absolute', top: 0, right: 0, width: '50%', height: '100%' }}>
        <View style={{ width: '100%', height: '100%' }}>
          <Scene2 />
          <OrbitControls makeDefault />
        </View>
      </div>

      {/* Single shared Canvas — eventSource must be the common parent */}
      <Canvas eventSource={containerRef} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        <View.Port />
      </Canvas>
    </div>
  )
}
```

### RenderTexture

Render a scene into a texture usable on any material:

```tsx
import { RenderTexture, Text, OrbitControls } from '@react-three/drei'

function TVScreen() {
  return (
    <mesh>
      <planeGeometry args={[4, 3]} />
      <meshStandardMaterial>
        <RenderTexture attach="map" width={512} height={384}>
          <color attach="background" args={['orange']} />
          <ambientLight intensity={0.5} />
          <Text fontSize={0.5} color="white">Live Feed</Text>
          <mesh>
            <boxGeometry />
            <meshStandardMaterial color="blue" />
          </mesh>
          <OrbitControls />
        </RenderTexture>
      </meshStandardMaterial>
    </mesh>
  )
}
```

### createPortal

R3F's portal for rendering into a different scene:

```tsx
import { createPortal, useFrame, useThree } from '@react-three/fiber'
import { useFBO } from '@react-three/drei'

function Portal() {
  const otherScene = useMemo(() => new THREE.Scene(), [])
  const target = useFBO(512, 512)
  const camera = useRef<THREE.PerspectiveCamera>(null!)

  useFrame(({ gl }) => {
    gl.setRenderTarget(target)
    gl.render(otherScene, camera.current)
    gl.setRenderTarget(null)
  })

  return (
    <>
      {createPortal(
        <>
          <perspectiveCamera ref={camera} position={[0, 0, 5]} />
          <ambientLight />
          <mesh>
            <torusGeometry />
            <meshNormalMaterial />
          </mesh>
        </>,
        otherScene
      )}
      <mesh>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial map={target.texture} />
      </mesh>
    </>
  )
}
```

## Extending Three.js

### Custom Objects

Register any Three.js class for JSX use:

```tsx
import { extend } from '@react-three/fiber'
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass'

extend({ TextGeometry, UnrealBloomPass })

declare module '@react-three/fiber' {
  interface ThreeElements {
    textGeometry: Object3DNode<TextGeometry, typeof TextGeometry>
    unrealBloomPass: Object3DNode<UnrealBloomPass, typeof UnrealBloomPass>
  }
}

// Now use in JSX
<mesh>
  <textGeometry args={[text, config]} />
</mesh>
```

### Custom attach

The `attach` prop controls how children connect to parents:

```tsx
// Auto-detected
<mesh>
  <boxGeometry />            {/* attach="geometry" */}
  <meshStandardMaterial />    {/* attach="material" */}
</mesh>

// Explicit for arrays
<mesh>
  <meshStandardMaterial attach="material-0" />  {/* material[0] */}
  <meshStandardMaterial attach="material-1" />  {/* material[1] */}
</mesh>

// Functional attach for complex hierarchies
<effectComposer>
  <renderPass attach={(parent, self) => {
    parent.addPass(self)
    return () => parent.removePass(self)
  }} />
</effectComposer>
```

## Next.js Integration

### App Router (RSC-safe)

```tsx
// components/Scene.tsx
'use client'

import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import { Suspense } from 'react'

export default function Scene() {
  return (
    <Canvas>
      <Suspense fallback={null}>
        <Environment preset="city" />
        <mesh>
          <boxGeometry />
          <meshStandardMaterial />
        </mesh>
      </Suspense>
      <OrbitControls />
    </Canvas>
  )
}
```

```tsx
// app/page.tsx (Server Component)
import dynamic from 'next/dynamic'

const Scene = dynamic(() => import('@/components/Scene'), { ssr: false })

export default function Page() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Scene />
    </div>
  )
}
```

**Key rules:**
- Mark all R3F components with `'use client'`
- Use `dynamic(() => import(...), { ssr: false })` for Canvas — WebGL requires DOM
- Or use `<Suspense>` with a client boundary wrapper

### Vite

Works out of the box — just import and use. For GLTF, ensure `public/` folder has models.

## Animations

### useFrame Animations

```tsx
function Orbiter({ radius = 2, speed = 1 }) {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame(({ clock }) => {
    const t = clock.elapsedTime * speed
    ref.current.position.x = Math.cos(t) * radius
    ref.current.position.z = Math.sin(t) * radius
    ref.current.lookAt(0, 0, 0)
  })

  return <mesh ref={ref}><sphereGeometry args={[0.2]} /><meshStandardMaterial /></mesh>
}
```

### GLTF Animations

```tsx
import { useGLTF, useAnimations } from '@react-three/drei'
import { useEffect } from 'react'

function AnimatedModel({ action = 'Walk' }) {
  const { scene, animations } = useGLTF('/character.glb')
  const { actions } = useAnimations(animations, scene)

  useEffect(() => {
    actions[action]?.reset().fadeIn(0.5).play()
    return () => { actions[action]?.fadeOut(0.5) }
  }, [action, actions])

  return <primitive object={scene} />
}
```

### Spring Animations (react-spring)

```tsx
import { useSpring, animated } from '@react-spring/three'

function AnimatedBox() {
  const [active, setActive] = useState(false)

  const { scale, color } = useSpring({
    scale: active ? 1.5 : 1,
    color: active ? 'hotpink' : 'orange',
  })

  return (
    <animated.mesh scale={scale} onClick={() => setActive(!active)}>
      <boxGeometry />
      <animated.meshStandardMaterial color={color} />
    </animated.mesh>
  )
}
```

## Suspense & Loading

```tsx
import { Suspense } from 'react'
import { useProgress, Html } from '@react-three/drei'

function Loader() {
  const { progress } = useProgress()
  return <Html center>{progress.toFixed(0)}%</Html>
}

function App() {
  return (
    <Canvas>
      <Suspense fallback={<Loader />}>
        <HeavyScene />
      </Suspense>
    </Canvas>
  )
}
```

## Accessing the Canvas from Outside

```tsx
import { useThree } from '@react-three/fiber'

// Screenshot
function Screenshot() {
  const gl = useThree((s) => s.gl)

  function capture() {
    const link = document.createElement('a')
    link.download = 'screenshot.png'
    link.href = gl.domElement.toDataURL('image/png')
    link.click()
  }

  return <Html><button onClick={capture}>Screenshot</button></Html>
}
```

## TypeScript Tips

```tsx
import { useRef } from 'react'
import * as THREE from 'three'
import { ThreeElements } from '@react-three/fiber'

// Ref typing
const meshRef = useRef<THREE.Mesh>(null!)
const groupRef = useRef<THREE.Group>(null!)
const materialRef = useRef<THREE.MeshStandardMaterial>(null!)

// Props typing for reusable components
type BoxProps = ThreeElements['mesh'] & { color?: string }

function Box({ color = 'orange', ...props }: BoxProps) {
  return (
    <mesh {...props}>
      <boxGeometry />
      <meshStandardMaterial color={color} />
    </mesh>
  )
}
```
