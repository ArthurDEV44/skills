# Drei Components Reference

`@react-three/drei` provides 100+ ready-to-use components for R3F. Below are the most commonly used, organized by category.

## Controls

### OrbitControls

```tsx
import { OrbitControls } from '@react-three/drei'

<OrbitControls
  makeDefault                   // Register as default controls
  enableDamping                 // Smooth deceleration
  dampingFactor={0.05}
  minDistance={2}               // Zoom limits
  maxDistance={20}
  minPolarAngle={0}             // Vertical rotation limits
  maxPolarAngle={Math.PI / 2}
  enablePan={false}             // Disable panning
  autoRotate                    // Auto-rotate
  autoRotateSpeed={1}
/>
```

### ScrollControls

Scroll-driven animation wrapping the entire scene:

```tsx
import { ScrollControls, Scroll, useScroll } from '@react-three/drei'

<ScrollControls pages={3} damping={0.1}>
  <Scroll>
    {/* 3D content — moves with scroll */}
    <ScrollScene />
  </Scroll>
  <Scroll html>
    {/* HTML content overlay */}
    <h1 style={{ position: 'absolute', top: '100vh' }}>Page 2</h1>
  </Scroll>
</ScrollControls>

function ScrollScene() {
  const scroll = useScroll()
  useFrame(() => {
    const offset = scroll.offset  // 0 to 1
    ref.current.rotation.y = offset * Math.PI * 2
  })
}
```

## Staging & Environment

### Environment

HDR environment maps for realistic lighting and reflections:

```tsx
import { Environment } from '@react-three/drei'

// Preset environments
<Environment preset="city" />
// Presets: apartment, city, dawn, forest, lobby, night, park, studio, sunset, warehouse

// As background
<Environment preset="sunset" background backgroundBlurriness={0.5} />

// From HDR file
<Environment files="/env.hdr" />

// Custom lightformer setup
<Environment>
  <Lightformer position={[0, 5, -5]} scale={10} intensity={4} />
  <Lightformer position={[0, 1, 5]} scale={5} color="blue" />
</Environment>
```

### Stage

Complete studio setup with lighting, shadows, and environment:

```tsx
import { Stage } from '@react-three/drei'

<Stage
  adjustCamera            // Auto-fit camera to content
  intensity={0.5}          // Light intensity
  shadows="contact"        // 'contact' or 'accumulative'
  preset="rembrandt"       // Lighting: rembrandt, portrait, upfront, soft
  environment="studio"     // Environment preset
>
  <Model />
</Stage>
```

### ContactShadows

Soft ground shadows without shadow maps:

```tsx
import { ContactShadows } from '@react-three/drei'

<ContactShadows
  position={[0, -0.5, 0]}
  opacity={0.5}
  scale={10}
  blur={1}
  far={10}
  resolution={256}
  frames={1}              // Render once for static scenes
/>
```

### Sky / Stars

```tsx
import { Sky, Stars } from '@react-three/drei'

<Sky sunPosition={[100, 20, 100]} turbidity={0.1} />
<Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />
```

## Loaders

### useGLTF

```tsx
import { useGLTF } from '@react-three/drei'

useGLTF.preload('/model.glb')  // Preload on module load

function Model() {
  const { scene, nodes, materials, animations } = useGLTF('/model.glb')
  return <primitive object={scene} />
}

// With Draco decoder
const gltf = useGLTF('/model.glb', '/draco/')
```

### useTexture

```tsx
import { useTexture } from '@react-three/drei'

function TexturedMesh() {
  const props = useTexture({
    map: '/color.jpg',
    normalMap: '/normal.jpg',
    roughnessMap: '/roughness.jpg',
    aoMap: '/ao.jpg',
  })
  return (
    <mesh>
      <sphereGeometry />
      <meshStandardMaterial {...props} />
    </mesh>
  )
}

// Single texture
const texture = useTexture('/texture.png')
```

## Abstractions

### Text (troika-three-text)

SDF text rendering — sharp at any size, supports wrapping and alignment:

```tsx
import { Text } from '@react-three/drei'

<Text
  position={[0, 1, 0]}
  fontSize={0.5}
  color="white"
  font="/fonts/Inter-Bold.woff"
  maxWidth={5}
  textAlign="center"
  anchorX="center"
  anchorY="middle"
>
  Hello World
</Text>
```

### Text3D

Extruded 3D geometry text:

```tsx
import { Text3D, Center } from '@react-three/drei'

<Center>
  <Text3D font="/fonts/Inter_Bold.json" size={1} height={0.2} bevelEnabled bevelSize={0.02}>
    Hello
    <meshNormalMaterial />
  </Text3D>
</Center>
```

### Html

Embed HTML in 3D space:

```tsx
import { Html } from '@react-three/drei'

<Html
  position={[0, 2, 0]}
  center                    // Center the HTML element
  distanceFactor={10}       // Scale with distance from camera
  occlude                   // Hidden behind 3D objects
  transform                 // CSS3D transform mode
>
  <div className="label">Annotation</div>
</Html>
```

### Float

Floating animation wrapper:

```tsx
import { Float } from '@react-three/drei'

<Float speed={2} rotationIntensity={1} floatIntensity={2} floatingRange={[-0.1, 0.1]}>
  <mesh>
    <boxGeometry />
    <meshStandardMaterial />
  </mesh>
</Float>
```

### Billboard

Always faces the camera:

```tsx
import { Billboard } from '@react-three/drei'

<Billboard follow lockX={false} lockY={false} lockZ={false}>
  <Text fontSize={0.5}>Always facing you</Text>
</Billboard>
```

## Materials

### MeshTransmissionMaterial

Glass/liquid with refraction, chromatic aberration, and caustics:

```tsx
import { MeshTransmissionMaterial } from '@react-three/drei'

<mesh>
  <sphereGeometry />
  <MeshTransmissionMaterial
    backside
    thickness={0.5}
    chromaticAberration={0.05}
    anisotropy={0.1}
    distortion={0.1}
    distortionScale={0.2}
    temporalDistortion={0.1}
    ior={1.5}
    color="white"
  />
</mesh>
```

### MeshReflectorMaterial

Reflective floors and surfaces:

```tsx
import { MeshReflectorMaterial } from '@react-three/drei'

<mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
  <planeGeometry args={[10, 10]} />
  <MeshReflectorMaterial
    blur={[300, 100]}
    resolution={1024}
    mixBlur={1}
    mixStrength={50}
    roughness={1}
    depthScale={1.2}
    minDepthThreshold={0.4}
    maxDepthThreshold={1.4}
    color="#151515"
    metalness={0.5}
  />
</mesh>
```

## Cameras

### PerspectiveCamera

```tsx
import { PerspectiveCamera } from '@react-three/drei'

<PerspectiveCamera makeDefault position={[0, 0, 5]} fov={75} />

// Children follow the camera
<PerspectiveCamera makeDefault position={[0, 2, 10]}>
  <mesh position={[0, 0, -2]}>
    <boxGeometry />
    <meshStandardMaterial />
  </mesh>
</PerspectiveCamera>
```

## Utilities

### Preload

Preload all assets used with useGLTF/useTexture:

```tsx
import { Preload } from '@react-three/drei'

<Canvas>
  <Suspense fallback={null}>
    <Scene />
  </Suspense>
  <Preload all />
</Canvas>
```

### useFBO

Create render targets (Frame Buffer Objects):

```tsx
import { useFBO } from '@react-three/drei'

function MyEffect() {
  const target = useFBO(512, 512, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
  })

  useFrame(({ gl, scene, camera }) => {
    gl.setRenderTarget(target)
    gl.render(scene, camera)
    gl.setRenderTarget(null)
  })
}
```

### useHelper

Visualize lights/cameras with helpers:

```tsx
import { useHelper } from '@react-three/drei'
import { DirectionalLightHelper } from 'three'

function Light() {
  const ref = useRef()
  useHelper(ref, DirectionalLightHelper, 1, 'red')
  return <directionalLight ref={ref} position={[5, 5, 5]} />
}
```
