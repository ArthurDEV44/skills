# R3F Events & Interactivity

## Pointer Events

Any mesh, line, or object that supports raycasting can receive pointer events in R3F. Events are modeled after DOM pointer events but include 3D intersection data.

### Available Events

| Event | Fires when |
|-------|-----------|
| `onClick` | Click (pointerdown + pointerup on same object, within `delta` threshold) |
| `onDoubleClick` | Double click |
| `onContextMenu` | Right-click / long-press |
| `onPointerDown` | Pointer pressed on object |
| `onPointerUp` | Pointer released |
| `onPointerOver` | Pointer enters object (like mouseenter) |
| `onPointerOut` | Pointer leaves object (like mouseleave) |
| `onPointerEnter` | Pointer enters — does not bubble |
| `onPointerLeave` | Pointer leaves — does not bubble |
| `onPointerMove` | Pointer moves while over object |
| `onPointerMissed` | Click that misses this object (on Canvas too) |
| `onWheel` | Mouse wheel over object |

### Event Object

```tsx
interface ThreeEvent<TSourceEvent> {
  object: THREE.Object3D          // The intersected object
  eventObject: THREE.Object3D     // The object the handler is on
  point: THREE.Vector3            // World-space intersection point
  distance: number                // Distance from camera
  ray: THREE.Ray                  // The cast ray
  camera: THREE.Camera            // Camera used for raycasting
  face: THREE.Face | null         // Intersected face
  faceIndex: number | null        // Face index
  uv: THREE.Vector2 | undefined   // UV coordinates at intersection
  uv1: THREE.Vector2 | undefined  // Second UV set
  delta: number                   // Click distance (for onClick)
  unprojectedPoint: THREE.Vector3 // Point before projection
  stopped: boolean                // If propagation was stopped

  // DOM event passthrough
  nativeEvent: TSourceEvent
  stopPropagation: () => void
  intersections: Intersection[]   // All intersections along the ray
}
```

### Propagation

Events propagate from the **closest** intersection outward through ancestors. Call `e.stopPropagation()` to prevent an event from reaching objects behind:

```tsx
<mesh onClick={(e) => {
  e.stopPropagation()  // Objects behind won't receive this click
  handleClick()
}}>
```

Without `stopPropagation`, ALL intersected objects along the ray receive the event.

### Hover Cursor Pattern

```tsx
function HoverableMesh({ children, ...props }) {
  const [hovered, setHovered] = useState(false)

  useEffect(() => {
    document.body.style.cursor = hovered ? 'pointer' : 'auto'
    return () => { document.body.style.cursor = 'auto' }
  }, [hovered])

  return (
    <mesh
      {...props}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true) }}
      onPointerOut={() => setHovered(false)}
    >
      {children}
    </mesh>
  )
}
```

### Drag Pattern

```tsx
function DraggableMesh() {
  const ref = useRef<THREE.Mesh>(null!)
  const [dragging, setDragging] = useState(false)
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), [])
  const intersection = useMemo(() => new THREE.Vector3(), [])

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    e.target.setPointerCapture(e.pointerId)
    setDragging(true)
  }

  const onPointerUp = (e: ThreeEvent<PointerEvent>) => {
    e.target.releasePointerCapture(e.pointerId)
    setDragging(false)
  }

  const onPointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!dragging) return
    e.ray.intersectPlane(plane, intersection)
    ref.current.position.copy(intersection)
  }

  return (
    <mesh
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      onPointerMove={onPointerMove}
    >
      <boxGeometry />
      <meshStandardMaterial color={dragging ? 'hotpink' : 'orange'} />
    </mesh>
  )
}
```

## Raycaster Configuration

### Canvas-Level

```tsx
<Canvas raycaster={{ computeOffsets: (e) => ({ offsetX: e.clientX, offsetY: e.clientY }) }}>
```

### Object-Level

Filter events per-object using `raycast`:

```tsx
<mesh raycast={THREE.Mesh.prototype.raycast}>
```

Or disable raycasting entirely:

```tsx
<mesh raycast={() => null}>
```

## Event Source

By default, events are attached to the canvas's parent. Change this for overlays or split views:

```tsx
const domRef = useRef<HTMLDivElement>(null)

<div ref={domRef} style={{ width: '100%', height: '100%' }}>
  <Canvas eventSource={domRef} eventPrefix="client">
    <Scene />
  </Canvas>
</div>
```

`eventPrefix` options: `'offset'` (default), `'client'`, `'page'`, `'layer'`, `'screen'`

## Pointer Lock

```tsx
function PointerLockScene() {
  const gl = useThree((s) => s.gl)

  useEffect(() => {
    const handleClick = () => gl.domElement.requestPointerLock()
    gl.domElement.addEventListener('click', handleClick)
    return () => gl.domElement.removeEventListener('click', handleClick)
  }, [gl])

  return <Scene />
}
```

## Performance Notes

- Events only fire on objects with `onPointer*` or `onClick` props — no overhead for non-interactive meshes
- Use `<Bvh>` for complex scenes with many interactive objects
- `e.stopPropagation()` reduces intersection checks
- For thousands of instances, consider a single hit-test mesh over individual instance events
