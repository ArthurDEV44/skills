# Caustic Plane Positioning and Scaling

## Table of Contents
- [1. Bounding Box Vertices](#1-bounding-box-vertices)
- [2. Projecting Vertices onto the Ground](#2-projecting-vertices-onto-the-ground)
- [3. Calculating Plane Position (Weighted Center)](#3-calculating-plane-position)
- [4. Calculating Plane Scale](#4-calculating-plane-scale)
- [5. Complete Implementation](#5-complete-implementation)

---

## 1. Bounding Box Vertices

Extract the 8 corners of the mesh's axis-aligned bounding box:

```js
const bounds = new THREE.Box3().setFromObject(mesh.current, true)

const boundsVertices = [
  new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.min.z),
  new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
  new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
  new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
  new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
  new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
  new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
  new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.max.z),
]
```

---

## 2. Projecting Vertices onto the Ground

Project each bounding box vertex along the light direction to intersect with the ground plane (Y=0).

**Formula:**

```
projectedVertex = vertex + lightDir * ((planeY - vertex.y) / lightDir.y)
```

With `planeY = 0`:

```js
const lightDir = new THREE.Vector3(light.x, light.y, light.z).normalize()

const projectedVertices = boundsVertices.map((v) => {
  const newX = v.x + lightDir.x * (-v.y / lightDir.y)
  const newY = v.y + lightDir.y * (-v.y / lightDir.y)  // will be ~0
  const newZ = v.z + lightDir.z * (-v.y / lightDir.y)
  return new THREE.Vector3(newX, newY, newZ)
})
```

---

## 3. Calculating Plane Position

Weighted average of all projected vertices gives the center:

```js
const centerPos = projectedVertices
  .reduce((a, b) => a.add(b), new THREE.Vector3(0, 0, 0))
  .divideScalar(projectedVertices.length)

causticsPlane.current.position.set(centerPos.x, centerPos.y, centerPos.z)
```

---

## 4. Calculating Plane Scale

Find the maximum Euclidean distance from center to any projected vertex, then apply a correction factor:

```js
const scale = projectedVertices
  .map((p) =>
    Math.sqrt(
      Math.pow(p.x - centerPos.x, 2) + Math.pow(p.z - centerPos.z, 2)
    )
  )
  .reduce((a, b) => Math.max(a, b), 0)

// Correction factor to avoid pattern being cut at edges
const scaleCorrection = 1.75

causticsPlane.current.scale.set(
  scale * scaleCorrection,
  scale * scaleCorrection,
  scale * scaleCorrection
)
```

The `scaleCorrection = 1.75` is empirical. Drei's own Caustics component uses a more robust projection but the principle is the same.

---

## 5. Complete Implementation

Full positioning/scaling code for `useFrame`:

```js
useFrame((state) => {
  const { gl } = state
  const bounds = new THREE.Box3().setFromObject(mesh.current, true)
  const lightVec = new THREE.Vector3(light.x, light.y, light.z)
  const lightDir = lightVec.clone().normalize()

  // 1. Build bounding box vertices
  const boundsVertices = [
    new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.min.z),
    new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
    new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
    new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
    new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
    new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
    new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
    new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.max.z),
  ]

  // 2. Project onto ground (Y=0)
  const projected = boundsVertices.map((v) => {
    return new THREE.Vector3(
      v.x + lightDir.x * (-v.y / lightDir.y),
      0,
      v.z + lightDir.z * (-v.y / lightDir.y)
    )
  })

  // 3. Position = weighted center
  const center = projected
    .reduce((a, b) => a.clone().add(b), new THREE.Vector3())
    .divideScalar(projected.length)

  causticsPlane.current.position.set(center.x, 0.01, center.z)

  // 4. Scale = max distance from center * correction
  const maxDist = projected
    .map((p) => Math.sqrt(
      Math.pow(p.x - center.x, 2) + Math.pow(p.z - center.z, 2)
    ))
    .reduce((a, b) => Math.max(a, b), 0)

  const s = maxDist * 1.75
  causticsPlane.current.scale.set(s, s, s)

  // ... then proceed with the 3-pass render pipeline
})
```

### Notes

- The plane is positioned at `y = 0.01` (slightly above ground) to avoid z-fighting.
- The plane uses `rotation={[-Math.PI / 2, 0, 0]}` to face upward (Y-up).
- Moving the light source changes both the projected center and scale, making the caustic pattern follow the light naturally.
- For non-ground planes, replace `planeY = 0` with the target surface's Y coordinate in the projection formula.
