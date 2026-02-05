# Workarounds and Advanced Patterns

## Table of Contents

1. [Displaced Mesh Normal Issue](#1-displaced-mesh-normal-issue)
2. [Per-Mesh Material Override via Traversal](#2-per-mesh-material-override-via-traversal)
3. [Ground Material with Displacement](#3-ground-material-with-displacement)
4. [Performance Considerations](#4-performance-considerations)
5. [Customization Recipes](#5-customization-recipes)

---

## 1. Displaced Mesh Normal Issue

**Problem:** When a mesh is displaced in its vertex shader (e.g., Perlin noise terrain), the Normal buffer captured via `scene.overrideMaterial` does not contain displacement information. The Sobel filter sees a flat plane and draws no inner outlines.

**Solution:** Create a separate custom Normal material for displaced meshes that replicates the displacement and recomputes normals accordingly.

```glsl
// In the displaced mesh's custom Normal material vertex shader:
// 1. Apply the same displacement as the original material
// 2. Recompute normals using partial derivatives (orthogonal method)

vec3 displacedPosition = position + normal * noiseValue;

// Recompute normals from displaced neighbors
float offset = 0.01;
vec3 tangent = orthogonal(normal);
vec3 bitangent = cross(normal, tangent);

vec3 neighbour1 = position + tangent * offset;
vec3 neighbour2 = position + bitangent * offset;

// Apply same displacement to neighbors
vec3 displacedNeighbour1 = neighbour1 + normal * noise(neighbour1);
vec3 displacedNeighbour2 = neighbour2 + normal * noise(neighbour2);

vec3 displacedTangent = displacedNeighbour1 - displacedPosition;
vec3 displacedBitangent = displacedNeighbour2 - displacedPosition;
vec3 recomputedNormal = normalize(cross(displacedTangent, displacedBitangent));
```

---

## 2. Per-Mesh Material Override via Traversal

Instead of `scene.overrideMaterial` (which applies one material globally), traverse the scene to assign different Normal materials per mesh:

```js
useFrame((state) => {
  const { gl, scene, camera } = state;

  // Store original materials
  const materials = [];

  gl.setRenderTarget(normalRenderTarget);

  scene.traverse((obj) => {
    if (obj.isMesh) {
      materials.push(obj.material);
      if (obj.name === "ground") {
        // Displaced mesh gets its own normal material
        obj.material = GroundNormalMaterial;
        obj.material.uniforms.uTime.value = clock.elapsedTime;
        obj.material.uniforms.lightPosition.value = lightPosition;
      } else {
        obj.material = CustomNormalMaterial;
        obj.material.uniforms.lightPosition.value = lightPosition;
      }
    }
  });

  gl.render(scene, camera);

  // Restore original materials
  scene.traverse((obj) => {
    if (obj.isMesh) {
      obj.material = materials.shift();
    }
  });
});
```

---

## 3. Ground Material with Displacement

Extend `MeshStandardMaterial` using `onBeforeCompile` to inject custom vertex displacement while keeping PBR lighting:

```js
class GroundMaterial extends THREE.MeshStandardMaterial {
  constructor() {
    super();
    this.uniforms = {
      uTime: { value: 0.0 },
    };
  }

  onBeforeCompile(shader) {
    shader.uniforms = { ...shader.uniforms, ...this.uniforms };

    // Prepend noise functions to vertex shader
    shader.vertexShader =
      `
      uniform float uTime;
      // Add Perlin noise, displacement functions here
    ` + shader.vertexShader;

    // Inject displacement after clipping planes
    shader.vertexShader = shader.vertexShader.replace(
      "#include <clipping_planes_vertex>",
      `#include <clipping_planes_vertex>
      // Apply displacement and recompute normals here
      `
    );
  }
}
```

This approach preserves shadows following the displaced geometry because `MeshStandardMaterial` uses the modified normals for its lighting model.

---

## 4. Performance Considerations

- **Render targets:** The pipeline requires 3 render passes (depth, normal, final composite). Minimize resolution of depth/normal targets on lower-end devices.
- **Sobel filter:** 9 texture samples per buffer (18 total for depth + normal). This is lightweight but scales with resolution.
- **Shadow patterns:** `mod()` operations are cheap. Crosshatched patterns add negligible cost.
- **Scene traversal:** Material swap via `scene.traverse` is O(n) per frame. Cache mesh references to avoid traversal overhead.

---

## 5. Customization Recipes

### Blueprint style
- Set background to deep blue
- Set `outlineColor` to white `vec4(1.0)`
- Disable shadow patterns
- Keep outline displacement

### Manga / comic style
- Increase `outlineThickness` to 4-5
- Use tonal shadows instead of crosshatched
- Reduce displacement amplitude to 0

### Engraving style
- Use only diagonal stripe shadows at multiple angles
- Set very thin outlines (thickness 1-2)
- Monochrome palette (sepia or black/white)

### Colored outlines
- Replace `outlineColor` with a color derived from `pixelColor`:
  ```glsl
  vec4 outlineColor = vec4(pixelColor.rgb * 0.3, 1.0);
  ```
