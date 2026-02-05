# Multi-Light and Advanced Setups

## Multiple Directional/Spot Lights

Each additional light source requires its own:
- `lightCamera` (PerspectiveCamera)
- `shadowFBO` (with DepthTexture)
- Set of uniforms passed to the shader (position, direction, color, matrices)

### React Setup Pattern

```tsx
// For each light i:
const lightCamera_i = useMemo(() => {
  const cam = new THREE.PerspectiveCamera(90, 1.0, 0.1, 100);
  cam.fov = coneAngle_i;
  return cam;
}, [coneAngle_i]);

const shadowFBO_i = useFBO(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, {
  depth: true,
  depthTexture: new THREE.DepthTexture(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, THREE.FloatType),
});
```

### Rendering Multiple Shadow Maps

```tsx
useFrame((state) => {
  const { gl, camera, scene } = state;

  // Render shadow map for light 1
  lightCamera1.lookAt(target1);
  lightCamera1.position.copy(lightPos1);
  lightCamera1.updateMatrixWorld();
  lightCamera1.updateProjectionMatrix();
  gl.setRenderTarget(shadowFBO1);
  gl.clear(false, true, false);
  gl.render(scene, lightCamera1);

  // Render shadow map for light 2
  lightCamera2.lookAt(target2);
  lightCamera2.position.copy(lightPos2);
  lightCamera2.updateMatrixWorld();
  lightCamera2.updateProjectionMatrix();
  gl.setRenderTarget(shadowFBO2);
  gl.clear(false, true, false);
  gl.render(scene, lightCamera2);

  // Restore main render
  gl.setRenderTarget(null);
  gl.render(scene, camera);
});
```

### GLSL: Combining Multiple Lights

```glsl
// Duplicate uniforms for each light
uniform vec3 lightPosition1;
uniform vec3 lightDirection1;
uniform vec3 lightColor1;
uniform sampler2D shadowMap1;
uniform mat4 lightViewMatrix1;
uniform mat4 lightProjectionMatrix1;

uniform vec3 lightPosition2;
uniform vec3 lightDirection2;
uniform vec3 lightColor2;
uniform sampler2D shadowMap2;
uniform mat4 lightViewMatrix2;
uniform mat4 lightProjectionMatrix2;

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  // ... setup rayOrigin, rayDir, sceneDepth ...

  vec3 accLight1 = vec3(0.0);
  vec3 accLight2 = vec3(0.0);
  float transmittance1 = 5.0;
  float transmittance2 = 5.0;

  float t = STEP_SIZE * offset;

  for (int i = 0; i < NUM_STEPS; i++) {
    vec3 samplePos = rayOrigin + rayDir * t;
    if (t > sceneDepth || t > cameraFar) break;

    // Light 1 contribution
    float shadow1 = calculateShadow(samplePos, shadowMap1, lightViewMatrix1, lightProjectionMatrix1);
    if (shadow1 > 0.0) {
      // ... compute SDF shape, luminance, density for light 1 ...
      // accLight1 += ...
    }

    // Light 2 contribution
    float shadow2 = calculateShadow(samplePos, shadowMap2, lightViewMatrix2, lightProjectionMatrix2);
    if (shadow2 > 0.0) {
      // ... compute SDF shape, luminance, density for light 2 ...
      // accLight2 += ...
    }

    t += STEP_SIZE;
  }

  vec3 finalColor = inputColor.rgb + accLight1 + accLight2;
  outputColor = vec4(finalColor, 1.0);
}
```

## Dynamic Light Following Camera (Space Scene Pattern)

For omnidirectional-looking effects (e.g., a star), point the light toward the camera:

```tsx
const lightDirection = new THREE.Vector3()
  .subVectors(camera.position, lightPosition)
  .normalize();

lightCamera.position.copy(lightPosition);
lightCamera.lookAt(camera.position);
lightCamera.fov = 90; // wide FOV for maximum shadow coverage
lightCamera.updateMatrixWorld();
lightCamera.updateProjectionMatrix();
```

This ensures volumetric light is always visible from the viewer's perspective, while shadows cast by objects between the light and camera remain correct.

## Performance Considerations

- Each additional light doubles shadow map rendering cost
- Share the same `NUM_STEPS` loop for all lights to avoid multiple ray traversals
- Use lower shadow map resolution (256x256) per light when using multiple lights
- Consider reducing `NUM_STEPS` proportionally with light count
