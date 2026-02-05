# Dynamic & Interactive Effects

## Table of Contents

- [Progressive Depixelation](#progressive-depixelation)
- [Pixelating Mouse Trail](#pixelating-mouse-trail)
- [FBO & Ping-Pong Rendering](#fbo--ping-pong-rendering)
- [Combining Techniques](#combining-techniques)

---

## Progressive Depixelation

Animate pixelation level over time or progress, de-pixelating the screen row-by-row.

### Core Logic

```glsl
uniform float progress;  // 0.0 to 1.0

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  float LEVELS = 5.0;
  float basePixelSize = pow(2.0, LEVELS);  // e.g. 32

  float currentLevel = floor(progress * LEVELS);
  float currentPixelSize = max(basePixelSize / pow(2.0, currentLevel), 1.0);

  float currentPixelsPerRow = ceil(resolution.x / currentPixelSize);
  float currentPixelsPerCol = ceil(resolution.y / currentPixelSize);
  float currentTotalPixels = currentPixelsPerRow * currentPixelsPerCol;

  float levelProgress = fract(progress * LEVELS) * currentTotalPixels;
  float currentRowInLevel = floor(levelProgress / currentPixelsPerRow);
  float currentPixelInRow = mod(levelProgress, currentPixelsPerRow);

  vec2 gridPos = floor(uv * resolution / currentPixelSize);
  float row = floor(currentPixelsPerCol - gridPos.y - 1.0);
  float posInRow = floor(gridPos.x);

  float nextPixelSize = max(currentPixelSize / 2.0, 1.0);
  vec2 normalizedPixelSize;

  // Already-processed rows: use finer pixel size
  if (row < currentRowInLevel) {
    normalizedPixelSize = vec2(nextPixelSize) / resolution;
  }
  // Current row, processed columns: use finer pixel size
  else if (row == currentRowInLevel && posInRow <= currentPixelInRow) {
    normalizedPixelSize = vec2(nextPixelSize) / resolution;
  }
  // Not yet processed: use current (coarser) pixel size
  else {
    normalizedPixelSize = vec2(currentPixelSize) / resolution;
  }

  // Final pass: no pixelation
  if (currentPixelSize <= 1.0) {
    outputColor = texture2D(inputBuffer, uv);
    return;
  }

  vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
  outputColor = texture2D(inputBuffer, uvPixel);
}
```

### R3F Integration

Drive `progress` with a ref and useFrame:

```jsx
const progressRef = useRef(0);

useFrame((_, delta) => {
  progressRef.current = Math.min(progressRef.current + delta * 0.2, 1.0);
  effectRef.current.uniforms.get("progress").value = progressRef.current;
});
```

Or link to scroll position, button click, or any other trigger.

---

## Pixelating Mouse Trail

Dynamic pixelation and distortion that follows mouse movement. Uses a mouse trail FBO texture to drive the effect.

### Mouse Trail FBO Setup (R3F)

```jsx
import { useFBO } from "@react-three/drei";
import { createPortal, useThree } from "@react-three/fiber";

const PixelatingMouseTrail = () => {
  const mouseTrail = useMemo(() => new THREE.Scene(), []);
  const mouseTrailFBO = useFBO({
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
  });

  const smoothedMouse = useRef(new THREE.Vector2());
  const smoothedMouseDirection = useRef(new THREE.Vector2());

  useFrame((state) => {
    const { gl, camera } = state;

    // Render mouse trail to FBO
    gl.setRenderTarget(mouseTrailFBO);
    gl.render(mouseTrail, camera);
    gl.setRenderTarget(null);
  });

  const { camera } = useThree();

  return (
    <>
      {createPortal(
        <MouseTrail mouse={smoothedMouse} />,
        mouseTrail,
        { camera }
      )}
      <YourScene />
      <PixelatingMouseTrailEffect
        mouseTrailTexture={mouseTrailFBO.texture}
        mouse={smoothedMouse.current}
        mouseDirection={smoothedMouseDirection.current}
      />
    </>
  );
};
```

### Fragment Shader

```glsl
uniform sampler2D mouseTrailTexture;
uniform vec2 mouse;
uniform vec2 mouseDirection;

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec4 mouseTrailOG = texture2D(mouseTrailTexture, uv);
  float distanceToCenter = 1.0 - distance(uv, mouse);

  // Pixel size increases near the trail
  float pixelSize = 32.0 + length(mouseTrailOG.rg) * distanceToCenter;
  vec2 normalizedPixelSize = pixelSize / resolution;
  vec2 uvPixel = normalizedPixelSize * floor(uv / normalizedPixelSize);
  vec4 mouseTrail = texture2D(mouseTrailTexture, uvPixel);

  // Distort UVs based on mouse direction
  vec2 textureUV = uv;
  textureUV -= mouseTrail.rg * distanceToCenter * mouseDirection;

  vec4 color = texture2D(inputBuffer, textureUV);
  vec4 trailColor = vec4(0.9, 0.9, 0.9, 0.1);
  outputColor = max(color, mix(color, trailColor, mouseTrail.r));
}
```

**Key idea:** The mouse trail FBO encodes movement direction in R (vertical) and G (horizontal) channels. This drives both pixel size and UV distortion.

---

## FBO & Ping-Pong Rendering

Technique for creating persistent visual effects (trails, feedback loops).

### Concept

Two FBOs alternate as read/write targets each frame:
1. Frame N: Read from FBO-A, write to FBO-B
2. Frame N+1: Read from FBO-B, write to FBO-A

This creates a feedback loop where previous frames persist and fade.

### R3F Setup

```jsx
const fboA = useFBO(/* options */);
const fboB = useFBO(/* options */);
const currentFBO = useRef(0);

useFrame(({ gl, camera }) => {
  const readFBO = currentFBO.current === 0 ? fboA : fboB;
  const writeFBO = currentFBO.current === 0 ? fboB : fboA;

  // Pass readFBO.texture as uniform to the trail material
  trailMaterial.uniforms.previousFrame.value = readFBO.texture;

  gl.setRenderTarget(writeFBO);
  gl.render(trailScene, camera);
  gl.setRenderTarget(null);

  // Pass writeFBO.texture to the post-processing effect
  effect.uniforms.get("trailTexture").value = writeFBO.texture;

  currentFBO.current = 1 - currentFBO.current; // swap
});
```

### Use Cases

- Mouse trails with fade-out
- Motion blur effects
- Feedback/echo distortion
- Any effect requiring temporal persistence

---

## Combining Techniques

The power of post-processing comes from combining simple techniques:

| Combination | Result |
|---|---|
| Pixelation + SDF + Lighting | Lego bricks with 3D studs |
| Pixelation + Stagger + Border | LED panel display |
| Pixelation + Rotation + Noise | Crochet/knitted fabric |
| Sine distortion + Normal + Specular | Frosted/fluted glass |
| FBO + Pixelation + Mouse | Interactive pixel trail |
| Pixelation + Threshold matrix | Complex woven/stripe patterns |
| Pixelation + ASCII texture | Text-based rendering |
| Pixelation + Color quantization | Retro/posterized look |

### Polish Techniques (add sparingly)

- **Noise:** Break up uniform areas, add organic feel
- **Hue shift:** Per-cell color variation for realism
- **Chromatic aberration:** Split RGB channels at slight UV offsets
- **Gaussian blur:** Soften hard edges, add depth
- **Color quantization:** `floor(color * levels + 0.5) / levels`
- **Clamp min/max brightness:** Keep patterns visible in extreme dark/light
- **Time-based animation:** Drive any uniform with `time` for motion
