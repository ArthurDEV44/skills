# SDF Light Shaping

Use Signed Distance Functions to constrain volumetric light to arbitrary shapes. Within the raymarching loop, evaluate the SDF at each sample point and use the result to determine whether to accumulate light.

## Common Light Shapes

### Cone (Spotlight)

```glsl
float sdCone(vec3 p, vec3 origin, vec3 dir, float halfAngle) {
  vec3 toP = p - origin;
  float projLen = dot(toP, dir);
  if (projLen < 0.0) return length(toP);
  float radius = projLen * tan(halfAngle);
  float perpDist = length(toP - dir * projLen);
  return perpDist - radius;
}
```

### Cylinder

```glsl
float sdCylinder(vec3 p, vec3 axisOrigin, vec3 axisDir, float radius) {
  vec3 p_to_origin = p - axisOrigin;
  float projectionLength = dot(p_to_origin, axisDir);
  vec3 closestPointOnAxis = axisOrigin + axisDir * projectionLength;
  float distanceToAxis = length(p - closestPointOnAxis);
  return distanceToAxis - radius;
}
```

### Sphere

```glsl
float sdSphere(vec3 p, vec3 center, float radius) {
  return length(p - center) - radius;
}
```

### Torus

```glsl
float sdTorus(vec3 p, vec3 center, vec3 axis, float majorR, float minorR) {
  vec3 q = p - center;
  float projOnAxis = dot(q, axis);
  vec3 projPerp = q - axis * projOnAxis;
  float distToRing = length(projPerp) - majorR;
  return length(vec2(distToRing, projOnAxis)) - minorR;
}
```

## Integration Pattern

```glsl
float smoothEdgeWidth = 0.1;

// Inside raymarching loop:
float sdfVal = sdCone(samplePos, lightPos, lightDir, halfConeAngleRad);
float shapeFactor = smoothstep(0.0, -smoothEdgeWidth, sdfVal);

if (shapeFactor < 0.1) {
  t += STEP_SIZE;
  continue;
}
```

`smoothstep` provides soft edges. Increase `smoothEdgeWidth` for softer falloff.

## Adding Fog/Noise to Shapes

Replace the hard shape factor with noise-modulated density:

```glsl
float sdfVal = sdCone(samplePos, lightPos, lightDir, halfConeAngleRad);
float shapeFactor = -sdfVal + fbm(samplePos); // noise breaks up the clean edge

if (shapeFactor < 0.1) {
  t += STEP_SIZE;
  continue;
}
```

This creates organic, cloud-like light beams. Refer to Inigo Quilez's SDF reference for more shapes: https://iquilezles.org/articles/distfunctions/
