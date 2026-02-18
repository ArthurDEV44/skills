# New Utilities in v4

## 3D Transforms

### Perspective

```html
<div class="perspective-dramatic">
  <div class="rotate-y-12 translate-z-8">3D card</div>
</div>

<div class="perspective-normal">
  <div class="rotate-x-6 -translate-z-4">Subtle 3D</div>
</div>
```

| Class | Value |
|-------|-------|
| `perspective-dramatic` | 100px |
| `perspective-near` | 300px |
| `perspective-normal` | 500px |
| `perspective-midrange` | 800px |
| `perspective-distant` | 1200px |
| `perspective-none` | none |

### Transform Style

```html
<!-- Children preserve 3D space -->
<div class="transform-3d perspective-normal">
  <div class="rotate-y-45">I'm in 3D!</div>
</div>

<!-- Children are flat (default) -->
<div class="transform-flat">
  <div class="rotate-y-45">I'm flat</div>
</div>
```

### 3D Rotation and Translation

```html
<div class="rotate-x-12 rotate-y-6 translate-z-8">
  Full 3D transform
</div>
```

## Gradients

### Linear Gradients

```html
<div class="bg-linear-to-r from-blue-500 to-purple-600">
  Left to right gradient
</div>

<div class="bg-linear-to-br from-rose-500 via-pink-500 to-purple-600">
  Diagonal with mid stop
</div>
```

v4 renamed `bg-gradient-to-*` to `bg-linear-to-*`.

### Custom Angle Gradients

```html
<div class="bg-linear-[25deg,red_5%,yellow_60%,lime_90%,teal]">
  Custom angle gradient
</div>

<div class="bg-linear-(--my-gradient)">
  CSS variable gradient
</div>
```

### Conic and Radial Gradients

```html
<div class="bg-conic from-red-500 via-yellow-500 to-red-500">
  Color wheel
</div>

<div class="bg-radial from-white to-blue-500">
  Radial gradient
</div>
```

## Text Shadow

```html
<h1 class="text-shadow-sm text-white">Subtle text shadow</h1>
<h1 class="text-shadow-md text-white">Medium text shadow</h1>
<h1 class="text-shadow-lg text-white">Large text shadow</h1>

<!-- Custom value -->
<h1 class="text-shadow-[0_2px_4px_rgba(0,0,0,0.5)]">Custom shadow</h1>
```

## Inset Shadows

```html
<div class="inset-shadow-sm">Subtle inner shadow</div>
<div class="inset-shadow-md">Medium inner shadow</div>
```

## Color Mixing

Use oklch with opacity directly:

```html
<div class="bg-blue-500/50">50% opacity blue</div>
<div class="bg-black/10">10% opacity black overlay</div>
<div class="border-white/20">Subtle white border</div>
```

## Field Sizing

```html
<textarea class="field-sizing-content">
  Auto-resizes to content
</textarea>
```

## Arbitrary Values

Works with any utility:

```html
<div class="w-[calc(100%-2rem)]">Calc width</div>
<div class="grid-cols-[1fr_2fr_1fr]">Custom grid</div>
<div class="bg-[#1da1f2]">Exact hex color</div>
<div class="text-[clamp(1rem,2.5vw,2rem)]">Fluid typography</div>
<div class="p-[var(--custom-padding)]">CSS variable</div>
```

## Arbitrary Properties

For one-off CSS properties with no utility:

```html
<div class="[mask-type:luminance]">Custom CSS property</div>
<div class="[--scroll-offset:56px]">Set CSS variable</div>
```

[Docs](https://tailwindcss.com/docs/upgrade-guide)
