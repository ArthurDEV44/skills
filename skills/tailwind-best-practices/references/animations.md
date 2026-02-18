# Animations and Transitions

## Animations in @theme

Define animations and keyframes inside `@theme`:

```css
@import "tailwindcss";

@theme {
  --animate-fade-in: fade-in 0.3s ease-out;
  --animate-fade-in-up: fade-in-up 0.4s ease-out;
  --animate-fade-in-scale: fade-in-scale 0.3s ease-out;
  --animate-slide-in-right: slide-in-right 0.3s var(--ease-snappy);
  --animate-pulse-soft: pulse-soft 2s ease-in-out infinite;

  --ease-fluid: cubic-bezier(0.3, 0, 0, 1);
  --ease-snappy: cubic-bezier(0.2, 0, 0, 1);
  --ease-bounce: cubic-bezier(0.2, 0, 0, 1.4);

  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fade-in-up {
    from { opacity: 0; transform: translateY(1rem); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes fade-in-scale {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
  }

  @keyframes slide-in-right {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }

  @keyframes pulse-soft {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
}
```

Usage:

```html
<div class="animate-fade-in">Fades in</div>
<div class="animate-fade-in-up">Slides up while fading</div>
<div class="animate-fade-in-scale">Scales up while fading</div>
```

## Built-in Animations

| Class | Effect |
|-------|--------|
| `animate-spin` | Continuous rotation (spinners) |
| `animate-ping` | Ping effect (notifications) |
| `animate-pulse` | Subtle pulse (skeleton screens) |
| `animate-bounce` | Bounce effect (attention) |

## Transitions

```html
<!-- Basic transition -->
<button class="transition-colors duration-200 hover:bg-blue-600">
  Smooth color change
</button>

<!-- Multiple properties -->
<div class="transition-all duration-300 ease-out hover:scale-105 hover:shadow-lg">
  Scale + shadow on hover
</div>

<!-- Specific properties -->
<nav class="transition-transform duration-300 ease-snappy -translate-x-full data-[open]:translate-x-0">
  Sliding sidebar
</nav>
```

### Transition Utilities

| Class | Effect |
|-------|--------|
| `transition` | color, background-color, border-color, box-shadow, opacity, transform |
| `transition-all` | All properties |
| `transition-colors` | Color properties only |
| `transition-opacity` | Opacity only |
| `transition-shadow` | Box shadow only |
| `transition-transform` | Transform only |
| `transition-none` | Disable transitions |

### Duration

`duration-75`, `duration-100`, `duration-150`, `duration-200`, `duration-300`, `duration-500`, `duration-700`, `duration-1000`

### Timing

`ease-linear`, `ease-in`, `ease-out`, `ease-in-out` + custom `--ease-*` tokens.

## Conditional Animations

```html
<!-- Only animate on hover -->
<div class="hover:animate-pulse-soft">Pulses on hover</div>

<!-- Animate when entering viewport (with JS) -->
<div class="opacity-0 data-[visible]:animate-fade-in-up data-[visible]:opacity-100">
  Animates when visible
</div>

<!-- Reduce motion preference -->
<div class="animate-fade-in motion-reduce:animate-none">
  Respects prefers-reduced-motion
</div>
```

## Best Practices

1. **Respect `prefers-reduced-motion`** - Always add `motion-reduce:animate-none` for essential animations
2. **Use custom easings** - `--ease-fluid` and `--ease-snappy` feel more natural than `ease-in-out`
3. **Keep durations short** - 150-300ms for UI interactions, 300-500ms for content transitions
4. **Animate transforms and opacity** - GPU-accelerated, avoid animating layout properties
5. **Use `will-change-transform`** for heavy animations - But remove after animation ends

[Docs](https://tailwindcss.com/docs/animation)
