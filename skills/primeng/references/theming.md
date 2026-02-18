# PrimeNG Theming

PrimeNG v20 uses a three-layer design token architecture powered by `@primeuix/themes`.

## Token Architecture

### Layer 1: Primitive Tokens

Raw color palettes and base values. Not used directly by components.

```
{emerald.500}  →  #10b981
{zinc.900}     →  #18181b
{indigo.400}   →  #818cf8
```

Available palettes: `slate`, `gray`, `zinc`, `neutral`, `stone`, `red`, `orange`, `amber`, `yellow`, `lime`, `green`, `emerald`, `teal`, `cyan`, `sky`, `blue`, `indigo`, `violet`, `purple`, `fuchsia`, `pink`, `rose`

### Layer 2: Semantic Tokens

Meaningful aliases that map to primitives. Shared across all components.

```typescript
semantic: {
  primary: { 50: '{indigo.50}', ..., 950: '{indigo.950}' },
  formField: {
    paddingX: '0.75rem',
    paddingY: '0.5rem',
    borderRadius: '{border.radius.md}',
    focusRing: { width: '0', style: 'none', color: 'transparent', offset: '0' }
  },
  colorScheme: {
    light: { surface: { 0: '#ffffff', ... }, primary: { color: '{primary.500}' } },
    dark:  { surface: { 0: '#ffffff', ... }, primary: { color: '{primary.400}' } }
  }
}
```

### Layer 3: Component Tokens

Per-component overrides. Scoped to a single component.

```typescript
const MyPreset = definePreset(Aura, {
  components: {
    button: {
      borderRadius: '2rem',
      paddingX: '1.5rem',
      paddingY: '0.75rem'
    },
    card: {
      borderRadius: '1rem',
      shadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
    },
    inputtext: {
      borderRadius: '0.5rem'
    }
  }
});
```

## definePreset

Extend a base preset with customizations:

```typescript
import { definePreset } from '@primeuix/themes';
import Aura from '@primeuix/themes/aura';

const MyPreset = definePreset(Aura, {
  // Override primitive palette
  primitive: {
    brand: {
      50: '#f0f4ff', 100: '#dbe4ff', 200: '#bac8ff',
      300: '#91a7ff', 400: '#748ffc', 500: '#5c7cfa',
      600: '#4c6ef5', 700: '#4263eb', 800: '#3b5bdb',
      900: '#364fc7', 950: '#2b3f9e'
    }
  },

  // Override semantic tokens
  semantic: {
    primary: {
      50: '{brand.50}', 100: '{brand.100}', 200: '{brand.200}',
      300: '{brand.300}', 400: '{brand.400}', 500: '{brand.500}',
      600: '{brand.600}', 700: '{brand.700}', 800: '{brand.800}',
      900: '{brand.900}', 950: '{brand.950}'
    },
    // Font
    fontFamily: '"Inter", sans-serif',
    // Border radius
    borderRadius: '0.5rem'
  },

  // Override component tokens
  components: {
    button: { borderRadius: '2rem' },
    card: { borderRadius: '1rem' }
  }
});
```

## Dark Mode

### System Preference (default)

```typescript
providePrimeNG({
  theme: {
    preset: Aura,
    options: { darkModeSelector: 'system' }
  }
})
```

### Class-Based Toggle

```typescript
// Config
options: { darkModeSelector: '.dark-mode' }
```

```typescript
// Toggle in component
toggleDarkMode() {
  document.documentElement.classList.toggle('dark-mode');
}
```

### Disable Dark Mode

```typescript
options: { darkModeSelector: false }
```

## Unstyled Mode

Strip all visual styles — bring your own CSS:

```typescript
providePrimeNG({
  theme: {
    preset: Aura,
    options: { darkModeSelector: false }
  },
  unstyled: true  // Removes all PrimeNG styles
})
```

Components render with structural HTML only — add classes via `styleClass`, `[ngClass]`, or Tailwind.

## CSS Layers

Control specificity ordering with CSS `@layer`:

```typescript
providePrimeNG({
  theme: {
    preset: Aura,
    options: {
      cssLayer: {
        name: 'primeng',
        order: 'tailwind-base, primeng, tailwind-utilities'
      }
    }
  }
})
```

This ensures Tailwind utility classes can override PrimeNG styles without `!important`.

## Runtime Theme Switching

Use `PrimeNG` service to change themes at runtime:

```typescript
import { PrimeNG } from 'primeng/config';
import Lara from '@primeuix/themes/lara';
import Aura from '@primeuix/themes/aura';

@Component({ ... })
export class ThemeSwitcher {
  constructor(private primeng: PrimeNG) {}

  switchToLara() {
    this.primeng.theme.set({ preset: Lara });
  }

  switchToAura() {
    this.primeng.theme.set({ preset: Aura });
  }

  // Update specific tokens at runtime
  updatePrimaryColor() {
    this.primeng.theme.set({
      preset: definePreset(Aura, {
        semantic: {
          primary: {
            50: '{violet.50}', 100: '{violet.100}', /* ... */
            950: '{violet.950}'
          }
        }
      })
    });
  }
}
```

## CSS Variable Reference

All tokens are exposed as CSS variables with the configured prefix (default `p`):

```css
/* Semantic */
var(--p-primary-color)
var(--p-primary-contrast-color)
var(--p-surface-0)       /* Background */
var(--p-surface-900)     /* Text */
var(--p-text-color)
var(--p-text-muted-color)
var(--p-border-color)

/* Component */
var(--p-button-border-radius)
var(--p-inputtext-padding-x)
var(--p-card-border-radius)
```

Override directly in CSS when needed:

```css
:root {
  --p-primary-color: #6366f1;
  --p-border-radius: 8px;
}
```

## Scoped Component Styling

Use `dt()` function in component tokens to reference design tokens:

```typescript
const MyPreset = definePreset(Aura, {
  components: {
    button: {
      colorScheme: {
        light: {
          root: {
            primary: {
              background: '{primary.500}',
              hoverBackground: '{primary.600}',
              color: '#ffffff'
            }
          }
        }
      }
    }
  }
});
```
