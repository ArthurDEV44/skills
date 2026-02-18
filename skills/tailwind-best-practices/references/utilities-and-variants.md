# Custom Utilities and Variants

## @utility: Custom Utility Classes

Register custom utilities that work with all Tailwind variants (hover, focus, responsive, dark, etc.).

### Basic Custom Utility

```css
@utility tab-4 {
  tab-size: 4;
}

/* Usage: <pre class="tab-4 hover:tab-8 lg:tab-4"> */
```

### Functional Utility (with values)

```css
@utility content-auto {
  content-visibility: auto;
}

@utility scrollbar-hidden {
  scrollbar-width: none;
  &::-webkit-scrollbar {
    display: none;
  }
}
```

### Component-Like Utility

```css
@utility btn {
  border-radius: 0.5rem;
  padding-inline: 1rem;
  padding-block: 0.5rem;
  font-weight: 600;
  background-color: var(--color-primary);
  color: white;
}

/* Overridable with other utilities: <button class="btn rounded-full px-6"> */
/* v4 sorts by property count, so single-property utilities override multi-property ones */
```

### Custom Container

v4 removed container config. Customize with `@utility`:

```css
@utility container {
  margin-inline: auto;
  padding-inline: 2rem;
  max-width: 80rem;
}
```

### vs @layer utilities (v3)

```css
/* v3 (deprecated) */
@layer utilities {
  .tab-4 { tab-size: 4; }
}

/* v4 (correct) */
@utility tab-4 {
  tab-size: 4;
}
```

Key difference: `@utility` integrates with the variant system. `@layer utilities` does not in v4.

## @custom-variant: Custom Variants

Create new variants usable with any utility class.

### Theme-Based Variant

```css
@custom-variant theme-midnight (&:where([data-theme="midnight"] *));

/* Usage: <div class="theme-midnight:bg-slate-900 theme-midnight:text-white"> */
```

### Custom State Variant

```css
@custom-variant checked (&:checked);

/* Usage: <input class="checked:bg-blue-500 checked:border-blue-600" type="checkbox"> */
```

### Parent State Variant

```css
@custom-variant sidebar-open (&:where([data-sidebar="open"] *));

/* Usage: <nav class="sidebar-open:translate-x-0 -translate-x-full"> */
```

### Group/Peer-Like Custom Variants

```css
@custom-variant dialog-open (&:where([data-state="open"] *));

/* Usage: <div class="dialog-open:opacity-100 opacity-0 transition-opacity"> */
```

### Override Default Variant (e.g., dark mode)

```css
/* Switch dark mode from media query to class-based */
@custom-variant dark (&:where(.dark, .dark *));
```

## @variant: Apply Variants in CSS

Use variants inside custom CSS (not for defining new variants):

```css
.my-element {
  background: white;
  @variant dark {
    background: black;
  }
  @variant hover {
    background: gray;
  }
}
```

## @plugin: Load JS Plugins

Load existing v3 JS plugins from CSS:

```css
@plugin "@tailwindcss/typography";
@plugin "@tailwindcss/forms";
@plugin "@tailwindcss/container-queries";
```

Load local plugin:

```css
@plugin "./my-plugin.js";
```

## @config: Legacy JS Config (Migration)

Load a v3 config file during migration:

```css
@config "../../tailwind.config.js";
```

Can coexist with `@theme`. CSS takes precedence over JS config for conflicts.

## Best Practices

1. **Prefer `@utility` over `@apply`** for reusable patterns
2. **Name utilities in kebab-case** matching Tailwind conventions
3. **Keep utilities single-responsibility** when possible
4. **Use `@custom-variant` for app-specific states** (sidebar-open, theme-dark, etc.)
5. **Load v3 plugins with `@plugin`** until they migrate to v4 CSS-native API

[Docs](https://tailwindcss.com/docs/functions-and-directives)
