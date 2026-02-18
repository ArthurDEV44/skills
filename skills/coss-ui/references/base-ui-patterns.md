# coss ui - Base UI Foundation Patterns

Base UI (`@base-ui/react`) is the headless component library that powers all coss ui components. Understanding these patterns is essential for customizing, extending, or building new components.

## Table of Contents
- [useRender Hook](#userender-hook)
- [mergeProps Utility](#mergeprops-utility)
- [render Prop Deep Dive](#render-prop-deep-dive)
- [Data Attributes](#data-attributes)
- [State-Aware className and style](#state-aware-classname-and-style)
- [CSS Transitions and Animations](#css-transitions-and-animations)
- [Field and Form Validation](#field-and-form-validation)
- [React Hook Form Integration](#react-hook-form-integration)
- [Drawer Component](#drawer-component)
- [Accessibility Patterns](#accessibility-patterns)

---

## useRender Hook

The `useRender` hook is the core of Base UI's composition model. It powers the `render` prop on all components.

```tsx
import { useRender } from '@base-ui/react/use-render';
import { mergeProps } from '@base-ui/react/merge-props';

interface CounterState { odd: boolean }
interface CounterProps extends useRender.ComponentProps<'button', CounterState> {}

function Counter(props: CounterProps) {
  const { render, ...otherProps } = props;
  const [count, setCount] = React.useState(0);
  const odd = count % 2 === 1;
  const state = React.useMemo(() => ({ odd }), [odd]);

  const defaultProps: useRender.ElementProps<'button'> = {
    className: 'counter-btn',
    type: 'button',
    children: <>Counter: <span>{count}</span></>,
    onClick() { setCount((prev) => prev + 1) },
    'aria-label': `Count is ${count}, click to increase.`,
  };

  return useRender({
    defaultTagName: 'button',
    render,
    state,
    props: mergeProps<'button'>(defaultProps, otherProps),
  });
}
```

**Type signature:** `useRender.ComponentProps<TagName, State>` gives you the correct props type for any component built with useRender, including the `render`, `className`, and `style` props with state awareness.

---

## mergeProps Utility

Safely merges multiple props objects, properly composing event handlers and classNames:

```tsx
import { mergeProps } from '@base-ui/react/merge-props';

const merged = mergeProps<'button'>(
  { onClick: handleA, className: 'base' },
  { onClick: handleB, className: 'override' }
);
// Both onClick handlers will fire; className is 'base override'
```

Use `mergeProps` when building custom components that wrap Base UI primitives to avoid overwriting internal event handlers.

---

## render Prop Deep Dive

Every Base UI component accepts a `render` prop with two forms:

### Element form (composition)
Replace the underlying HTML element while keeping all component behavior:

```tsx
// Render a Button as a Link
<Button render={<Link href="/login" />}>Login</Button>

// Render a MenuTrigger as a custom Button
<MenuTrigger render={<Button variant="outline" />}>Open menu</MenuTrigger>

// Render a Badge as a Link
<Badge render={<Link href="/pricing" />}>New</Badge>
```

### Callback form (state access)
Access internal state and props for full control:

```tsx
<Popover.Popup
  render={(props, state) => (
    <motion.div
      {...(props as HTMLMotionProps<'div'>)}
      initial={false}
      animate={{
        opacity: state.open ? 1 : 0,
        scale: state.open ? 1 : 0.8,
      }}
    />
  )}
>
  Popup content
</Popover.Popup>
```

```tsx
// Counter with emoji based on state
<Counter
  render={(props, state) => (
    <button {...props}>
      {props.children}
      <span>{state.odd ? '👎' : '👍'}</span>
    </button>
  )}
/>
```

### nativeButton prop
When using `render` to replace a button with a non-button element, set `nativeButton={false}`:

```tsx
<Menu.Item render={<div />} nativeButton={false}>
  Non-button item
</Menu.Item>
```

---

## Data Attributes

Base UI components automatically apply data attributes reflecting their state. Use these for CSS styling.

### Common overlay attributes
| Attribute | Description |
|---|---|
| `data-open` | Present when open (Dialog, Popover, Menu, Tooltip, etc.) |
| `data-closed` | Present when closed |
| `data-nested` | Present when dialog is nested within another dialog |
| `data-nested-dialog-open` | Present when this dialog has nested dialogs open |

### Animation attributes
| Attribute | Description |
|---|---|
| `data-starting-style` | Present when animating in (mount/open) |
| `data-ending-style` | Present when animating out (unmount/close) |

### Toggle/selection attributes
| Attribute | Description |
|---|---|
| `data-checked` | Present when checked (Switch, Checkbox, Toggle) |
| `data-unchecked` | Present when unchecked |
| `data-selected` | Present when selected (Combobox item, Select item) |
| `data-highlighted` | Present when highlighted (via keyboard/hover) |

### Field/form attributes
| Attribute | Description |
|---|---|
| `data-valid` | Present when field is valid |
| `data-invalid` | Present when field is invalid |
| `data-dirty` | Present when field value changed from initial |
| `data-touched` | Present when field has been touched |
| `data-filled` | Present when field contains a value |
| `data-focused` | Present when field control is focused |
| `data-disabled` | Present when disabled |

### Directional attributes
| Attribute | Description |
|---|---|
| `data-orientation` | `'horizontal'` or `'vertical'` (Toolbar, Tabs) |
| `data-side` | `'top'`, `'bottom'`, `'left'`, `'right'` (positioned popups) |

### Using in Tailwind CSS
```tsx
// With Tailwind's arbitrary data attribute selector
<div className="data-[open]:opacity-100 data-[closed]:opacity-0">...</div>
<div className="data-[checked]:bg-primary data-[unchecked]:bg-muted">...</div>
```

---

## State-Aware className and style

Components accept `className` and `style` as functions receiving the component's state:

```tsx
// className as function
<Combobox.Item
  className={(state) => {
    if (state.selected) return 'bg-primary text-primary-foreground';
    if (state.highlighted) return 'bg-accent';
    return '';
  }}
>
  Option
</Combobox.Item>

// style as function
<Menu.Item
  style={(state) => ({
    backgroundColor: state.highlighted ? '#e0e0e0' : 'transparent',
    opacity: state.disabled ? 0.5 : 1,
  })}
>
  Item
</Menu.Item>
```

### Available state per component type

**Combobox.Item.State:** `selected`, `highlighted`, `disabled`
**Menu.Item.State / Menu.LinkItem.State:** `highlighted`, `disabled`
**Switch.State:** `checked`, `disabled`
**Field.*.State:** `valid`, `invalid`, `dirty`, `touched`, `filled`, `focused`, `disabled`
**Dialog.*.State / Popover.*.State:** `open`, `nested`, `nestedDialogOpen`
**Toolbar.*.State:** `orientation`

---

## CSS Transitions and Animations

### CSS transitions with data attributes

The recommended approach for animating Base UI components. Transitions can be cancelled midway for responsive UX:

```css
.Popup {
  box-sizing: border-box;
  padding: 1rem 1.5rem;
  background-color: canvas;
  transform-origin: var(--transform-origin);
  transition:
    transform 150ms,
    opacity 150ms;

  &[data-starting-style],
  &[data-ending-style] {
    opacity: 0;
    transform: scale(0.9);
  }
}
```

### Motion (framer-motion) integration

Use the callback `render` prop with `keepMounted` on Portal:

```tsx
<Popover.Root>
  <Popover.Trigger>Trigger</Popover.Trigger>
  <Popover.Portal keepMounted>
    <Popover.Positioner>
      <Popover.Popup
        render={(props, state) => (
          <motion.div
            {...(props as HTMLMotionProps<'div'>)}
            initial={false}
            animate={{
              opacity: state.open ? 1 : 0,
              scale: state.open ? 1 : 0.8,
            }}
          />
        )}
      >
        Content
      </Popover.Popup>
    </Popover.Positioner>
  </Popover.Portal>
</Popover.Root>
```

### onOpenChangeComplete callback

For running code after animations finish:

```tsx
<Dialog
  onOpenChangeComplete={(open) => {
    if (!open) resetForm();
  }}
>
  ...
</Dialog>
```

---

## Field and Form Validation

Base UI's Field component provides accessible form field wrappers with built-in validation state tracking.

### Field structure
```tsx
import { Field } from '@base-ui/react/field';

<Field.Root name="email" invalid={hasError}>
  <Field.Label>Email</Field.Label>
  <Field.Description>Your work email address.</Field.Description>
  <Field.Control placeholder="[email protected]" required />
  <Field.Error match="typeMismatch">Please enter a valid email.</Field.Error>
  <Field.Error match="valueMissing">Email is required.</Field.Error>
</Field.Root>
```

### Field.Error match values

The `match` prop accepts `boolean` or a `ValidityState` key:
- `true` - Always show (for external validation)
- `'badInput'`, `'customError'`, `'patternMismatch'`, `'rangeOverflow'`, `'rangeUnderflow'`
- `'stepMismatch'`, `'tooLong'`, `'tooShort'`, `'typeMismatch'`, `'valueMissing'`

### FieldValidity render prop

Access the full ValidityState inside Field:
```tsx
<Field.Validity>
  {(state) => state.valueMissing && <p>This field is required</p>}
</Field.Validity>
```

---

## React Hook Form Integration

Connect React Hook Form's Controller with Base UI Field components:

```tsx
import { useForm, Controller } from 'react-hook-form';
import { Field } from '@base-ui/react/field';

function MyForm() {
  const { control, handleSubmit } = useForm({
    defaultValues: { username: '' },
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Controller
        name="username"
        control={control}
        render={({
          field: { name, ref, value, onBlur, onChange },
          fieldState: { invalid, isTouched, isDirty, error },
        }) => (
          <Field.Root name={name} invalid={invalid} touched={isTouched} dirty={isDirty}>
            <Field.Label>Username</Field.Label>
            <Field.Description>Your display name.</Field.Description>
            <Field.Control
              placeholder="e.g. alice132"
              value={value}
              onBlur={onBlur}
              onValueChange={onChange}
              ref={ref}
            />
            <Field.Error match={!!error}>{error?.message}</Field.Error>
          </Field.Root>
        )}
      />
    </form>
  );
}
```

**Key points:**
- Forward `ref` from Controller to `Field.Control` for focus management during validation
- Map `fieldState.invalid` to `Field.Root`'s `invalid` prop
- Map `fieldState.isTouched` and `isDirty` to `touched` and `dirty` props
- Use `Field.Error match={!!error}` with `error?.message` for external validation display

---

## Drawer Component

Base UI provides a swipeable Drawer component used in coss ui's Sheet with mobile support:

```tsx
import { Drawer } from '@base-ui/react/drawer';

<Drawer modal={true} swipeDirection="down">
  <Drawer.Trigger>Open</Drawer.Trigger>
  <Drawer.Portal>
    <Drawer.Backdrop />
    <Drawer.Popup>
      <Drawer.Title>Drawer Title</Drawer.Title>
      <Drawer.Description>Drawer content here.</Drawer.Description>
      <Drawer.Close>Close</Drawer.Close>
    </Drawer.Popup>
  </Drawer.Portal>
</Drawer>
```

### Key Drawer props
| Prop | Type | Default | Description |
|---|---|---|---|
| `modal` | `boolean \| 'trap-focus'` | `true` | `true`: trap focus + lock scroll; `false`: non-modal; `'trap-focus'`: trap focus only |
| `swipeDirection` | `DrawerSwipeDirection` | `'down'` | Swipe direction to dismiss |
| `snapToSequentialPoints` | `boolean` | `false` | Disable velocity-based snap skipping |
| `disablePointerDismissal` | `boolean` | `false` | Prevent close on outside click |
| `onOpenChangeComplete` | `(open: boolean) => void` | - | Called after animations complete |

### Detached trigger with handle
```tsx
const handle = Drawer.createHandle();

<button onClick={() => handle.open()}>External Trigger</button>
<Drawer handle={handle}>
  <Drawer.Portal>
    <Drawer.Backdrop />
    <Drawer.Popup>Content</Drawer.Popup>
  </Drawer.Portal>
</Drawer>
```

---

## Accessibility Patterns

Base UI components are built with WAI-ARIA compliance. Key patterns:

### Focus management
- `initialFocus` - Control which element receives focus when opened (Dialog, Popover, Drawer)
- `finalFocus` - Control where focus returns when closed
- Accept `boolean`, `RefObject`, or `function` returning element based on interaction type (`mouse`, `touch`, `pen`, `keyboard`)

```tsx
<Dialog.Popup initialFocus={inputRef} finalFocus={triggerRef}>
  <input ref={inputRef} />
</Dialog.Popup>
```

### Keyboard navigation
- **Menu:** Arrow keys navigate items, Enter/Space selects, Escape closes
- **Combobox/Autocomplete:** Arrow keys navigate list, typing filters
- **Tabs:** Arrow keys switch tabs (respects orientation)
- **Toolbar:** Arrow keys navigate between items (respects orientation)
- **Dialog/AlertDialog:** Tab cycles through focusable elements, Escape closes

### focusableWhenDisabled
Toolbar items support `focusableWhenDisabled={true}` (default) to maintain keyboard navigation even when disabled:

```tsx
<Toolbar.Button disabled focusableWhenDisabled>
  Disabled but focusable
</Toolbar.Button>
```
