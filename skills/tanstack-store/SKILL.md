---
name: tanstack-store
description: "TanStack Store framework-agnostic reactive state management with Store, Derived, Effect, and batch. Use when writing, reviewing, or refactoring code that involves: (1) Creating or managing state with TanStack Store or @tanstack/store, (2) Using useStore hook from @tanstack/react-store, @tanstack/vue-store, @tanstack/solid-store, @tanstack/angular-store, @tanstack/svelte-store, or @tanstack/preact-store, (3) Creating derived/computed state with Derived class, (4) Managing side effects with Effect class, (5) Batching state updates with batch(), (6) Choosing a lightweight state management solution for framework-agnostic libraries."
---

# TanStack Store

Framework-agnostic reactive state management. Core primitives: `Store`, `Derived`, `Effect`, `batch`.

## Installation

```sh
# Core (vanilla JS/TS)
npm install @tanstack/store

# Framework adapters
npm install @tanstack/react-store    # React 16.8+ (ReactDOM only)
npm install @tanstack/vue-store      # Vue 2/3
npm install @tanstack/solid-store    # Solid / SolidStart
npm install @tanstack/angular-store  # Angular 19+
npm install @tanstack/svelte-store   # Svelte 5
npm install @tanstack/preact-store   # Preact 10+
```

## Quick Start

### Store — Reactive state container

```typescript
import { Store } from '@tanstack/store';

const count = new Store(0);

count.state;              // 0
count.setState(() => 1);  // update
count.state;              // 1

const unsub = count.subscribe(() => {
  console.log(count.state);
});
unsub(); // cleanup
```

### Derived — Lazy computed values

```typescript
import { Store, Derived } from '@tanstack/store';

const count = new Store(0);
const double = new Derived({
  fn: () => count.state * 2,
  deps: [count],
});

const unmount = double.mount(); // required to start listening
double.state; // 0
count.setState(() => 5);
double.state; // 10
unmount();
```

Access previous value and dependency values via `fn` props:

```typescript
const derived = new Derived({
  fn: ({ prevVal, prevDepVals, currDepVals }) => {
    // prevVal: previous derived value (undefined on first run)
    // prevDepVals: previous dep values (undefined on first run)
    // currDepVals: current dep values (same order as deps array)
    return currDepVals[0] + (prevVal ?? 0);
  },
  deps: [count],
});
```

### Effect — Side effects on state changes

```typescript
import { Store, Effect } from '@tanstack/store';

const count = new Store(0);
const effect = new Effect({
  fn: () => console.log('count:', count.state),
  deps: [count],
  eager: true, // run immediately on mount (default: false)
});

const unmount = effect.mount();
unmount(); // cleanup
```

### batch — Coalesce updates

```typescript
import { batch } from '@tanstack/store';

batch(() => {
  store.setState(() => 1);
  store.setState(() => 2);
}); // subscribers fire once with final state
```

### Store options — updateFn and onUpdate

```typescript
// Transform updates before applying
const count = new Store(12, {
  updateFn: (prev) => (updater) => updater(prev) + prev,
});
count.setState(() => 12); // state === 24

// Primitive derived state via onUpdate
let double = 0;
const count2 = new Store(0, {
  onUpdate: () => { double = count2.state * 2; },
});
```

## React Adapter

```tsx
import { Store, useStore } from '@tanstack/react-store';

const store = new Store({ dogs: 0, cats: 0 });

function Counter({ animal }: { animal: 'dogs' | 'cats' }) {
  // Only re-renders when selected slice changes
  const count = useStore(store, (s) => s[animal]);
  return <span>{count}</span>;
}
```

`useStore` accepts both `Store` and `Derived`. Use `shallow` from `@tanstack/react-store` for shallow equality comparison on selected objects/arrays.

## References

- **Core API details**: See [references/core-api.md](references/core-api.md) for full Store, Derived, Effect, batch API signatures and types.
- **React adapter details**: See [references/react-adapter.md](references/react-adapter.md) for useStore, shallow, and full React example.
- **Official docs**: https://tanstack.com/store/latest
