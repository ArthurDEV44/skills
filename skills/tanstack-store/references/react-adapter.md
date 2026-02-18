# TanStack Store React Adapter

## Table of Contents

- [Installation](#installation)
- [useStore hook](#usestore-hook)
- [shallow function](#shallow-function)
- [Full React example](#full-react-example)
- [TanStack Form integration](#tanstack-form-integration)

## Installation

```sh
npm install @tanstack/react-store
```

Compatible with React v16.8+ (ReactDOM only).

## useStore hook

```typescript
import { useStore } from '@tanstack/react-store';

// With a Store
function useStore<TState, TSelected = TState>(
  store: Store<TState>,
  selector?: (state: TState) => TSelected,
  equalityFn?: (a: TSelected, b: TSelected) => boolean
): TSelected;

// With a Derived
function useStore<TState, TSelected = TState>(
  store: Derived<TState>,
  selector?: (state: TState) => TSelected,
  equalityFn?: (a: TSelected, b: TSelected) => boolean
): TSelected;
```

- Accepts both `Store` and `Derived` instances.
- `selector` extracts a slice of state — component only re-renders when the selected value changes.
- `equalityFn` (3rd argument) — custom equality comparison. Pass `shallow` for object/array selectors.
- Without a selector, returns the full state and re-renders on any change.

### Usage

```tsx
import { useStore } from '@tanstack/react-store';
import { Store } from '@tanstack/store';

const store = new Store({ count: 0, name: 'World' });

function Counter() {
  // Only re-renders when `count` changes
  const count = useStore(store, (s) => s.count);
  return <span>{count}</span>;
}
```

## shallow function

```typescript
import { shallow } from '@tanstack/react-store';

function shallow<T>(objA: T, objB: T): boolean;
```

Shallow equality comparison. Pass as 3rd argument to `useStore` to prevent re-renders when selecting objects/arrays that are structurally equal:

```tsx
import { useStore, shallow } from '@tanstack/react-store';

// Without shallow: re-renders every time store changes (new object ref each time)
const info = useStore(store, (s) => ({ count: s.count, name: s.name }));

// With shallow: only re-renders when count or name actually change
const info = useStore(store, (s) => ({ count: s.count, name: s.name }), shallow);
```

## Full React example

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { useStore } from '@tanstack/react-store';
import { Store } from '@tanstack/store';

export const store = new Store({
  dogs: 0,
  cats: 0,
});

const Display = ({ animal }: { animal: 'dogs' | 'cats' }) => {
  const count = useStore(store, (state) => state[animal]);
  return <div>{`${animal}: ${count}`}</div>;
};

const updateState = (animal: 'dogs' | 'cats') => {
  store.setState((state) => ({
    ...state,
    [animal]: state[animal] + 1,
  }));
};

const Increment = ({ animal }: { animal: 'dogs' | 'cats' }) => (
  <button onClick={() => updateState(animal)}>My Friend Likes {animal}</button>
);

function App() {
  return (
    <div>
      <h1>How many of your friends like cats or dogs?</h1>
      <Increment animal="dogs" />
      <Display animal="dogs" />
      <Increment animal="cats" />
      <Display animal="cats" />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
```

## TanStack Form integration

TanStack Form uses TanStack Store as its reactivity engine. Every `FormApi` instance exposes a `form.store` that is a standard TanStack `Store`. Use `useStore` to subscribe to specific form state slices with fine-grained re-renders.

### Subscribing to form state

```tsx
import { useStore } from '@tanstack/react-store';
// or: import { useStore } from '@tanstack/form-core';

// Subscribe to a specific form value
const firstName = useStore(form.store, (state) => state.values.firstName);

// Subscribe to form errors
const errors = useStore(form.store, (state) => state.errorMap);

// Subscribe to submission state
const isSubmitting = useStore(form.store, (state) => state.isSubmitting);
const canSubmit = useStore(form.store, (state) => state.canSubmit);
```

### form.Subscribe component

Convenience wrapper around `useStore` for JSX:

```tsx
<form.Subscribe
  selector={(state) => [state.canSubmit, state.isSubmitting]}
  children={([canSubmit, isSubmitting]) => (
    <button type="submit" disabled={!canSubmit}>
      {isSubmitting ? '...' : 'Submit'}
    </button>
  )}
/>
```

### Performance: always use a selector

TanStack Form uses static class instances with reactive properties powered by TanStack Store. Context values are not directly reactive, preventing re-render cascades from context propagation.

```tsx
// CORRECT — only re-renders when firstName changes
const firstName = useStore(form.store, (state) => state.values.firstName);

// CORRECT — subscribe to error map
const errors = useStore(form.store, (state) => state.errorMap);

// WRONG — re-renders on every form state change
const store = useStore(form.store);
```

### Available form state fields

Key fields available on the form store state:

| Field | Type | Description |
|-------|------|-------------|
| `values` | `TFormData` | Current form values object |
| `errorMap` | `Record<string, string[]>` | Form-level errors by validation source |
| `canSubmit` | `boolean` | Whether form can be submitted |
| `isSubmitting` | `boolean` | Whether form is currently submitting |
| `isValidating` | `boolean` | Whether form is currently validating |
| `isTouched` | `boolean` | Whether any field has been touched |
| `isDirty` | `boolean` | Whether any field value differs from default |
| `isValid` | `boolean` | Whether form has no errors |
| `submissionAttempts` | `number` | Number of submission attempts |
