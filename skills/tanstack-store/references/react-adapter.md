# TanStack Store React Adapter

## Table of Contents

- [Installation](#installation)
- [useStore hook](#usestore-hook)
- [shallow function](#shallow-function)
- [Full React example](#full-react-example)

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
  options?: UseStoreOptions<TSelected>
): TSelected;

// With a Derived
function useStore<TState, TSelected = TState>(
  store: Derived<TState>,
  selector?: (state: TState) => TSelected,
  options?: UseStoreOptions<TSelected>
): TSelected;
```

- Accepts both `Store` and `Derived` instances.
- `selector` extracts a slice of state — component only re-renders when the selected value changes.
- Without a selector, returns the full state and re-renders on any change.

### Usage

```tsx
import { Store, useStore } from '@tanstack/react-store';

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

Shallow equality comparison. Use with `useStore` options to prevent re-renders when selecting objects/arrays that are structurally equal.

## Full React example

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Store, useStore } from '@tanstack/react-store';

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
