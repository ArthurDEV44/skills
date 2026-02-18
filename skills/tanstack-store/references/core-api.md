# TanStack Store Core API Reference

## Table of Contents

- [Store class](#store-class)
- [Derived class](#derived-class)
- [Effect class](#effect-class)
- [batch function](#batch-function)
- [StoreOptions interface](#storeoptions-interface)
- [DerivedOptions interface](#derivedoptions-interface)
- [DerivedFnProps interface](#derivedfnprops-interface)
- [Type aliases](#type-aliases)

## Store class

```typescript
import { Store } from '@tanstack/store';

// Create with initial state
const store = new Store<TState>(initialState, options?: StoreOptions<TState>);
```

### Properties

- `state: TState` — Current state value.

### Methods

- `setState(updater: (prev: TState) => TState): void` — Update state via updater function.
- `subscribe(callback: () => void): () => void` — Listen for state changes. Returns unsubscribe function.

### StoreOptions interface

```typescript
interface StoreOptions<TState> {
  updateFn?: (prevValue: TState) => (updateFn: (prev: TState) => TState) => TState;
  onUpdate?: () => void;
  onSubscribe?: (listener: () => void, store: Store<TState>) => (() => void) | void;
}
```

- `updateFn` — Transform updates before applying (middleware pattern).
- `onUpdate` — Callback fired after every state update (primitive derived state).
- `onSubscribe` — Lifecycle hook called when a new subscriber is added. Receives the listener and the store. Return a cleanup function that runs when the subscriber unsubscribes.

## Derived class

```typescript
import { Derived } from '@tanstack/store';

const derived = new Derived<TState>(options: DerivedOptions<TState>);
```

Lazily computed values that update when dependencies change. Must be mounted to start listening.

### Properties

- `state: TState` — Current derived value.

### Methods

- `mount(): () => void` — Start listening for dependency changes. Returns unmount function.

### DerivedOptions interface

```typescript
interface DerivedOptions<TState> {
  fn: (props: DerivedFnProps<TState>) => TState;
  deps: Array<Store<any> | Derived<any>>;
}
```

### DerivedFnProps interface

```typescript
interface DerivedFnProps<TState> {
  prevVal: TState | undefined;
  prevDepVals: Array<any> | undefined;
  currDepVals: Array<any>;
}
```

- `prevVal` — Previous derived value (`undefined` on first computation).
- `prevDepVals` — Previous dependency values (`undefined` on first computation).
- `currDepVals` — Current dependency values (same order as `deps` array).

## Effect class

```typescript
import { Effect } from '@tanstack/store';

const effect = new Effect(options: {
  fn: () => void;
  deps: Array<Store<any> | Derived<any>>;
  eager?: boolean; // default: false
});
```

### Methods

- `mount(): () => void` — Start listening. Returns unmount function.

### Options

- `fn` — Side effect function to run when deps change.
- `deps` — Array of `Store` or `Derived` instances to watch.
- `eager` — If `true`, run effect immediately on mount.

## batch function

```typescript
import { batch } from '@tanstack/store';

batch(() => {
  store1.setState(() => newVal1);
  store2.setState(() => newVal2);
});
```

Batch multiple `setState` calls. Subscribers fire only once at the end with final state.

## Type aliases

- `Updater<TState>` — `TState | ((prev: TState) => TState)`
- `UnwrapDerivedOrStore<T>` — Extracts state type from `Store<T>` or `Derived<T>`.
