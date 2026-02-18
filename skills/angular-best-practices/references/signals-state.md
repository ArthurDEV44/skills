# Signals and State Management

## signal()

Writable signal for local component state:

```typescript
import { signal } from '@angular/core';

const count = signal(0);

// Read
console.log(count()); // 0

// Set
count.set(5);

// Update based on previous value
count.update(prev => prev + 1);
```

**Never use `mutate`** — use `update` or `set` instead.

## computed()

Derived read-only signal:

```typescript
import { signal, computed } from '@angular/core';

const firstName = signal('John');
const lastName = signal('Doe');

// Automatically recalculates when dependencies change
const fullName = computed(() => `${firstName()} ${lastName()}`);
```

**Rules:**
- Computations must be pure (no side effects)
- Dependencies are tracked automatically
- Memoized — only recalculates when dependencies change

## effect()

Side effects that run when signals change:

```typescript
import { signal, effect, DestroyRef, inject } from '@angular/core';

@Component({ ... })
export class MyComponent {
  query = signal('');

  constructor() {
    // Runs whenever query() changes
    effect(() => {
      console.log('Search query:', this.query());
    });

    // Manual cleanup
    const ref = effect(() => { ... });
    ref.destroy();
  }
}
```

**Rules:**
- Only create effects in injection context (constructor or field initializer)
- Keep effects focused on a single side effect
- Do not set signals inside effects (use `computed` or `linkedSignal` instead)
- For writing signals inside effect, use `allowSignalWrites` option (use sparingly)

## linkedSignal()

Writable signal linked to a reactive computation:

```typescript
import { signal, linkedSignal } from '@angular/core';

const items = signal(['a', 'b', 'c']);

// Resets to first item whenever items change
const selectedItem = linkedSignal(() => items()[0]);

// Can still be written to manually
selectedItem.set('b');

// Advanced form with previous value
const page = linkedSignal({
  source: () => items(),
  computation: (items, previous) => {
    // Reset to 0 if items changed, keep page otherwise
    if (previous && previous.source.length === items.length) {
      return previous.value;
    }
    return 0;
  },
});
```

## resource()

Async data fetching tied to signals:

```typescript
import { resource, signal, computed } from '@angular/core';

@Component({ ... })
export class UserComponent {
  userId = signal('1');

  userResource = resource({
    params: () => ({ id: this.userId() }),
    loader: async ({ params }) => {
      const response = await fetch(`/api/users/${params.id}`);
      return response.json();
    },
  });

  // Access resource state
  user = computed(() => this.userResource.hasValue() ? this.userResource.value() : null);
  isLoading = computed(() => this.userResource.isLoading());
  error = computed(() => this.userResource.error());
}
```

## rxResource()

Resource backed by RxJS Observable:

```typescript
import { rxResource } from '@angular/core/rxjs-interop';
import { HttpClient } from '@angular/common/http';

@Component({ ... })
export class UserComponent {
  private http = inject(HttpClient);
  userId = signal('1');

  userResource = rxResource({
    params: () => ({ id: this.userId() }),
    loader: ({ params }) => this.http.get<User>(`/api/users/${params.id}`),
  });
}
```

## RxJS Interop

### toSignal()

Convert Observable to Signal:

```typescript
import { toSignal } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';

@Component({ ... })
export class MyComponent {
  // Signal<number | undefined>
  count = toSignal(interval(1000));

  // With initial value: Signal<number>
  count2 = toSignal(interval(1000), { initialValue: 0 });

  // With requireSync for synchronous observables (e.g. BehaviorSubject)
  value = toSignal(myBehaviorSubject$, { requireSync: true });
}
```

### toObservable()

Convert Signal to Observable:

```typescript
import { toObservable } from '@angular/core/rxjs-interop';
import { signal } from '@angular/core';
import { switchMap } from 'rxjs/operators';

@Component({ ... })
export class MyComponent {
  query = signal('');

  results$ = toObservable(this.query).pipe(
    switchMap(q => this.searchService.search(q))
  );
}
```

## State Management Patterns

### Component-level state

```typescript
@Component({ ... })
export class CounterComponent {
  count = signal(0);
  doubled = computed(() => this.count() * 2);

  increment() { this.count.update(c => c + 1); }
  decrement() { this.count.update(c => c - 1); }
}
```

### Shared state via service

```typescript
@Injectable({ providedIn: 'root' })
export class CartService {
  private items = signal<CartItem[]>([]);

  readonly items$ = this.items.asReadonly();
  readonly total = computed(() =>
    this.items().reduce((sum, item) => sum + item.price * item.qty, 0)
  );
  readonly count = computed(() => this.items().length);

  addItem(item: CartItem) {
    this.items.update(items => [...items, item]);
  }

  removeItem(id: string) {
    this.items.update(items => items.filter(i => i.id !== id));
  }
}
```
