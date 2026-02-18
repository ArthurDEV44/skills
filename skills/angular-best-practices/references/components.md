# Components

## Standalone Component (Default in v20+)

```typescript
import { ChangeDetectionStrategy, Component, input, output, computed } from '@angular/core';

@Component({
  selector: 'app-user-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="card">
      <h2>{{ fullName() }}</h2>
      <p>{{ user().email }}</p>
      <button (click)="edit.emit(user())">Edit</button>
    </div>
  `,
  styles: `.card { padding: 1rem; border: 1px solid #ccc; border-radius: 8px; }`,
})
export class UserCardComponent {
  // Signal inputs
  user = input.required<{ name: string; email: string }>();
  showActions = input(true); // default value

  // Outputs
  edit = output<{ name: string; email: string }>();

  // Derived state
  fullName = computed(() => this.user().name.toUpperCase());
}
```

**Key rules:**
- Do NOT set `standalone: true` in the decorator (it's the default in v20+)
- Always set `changeDetection: ChangeDetectionStrategy.OnPush`
- Prefer inline templates for small components
- Use relative paths for external templates/styles

## Signal Inputs

```typescript
import { Component, input } from '@angular/core';

@Component({ ... })
export class MyComponent {
  // Optional input (Signal<string | undefined>)
  label = input<string>();

  // Required input (Signal<string>)
  name = input.required<string>();

  // Input with default value (Signal<number>)
  count = input(0);

  // Input with alias
  size = input(16, { alias: 'fontSize' });

  // Input with transform
  disabled = input(false, {
    transform: (value: boolean | string) => typeof value === 'string' ? value !== 'false' : value,
  });
}
```

## Outputs

```typescript
import { Component, output } from '@angular/core';

@Component({ ... })
export class MyComponent {
  // Simple event
  closed = output<void>();

  // Event with payload
  saved = output<{ id: string; name: string }>();

  // Emit in method
  onSave() {
    this.saved.emit({ id: '1', name: 'Item' });
  }
}
```

## Model Inputs (Two-Way Binding)

```typescript
import { Component, model } from '@angular/core';

@Component({
  selector: 'app-toggle',
  template: `<button (click)="checked.set(!checked())">{{ checked() ? 'ON' : 'OFF' }}</button>`,
})
export class ToggleComponent {
  // Two-way bindable signal
  checked = model(false);
  // Required model
  value = model.required<string>();
}

// Usage: <app-toggle [(checked)]="isEnabled" />
```

## View Queries

```typescript
import { Component, viewChild, viewChildren, contentChild, contentChildren, ElementRef } from '@angular/core';

@Component({ ... })
export class MyComponent {
  // Single child query (Signal<ElementRef | undefined>)
  canvas = viewChild<ElementRef>('myCanvas');

  // Required child query (Signal<ElementRef>)
  header = viewChild.required<ElementRef>('header');

  // Multiple children (Signal<ReadonlyArray<ChildComponent>>)
  items = viewChildren(ChildComponent);

  // Content projection queries
  projectedHeader = contentChild<ElementRef>('header');
  projectedItems = contentChildren(ItemDirective);
}
```

## Lifecycle Hooks

```typescript
import { Component, OnInit, OnDestroy, AfterViewInit, input, effect } from '@angular/core';

@Component({ ... })
export class MyComponent implements OnInit, AfterViewInit, OnDestroy {
  data = input.required<string>();

  constructor() {
    // Prefer effect() for reacting to signal changes
    effect(() => {
      console.log('Data changed:', this.data());
    });
  }

  ngOnInit() {
    // Component initialized, inputs available
  }

  ngAfterViewInit() {
    // View children available
  }

  ngOnDestroy() {
    // Cleanup subscriptions, timers
  }
}
```

**Prefer `effect()` over `ngOnChanges`** for reacting to input changes.

## Content Projection

```typescript
// Card component
@Component({
  selector: 'app-card',
  template: `
    <div class="card">
      <div class="header">
        <ng-content select="[card-header]" />
      </div>
      <div class="body">
        <ng-content />
      </div>
      <div class="footer">
        <ng-content select="[card-footer]" />
      </div>
    </div>
  `,
})
export class CardComponent {}

// Usage:
// <app-card>
//   <h2 card-header>Title</h2>
//   <p>Body content</p>
//   <button card-footer>Action</button>
// </app-card>
```

## Host Bindings

Use the `host` object instead of `@HostBinding` / `@HostListener`:

```typescript
@Component({
  selector: 'app-button',
  host: {
    'role': 'button',
    '[class.active]': 'isActive()',
    '[attr.aria-disabled]': 'disabled()',
    '(click)': 'onClick($event)',
    '(keydown.enter)': 'onClick($event)',
  },
  template: `<ng-content />`,
})
export class ButtonComponent {
  isActive = input(false);
  disabled = input(false);

  onClick(event: Event) {
    if (!this.disabled()) {
      // handle click
    }
  }
}
```
