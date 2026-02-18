# Templates

## Control Flow

### @if

```html
@if (user()) {
  <h1>Welcome, {{ user()!.name }}</h1>
} @else if (isLoading()) {
  <app-spinner />
} @else {
  <a routerLink="/login">Sign in</a>
}
```

### @for

```html
@for (item of items(); track item.id) {
  <div class="item">{{ item.name }}</div>
} @empty {
  <p>No items found.</p>
}
```

**Always provide `track`** — use a unique identifier, not `$index` (unless order is stable).

### @switch

```html
@switch (status()) {
  @case ('active') {
    <span class="badge green">Active</span>
  }
  @case ('pending') {
    <span class="badge yellow">Pending</span>
  }
  @default {
    <span class="badge gray">Unknown</span>
  }
}
```

**Do NOT use** `*ngIf`, `*ngFor`, `*ngSwitch` — use native control flow instead.

## @defer Blocks

Lazy load parts of a template with code splitting:

```html
<!-- Load when browser is idle (default) -->
@defer (on idle) {
  <app-heavy-chart [data]="chartData()" />
} @placeholder {
  <div class="placeholder">Chart loading...</div>
} @loading (minimum 300ms) {
  <app-spinner />
} @error {
  <p>Failed to load chart.</p>
}
```

### Defer Triggers

| Trigger | Description |
|---------|-------------|
| `on idle` | Browser is idle (default) |
| `on viewport` | Element enters the viewport |
| `on interaction` | User clicks/focuses/touches |
| `on hover` | Mouse enters the element |
| `on immediate` | Immediately after initial render |
| `on timer(500ms)` | After specified duration |
| `when condition` | When a boolean expression becomes true |

### Prefetching

```html
@defer (on interaction; prefetch on idle) {
  <app-details />
}
```

## Pipes

### Built-in Pipes

```html
<!-- Date -->
{{ birthday() | date:'longDate' }}
{{ createdAt() | date:'yyyy-MM-dd HH:mm' }}

<!-- Currency -->
{{ price() | currency:'EUR':'symbol':'1.2-2' }}

<!-- Number -->
{{ ratio() | percent:'1.0-2' }}
{{ value() | number:'1.0-3' }}

<!-- Async (for Observables) -->
{{ data$ | async }}

<!-- JSON (debugging) -->
<pre>{{ config() | json }}</pre>

<!-- Text transforms -->
{{ name() | uppercase }}
{{ name() | lowercase }}
{{ name() | titlecase }}

<!-- Array -->
{{ items() | slice:0:5 }}
{{ items() | keyvalue }}
```

### Custom Pipe

```typescript
import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'truncate' })
export class TruncatePipe implements PipeTransform {
  transform(value: string, limit = 50, trail = '...'): string {
    return value.length > limit ? value.substring(0, limit) + trail : value;
  }
}

// Usage: {{ description() | truncate:100:'…' }}
```

## Directives

### Attribute Directive

```typescript
import { Directive, input, effect, ElementRef, inject } from '@angular/core';

@Directive({ selector: '[appHighlight]' })
export class HighlightDirective {
  appHighlight = input('yellow');

  private el = inject(ElementRef);

  constructor() {
    effect(() => {
      this.el.nativeElement.style.backgroundColor = this.appHighlight();
    });
  }
}

// Usage: <p appHighlight="lightblue">Highlighted text</p>
```

### Structural Directive (Rare — prefer @if/@for)

Only create when native control flow doesn't cover the use case.

## Template Reference Variables

```html
<input #nameInput />
<button (click)="greet(nameInput.value)">Greet</button>

<!-- Reference a component -->
<app-timer #timer />
<button (click)="timer.start()">Start</button>
```

## Two-Way Binding

```html
<!-- With model() -->
<app-toggle [(checked)]="isEnabled" />

<!-- With ngModel (requires FormsModule) -->
<input [(ngModel)]="searchQuery" />

<!-- Manual two-way -->
<input [value]="name()" (input)="name.set($any($event.target).value)" />
```

## Class and Style Bindings

Use class/style bindings instead of `ngClass`/`ngStyle`:

```html
<!-- Single class -->
<div [class.active]="isActive()">

<!-- Multiple classes -->
<div [class]="{ active: isActive(), disabled: isDisabled() }">

<!-- Single style -->
<div [style.width.px]="width()">

<!-- Multiple styles -->
<div [style]="{ width: width() + 'px', color: textColor() }">
```
