---
name: angular-best-practices
description: Angular v20+ best practices - standalone components, signals, dependency injection, routing, reactive forms, HttpClient, control flow, defer blocks, SSR, hydration, zoneless, OnPush. Use when writing, reviewing, or refactoring Angular code, creating components, services, guards, pipes, directives, or configuring Angular applications.
---

# Angular Best Practices

Apply these rules when writing or reviewing Angular v20+ code.

## Core Rules

- Always use standalone components (default in v20+, do NOT set `standalone: true` in decorators)
- Use `input()` and `output()` functions instead of `@Input()` / `@Output()` decorators
- Use `inject()` function instead of constructor injection
- Use signals for state management, `computed()` for derived state
- Set `changeDetection: ChangeDetectionStrategy.OnPush` on all components
- Use native control flow (`@if`, `@for`, `@switch`) instead of `*ngIf`, `*ngFor`, `*ngSwitch`
- Use `host` object in decorators instead of `@HostBinding` / `@HostListener`
- Use `NgOptimizedImage` for all static images
- Prefer reactive forms over template-driven forms

## Components

See [references/components.md](./references/components.md) for:
- Standalone component structure
- Signal inputs (`input()`, `input.required()`)
- Outputs (`output()`)
- Model inputs (`model()`)
- View queries (`viewChild`, `viewChildren`, `contentChild`, `contentChildren`)
- Lifecycle hooks
- OnPush change detection
- Content projection (`ng-content`)

## Signals and State

See [references/signals-state.md](./references/signals-state.md) for:
- `signal()`, `computed()`, `effect()`
- `linkedSignal()` for dependent writable state
- `resource()` and `rxResource()` for async data
- `toSignal()` / `toObservable()` RxJS interop
- State management patterns

## Dependency Injection

See [references/dependency-injection.md](./references/dependency-injection.md) for:
- `inject()` function
- `providedIn: 'root'` services
- `InjectionToken` for non-class values
- Provider configuration (`useClass`, `useValue`, `useFactory`, `useExisting`)
- Hierarchical injection

## Routing

See [references/routing.md](./references/routing.md) for:
- Route configuration with `provideRouter`
- Lazy loading with `loadComponent` / `loadChildren`
- Functional guards (`canActivate`, `canMatch`, `canDeactivate`)
- Resolvers
- Preloading strategies
- Router events and navigation

## Forms

See [references/forms.md](./references/forms.md) for:
- Reactive forms (`FormControl`, `FormGroup`, `FormBuilder`)
- `FormArray` for dynamic fields
- Validators (built-in, custom sync, async)
- Form state and validation display
- Typed forms

## HTTP

See [references/http.md](./references/http.md) for:
- `provideHttpClient` configuration
- `withFetch()` for SSR
- Functional interceptors with `withInterceptors`
- Typed requests and responses
- Error handling

## Templates

See [references/templates.md](./references/templates.md) for:
- Control flow (`@if`, `@for`, `@switch`)
- `@defer` blocks with triggers
- Pipes (built-in and custom)
- Directives (attribute and structural)
- Template reference variables
- Two-way binding

## SSR and Performance

See [references/ssr-performance.md](./references/ssr-performance.md) for:
- Server-side rendering setup
- `provideClientHydration` and incremental hydration
- Zoneless change detection (`provideZonelessChangeDetection`)
- `NgOptimizedImage`
- `@defer` for code splitting
- Preloading strategies
- OnPush and signal-based performance
