# SSR and Performance

## Server-Side Rendering Setup

```typescript
// app.config.ts
import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideClientHydration, withIncrementalHydration } from '@angular/platform-browser';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideClientHydration(withIncrementalHydration()),
    provideHttpClient(withFetch()), // Required for SSR
  ],
};
```

### Server Configuration

```typescript
// app.config.server.ts
import { mergeApplicationConfig } from '@angular/core';
import { provideServerRendering } from '@angular/ssr';
import { appConfig } from './app.config';

export const serverConfig = mergeApplicationConfig(appConfig, {
  providers: [provideServerRendering()],
});
```

### Server Routes

```typescript
// app.routes.server.ts
import { RenderMode, ServerRoute } from '@angular/ssr';

export const serverRoutes: ServerRoute[] = [
  { path: '', renderMode: RenderMode.Prerender },          // Static at build time
  { path: 'dashboard', renderMode: RenderMode.Server },     // SSR on each request
  { path: 'login', renderMode: RenderMode.Client },         // Client-only
  { path: '**', renderMode: RenderMode.Server },
];
```

## Incremental Hydration

Combine `@defer` with hydration triggers for fine-grained control:

```html
<!-- Hydrate on interaction (e.g., sidebar) -->
@defer (on idle; hydrate on interaction) {
  <app-sidebar />
} @placeholder {
  <div class="sidebar-placeholder">Menu</div>
}

<!-- Hydrate when visible -->
@defer (on viewport; hydrate on viewport) {
  <app-comments [postId]="postId()" />
} @placeholder {
  <div class="comments-skeleton">Loading comments...</div>
}

<!-- Hydrate immediately (critical content) -->
@defer (hydrate on immediate) {
  <app-hero-banner />
}
```

Available hydrate triggers: `hydrate on idle`, `hydrate on viewport`, `hydrate on interaction`, `hydrate on hover`, `hydrate on immediate`, `hydrate on timer(ms)`.

## Zoneless Change Detection

```typescript
// app.config.ts
import { provideZonelessChangeDetection } from '@angular/core';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    // ... other providers
  ],
};
```

**Requirements for zoneless:**
- All components must use `OnPush` change detection
- Use signals for all reactive state
- Avoid manual `ChangeDetectorRef.detectChanges()` calls
- Test with `provideZonelessChangeDetection()` in TestBed

## NgOptimizedImage

Use for all static images:

```typescript
import { NgOptimizedImage } from '@angular/common';

@Component({
  imports: [NgOptimizedImage],
  template: `
    <!-- LCP image — add priority -->
    <img ngSrc="/hero.jpg" width="1200" height="600" priority />

    <!-- Regular image with fill mode -->
    <div class="image-container" style="position: relative; width: 100%; height: 300px;">
      <img ngSrc="/photo.jpg" fill [alt]="altText()" />
    </div>

    <!-- Responsive with sizes -->
    <img ngSrc="/banner.jpg" width="800" height="400"
         sizes="(max-width: 768px) 100vw, 50vw" />
  `,
})
export class MyComponent {}
```

**Rules:**
- Always provide `width` and `height` (or use `fill`)
- Add `priority` to LCP (above-the-fold) images
- Does NOT work with inline base64 images
- Use a CDN image loader for production

## Performance Patterns

### OnPush + Signals

```typescript
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ListComponent {
  items = input.required<Item[]>();
  filter = signal('');

  filteredItems = computed(() =>
    this.items().filter(item =>
      item.name.toLowerCase().includes(this.filter().toLowerCase())
    )
  );
}
```

### Lazy Load Routes

```typescript
{
  path: 'reports',
  loadComponent: () => import('./reports.component').then(m => m.ReportsComponent),
}
```

### @defer for Heavy Components

```html
@defer (on viewport) {
  <app-data-visualization [data]="data()" />
} @placeholder {
  <div style="height: 400px;">Chart area</div>
}
```

### Preloading Strategy

```typescript
provideRouter(
  routes,
  withPreloading(PreloadAllModules), // Preload all lazy routes after initial load
)
```

### trackBy in @for

```html
<!-- Always use a stable unique key -->
@for (user of users(); track user.id) {
  <app-user-card [user]="user" />
}
```

### Avoid Expensive Template Expressions

```typescript
// BAD: recalculates every change detection cycle
// template: {{ getFilteredItems().length }}

// GOOD: computed signal, memoized
filteredCount = computed(() => this.filteredItems().length);
// template: {{ filteredCount() }}
```

## Testing with Zoneless

```typescript
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

describe('MyComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('should update on signal change', async () => {
    const fixture = TestBed.createComponent(MyComponent);
    fixture.componentInstance.name.set('Test');
    await fixture.whenStable();
    expect(fixture.nativeElement.textContent).toContain('Test');
  });
});
```
