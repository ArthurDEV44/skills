# Routing

## Route Configuration

```typescript
// app.routes.ts
import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: '/home', pathMatch: 'full' },
  {
    path: 'home',
    loadComponent: () => import('./home.component').then(m => m.HomeComponent),
  },
  {
    path: 'users/:id',
    loadComponent: () => import('./user-detail.component').then(m => m.UserDetailComponent),
  },
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.routes').then(m => m.adminRoutes),
    canActivate: [authGuard],
  },
  { path: '**', loadComponent: () => import('./not-found.component').then(m => m.NotFoundComponent) },
];
```

## App Configuration

```typescript
// app.config.ts
import { ApplicationConfig } from '@angular/core';
import { provideRouter, withPreloading, PreloadAllModules, withComponentInputBinding } from '@angular/router';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(
      routes,
      withPreloading(PreloadAllModules),
      withComponentInputBinding(), // Bind route params to component inputs
    ),
  ],
};
```

## Lazy Loading

Always lazy load feature routes with `loadComponent` or `loadChildren`:

```typescript
export const routes: Routes = [
  // Lazy load a single component
  {
    path: 'dashboard',
    loadComponent: () => import('./dashboard.component').then(m => m.DashboardComponent),
  },
  // Lazy load a set of child routes
  {
    path: 'settings',
    loadChildren: () => import('./settings/settings.routes').then(m => m.settingsRoutes),
  },
];
```

## Functional Guards

Use functional guards instead of class-based:

```typescript
// auth.guard.ts
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './auth.service';

export const authGuard: CanActivateFn = (route, state) => {
  const auth = inject(AuthService);
  const router = inject(Router);

  if (auth.isLoggedIn()) {
    return true;
  }
  return router.createUrlTree(['/login'], {
    queryParams: { returnUrl: state.url },
  });
};

// Role-based guard
export const roleGuard: CanActivateFn = (route) => {
  const auth = inject(AuthService);
  const requiredRole = route.data['role'] as string;
  return auth.hasRole(requiredRole);
};

// Usage
{
  path: 'admin',
  canActivate: [authGuard, roleGuard],
  data: { role: 'admin' },
  loadComponent: () => import('./admin.component'),
}
```

## Resolvers

Pre-fetch data before route activation:

```typescript
// user.resolver.ts
import { inject } from '@angular/core';
import { ResolveFn } from '@angular/router';
import { UserService } from './user.service';

export const userResolver: ResolveFn<User> = (route) => {
  const userService = inject(UserService);
  return userService.getUser(route.paramMap.get('id')!);
};

// Route config
{
  path: 'users/:id',
  resolve: { user: userResolver },
  loadComponent: () => import('./user-detail.component'),
}

// Component — access resolved data via input
@Component({ ... })
export class UserDetailComponent {
  user = input.required<User>(); // Bound via withComponentInputBinding()
}
```

## Accessing Route Data

With `withComponentInputBinding()`, route params bind directly to inputs:

```typescript
@Component({ ... })
export class UserDetailComponent {
  // Bound from :id param
  id = input.required<string>();

  // Bound from query params
  tab = input<string>();

  // Bound from resolver
  user = input.required<User>();
}
```

Without input binding, use `ActivatedRoute`:

```typescript
@Component({ ... })
export class UserDetailComponent {
  private route = inject(ActivatedRoute);

  id = toSignal(this.route.paramMap.pipe(
    map(params => params.get('id')!)
  ));
}
```

## Router Events

```typescript
import { Router, NavigationEnd } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter, map } from 'rxjs/operators';

@Component({ ... })
export class AppComponent {
  private router = inject(Router);

  loading = toSignal(
    this.router.events.pipe(
      map(() => !!this.router.getCurrentNavigation())
    ),
    { initialValue: false }
  );
}
```
