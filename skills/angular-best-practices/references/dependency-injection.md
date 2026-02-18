# Dependency Injection

## inject() Function (Preferred)

Use `inject()` instead of constructor injection:

```typescript
import { Component, inject } from '@angular/core';
import { UserService } from './user.service';
import { ActivatedRoute, Router } from '@angular/router';

@Component({ ... })
export class UserComponent {
  private userService = inject(UserService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
}
```

**Rules:**
- Call `inject()` in field initializers or constructor (injection context)
- Cannot call `inject()` inside lifecycle hooks or event handlers
- Use `{ optional: true }` when dependency may not exist

## Services

```typescript
import { Injectable, signal, computed } from '@angular/core';

@Injectable({
  providedIn: 'root', // Singleton available app-wide
})
export class AuthService {
  private currentUser = signal<User | null>(null);

  readonly isLoggedIn = computed(() => this.currentUser() !== null);
  readonly user = this.currentUser.asReadonly();

  login(credentials: { email: string; password: string }) {
    // ...
  }

  logout() {
    this.currentUser.set(null);
  }
}
```

Use `providedIn: 'root'` for all singleton services. Only provide at component level when you need a separate instance per component.

## InjectionToken

For non-class dependencies:

```typescript
import { InjectionToken, inject } from '@angular/core';

// Define token
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL');
export const APP_CONFIG = new InjectionToken<AppConfig>('APP_CONFIG');

// Provide in app config
export const appConfig: ApplicationConfig = {
  providers: [
    { provide: API_BASE_URL, useValue: 'https://api.example.com' },
    { provide: APP_CONFIG, useFactory: () => loadConfig() },
  ],
};

// Inject
@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = inject(API_BASE_URL);
}
```

## Provider Types

```typescript
providers: [
  // useClass — provide a class implementation
  { provide: LoggerService, useClass: ConsoleLoggerService },

  // useValue — provide a static value
  { provide: API_URL, useValue: 'https://api.example.com' },

  // useFactory — provide via a factory function
  {
    provide: DataService,
    useFactory: () => {
      const http = inject(HttpClient);
      const config = inject(APP_CONFIG);
      return new DataService(http, config.apiUrl);
    },
  },

  // useExisting — alias one token to another
  { provide: AbstractLogger, useExisting: ConsoleLoggerService },
]
```

## Hierarchical Injection

```typescript
// Component-level provider — new instance per component
@Component({
  providers: [FormStateService],
})
export class FormComponent {
  private formState = inject(FormStateService);
}

// Route-level providers
export const routes: Routes = [
  {
    path: 'admin',
    providers: [AdminService],
    loadComponent: () => import('./admin.component'),
  },
];
```

## Optional and Self/SkipSelf

```typescript
@Component({ ... })
export class MyComponent {
  // Optional — returns null if not found
  private logger = inject(LoggerService, { optional: true });

  // Self — only look in this component's injector
  private local = inject(LocalService, { self: true });

  // SkipSelf — skip this component, look in parent
  private parent = inject(ParentService, { skipSelf: true });
}
```
