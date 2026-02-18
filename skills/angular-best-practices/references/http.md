# HTTP

## Setup

```typescript
// app.config.ts
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';

export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(
      withFetch(),              // Required for SSR, recommended always
      withInterceptors([authInterceptor, loggingInterceptor]),
    ),
  ],
};
```

**Always use `withFetch()`** — it's required for SSR and recommended for all apps.

## Basic Requests

```typescript
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);

  getUsers(): Observable<User[]> {
    return this.http.get<User[]>('/api/users');
  }

  getUser(id: string): Observable<User> {
    return this.http.get<User>(`/api/users/${id}`);
  }

  createUser(data: CreateUserDto): Observable<User> {
    return this.http.post<User>('/api/users', data);
  }

  updateUser(id: string, data: Partial<User>): Observable<User> {
    return this.http.put<User>(`/api/users/${id}`, data);
  }

  deleteUser(id: string): Observable<void> {
    return this.http.delete<void>(`/api/users/${id}`);
  }
}
```

## With Query Params

```typescript
import { HttpParams } from '@angular/common/http';

search(query: string, page: number): Observable<SearchResult> {
  const params = new HttpParams()
    .set('q', query)
    .set('page', page.toString())
    .set('limit', '20');

  return this.http.get<SearchResult>('/api/search', { params });
}
```

## With Headers

```typescript
import { HttpHeaders } from '@angular/common/http';

upload(file: File): Observable<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);

  return this.http.post<UploadResult>('/api/upload', formData, {
    headers: new HttpHeaders({ 'Accept': 'application/json' }),
    reportProgress: true,
    observe: 'events',
  });
}
```

## Functional Interceptors

```typescript
import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';

// Auth interceptor
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const token = auth.getToken();

  if (token) {
    const cloned = req.clone({
      setHeaders: { Authorization: `Bearer ${token}` },
    });
    return next(cloned);
  }
  return next(req);
};

// Logging interceptor
export const loggingInterceptor: HttpInterceptorFn = (req, next) => {
  const started = Date.now();
  return next(req).pipe(
    tap({
      next: () => console.log(`${req.method} ${req.url} - ${Date.now() - started}ms`),
      error: (err) => console.error(`${req.method} ${req.url} FAILED - ${err.status}`),
    }),
  );
};

// Error handling interceptor
export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      if (error.status === 401) {
        inject(Router).navigate(['/login']);
      }
      return throwError(() => error);
    }),
  );
};
```

## Using with resource()

```typescript
import { resource } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';

@Component({ ... })
export class UserComponent {
  private http = inject(HttpClient);
  userId = signal('1');

  // Observable-based resource
  user = rxResource({
    params: () => ({ id: this.userId() }),
    loader: ({ params }) => this.http.get<User>(`/api/users/${params.id}`),
  });

  // Or fetch-based resource
  userFetch = resource({
    params: () => ({ id: this.userId() }),
    loader: async ({ params }) => {
      const res = await fetch(`/api/users/${params.id}`);
      return res.json() as Promise<User>;
    },
  });
}
```
