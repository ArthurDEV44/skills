# TanStack Query SSR, Suspense & Server Components

## Table of Contents
- [Suspense](#suspense)
- [Server Rendering & Hydration](#server-rendering--hydration)
- [Advanced Server Rendering (App Router / Server Components)](#advanced-server-rendering)
- [Testing](#testing)

## Suspense

Dedicated hooks: `useSuspenseQuery`, `useSuspenseInfiniteQuery`, `useSuspenseQueries`.

```tsx
import { useSuspenseQuery } from '@tanstack/react-query'

const { data } = useSuspenseQuery({ queryKey, queryFn })
// data is guaranteed defined (errors/loading handled by Suspense/ErrorBoundaries)
```

- Cannot conditionally enable/disable suspense queries (no `enabled` or `skipToken`).
- Use `startTransition` to prevent fallback during key changes.
- `throwOnError` default: only throws if no cached data exists.

### Error Boundaries

```tsx
import { QueryErrorResetBoundary } from '@tanstack/react-query'
import { ErrorBoundary } from 'react-error-boundary'

<QueryErrorResetBoundary>
  {({ reset }) => (
    <ErrorBoundary
      onReset={reset}
      fallbackRender={({ resetErrorBoundary }) => (
        <div>
          Error! <button onClick={resetErrorBoundary}>Try again</button>
        </div>
      )}
    >
      <Page />
    </ErrorBoundary>
  )}
</QueryErrorResetBoundary>
```

### Parallel Suspense Queries

Individual `useSuspenseQuery` calls in the same component run serially (each suspends in turn). Use `useSuspenseQueries` for parallel fetching:

```tsx
const [users, projects] = useSuspenseQueries({
  queries: [
    { queryKey: ['users'], queryFn: fetchUsers },
    { queryKey: ['projects'], queryFn: fetchProjects },
  ],
})
// Both fetch in parallel, component suspends until both resolve
```

### Prefetching with Suspense (flatten waterfalls)

Use `usePrefetchQuery` in a parent component to start fetching before the child suspends:

```tsx
function ParentComponent() {
  usePrefetchQuery(detailsOptions(id))

  return (
    <Suspense fallback={<Loading />}>
      <ChildComponent id={id} />
    </Suspense>
  )
}

function ChildComponent({ id }: { id: string }) {
  const { data } = useSuspenseQuery(detailsOptions(id))
  // ...
}
```

### Experimental: useQuery().promise with React.use()

Enable with `experimental_prefetchInRender: true` on QueryClient:

```tsx
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { experimental_prefetchInRender: true },
  },
})

function TodoList({ query }: { query: UseQueryResult<Todo[]> }) {
  const data = React.use(query.promise)
  return <ul>{data.map(todo => <li key={todo.id}>{todo.title}</li>)}</ul>
}

export function App() {
  const query = useQuery({ queryKey: ['todos'], queryFn: fetchTodos })
  return (
    <React.Suspense fallback={<div>Loading...</div>}>
      <TodoList query={query} />
    </React.Suspense>
  )
}
```

## Server Rendering & Hydration

### Setup (Pages Router / Remix)

Create `queryClient` inside the app in React state:
```tsx
// _app.tsx (Next.js) or app/root.tsx (Remix)
export default function MyApp({ Component, pageProps }) {
  const [queryClient] = React.useState(
    () => new QueryClient({
      defaultOptions: {
        queries: { staleTime: 60 * 1000 }, // avoid immediate refetch on client
      },
    }),
  )
  return (
    <QueryClientProvider client={queryClient}>
      <Component {...pageProps} />
    </QueryClientProvider>
  )
}
```

**Never** create queryClient at file root level (leaks data between users).

### Hydration Pattern (Recommended)

In loader/getStaticProps/getServerSideProps:
```tsx
export async function getStaticProps() {
  const queryClient = new QueryClient()
  await queryClient.prefetchQuery({
    queryKey: ['posts'],
    queryFn: getPosts,
  })
  return { props: { dehydratedState: dehydrate(queryClient) } }
}
```

In route component:
```tsx
export default function PostsRoute({ dehydratedState }) {
  return (
    <HydrationBoundary state={dehydratedState}>
      <Posts />
    </HydrationBoundary>
  )
}
```

Or remove boilerplate by placing `HydrationBoundary` in `_app.tsx`:
```tsx
<QueryClientProvider client={queryClient}>
  <HydrationBoundary state={pageProps.dehydratedState}>
    <Component {...pageProps} />
  </HydrationBoundary>
</QueryClientProvider>
```

### Quick Alternative: initialData

```tsx
export async function getServerSideProps() {
  const posts = await getPosts()
  return { props: { posts } }
}

function Posts(props) {
  const { data } = useQuery({
    queryKey: ['posts'],
    queryFn: getPosts,
    initialData: props.posts,
  })
}
```

Drawbacks: must pass down to every component, `dataUpdatedAt` is based on page load time, doesn't overwrite stale cache data.

### Prefetching Dependent Queries on Server

```tsx
export async function getServerSideProps() {
  const queryClient = new QueryClient()
  const user = await queryClient.fetchQuery({
    queryKey: ['user', email],
    queryFn: getUserByEmail,
  })
  if (user?.userId) {
    await queryClient.prefetchQuery({
      queryKey: ['projects', userId],
      queryFn: getProjectsByUser,
    })
  }
  return { props: { dehydratedState: dehydrate(queryClient) } }
}
```

### Error Handling

- `prefetchQuery` never throws (graceful degradation).
- `dehydrate` only includes successful queries by default.
- Use `fetchQuery` to throw on errors for critical content.
- Override with `shouldDehydrateQuery: (query) => true` to include failed queries.

## Advanced Server Rendering

### App Router Setup (Next.js)

#### Step 1: Create getQueryClient singleton

```tsx
// app/get-query-client.ts
import { QueryClient, defaultShouldDehydrateQuery } from '@tanstack/react-query'

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // avoid refetch on client after hydration
      },
      dehydrate: {
        // Include pending queries for streaming
        shouldDehydrateQuery: (query) =>
          defaultShouldDehydrateQuery(query) || query.state.status === 'pending',
        shouldRedactErrors: () => false,
      },
    },
  })
}

let browserQueryClient: QueryClient | undefined

export function getQueryClient() {
  if (typeof window === 'undefined') {
    // Server: always make a new query client
    return makeQueryClient()
  }
  // Browser: singleton (important for Suspense)
  if (!browserQueryClient) browserQueryClient = makeQueryClient()
  return browserQueryClient
}
```

#### Step 2: Providers (Client Component)

```tsx
// app/providers.tsx
'use client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { getQueryClient } from './get-query-client'

export default function Providers({ children }: { children: React.ReactNode }) {
  // NOTE: Use getQueryClient() instead of useState when you have Suspense
  // boundaries above this. React throws away state on initial render suspension.
  const queryClient = getQueryClient()
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}
```

#### Step 3: Layout

```tsx
// app/layout.tsx
import Providers from './providers'

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body><Providers>{children}</Providers></body>
    </html>
  )
}
```

### Prefetching in Server Components

```tsx
// app/posts/page.tsx (Server Component)
import { dehydrate, HydrationBoundary } from '@tanstack/react-query'
import { getQueryClient } from '../get-query-client'
import Posts from './posts'

export default async function PostsPage() {
  const queryClient = getQueryClient()
  await queryClient.prefetchQuery({
    queryKey: ['posts'],
    queryFn: getPosts,
  })
  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Posts />
    </HydrationBoundary>
  )
}
```

```tsx
// app/posts/posts.tsx (Client Component)
'use client'
export default function Posts() {
  const { data } = useSuspenseQuery({ queryKey: ['posts'], queryFn: getPosts })
  // ...
}
```

### Streaming with Server Components

Prefetch without `await` to enable streaming. The Promise is serialized and sent to the client:

```tsx
// app/posts/page.tsx (Server Component)
import { dehydrate, HydrationBoundary } from '@tanstack/react-query'
import { getQueryClient } from '../get-query-client'
import Posts from './posts'

export default function PostsPage() {
  const queryClient = getQueryClient()

  // No await! Promise streams to client
  queryClient.prefetchQuery({
    queryKey: ['posts'],
    queryFn: getPosts,
  })

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Posts />
    </HydrationBoundary>
  )
}
```

**How it works**: React can serialize Promises over the wire. When you prefetch without awaiting, the Promise is passed to the client through `HydrationBoundary`. On the client, the Promise is automatically put into the QueryCache. `useSuspenseQuery` picks up that Promise and suspends until it resolves.

**Requirements for streaming**:
- `shouldDehydrateQuery` must include pending queries (see getQueryClient setup above)
- Client component must use `useSuspenseQuery` to consume streamed data

### Experimental: Streaming without Prefetching

`@tanstack/react-query-next-experimental`:
```tsx
// app/providers.tsx
'use client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryStreamedHydration } from '@tanstack/react-query-next-experimental'
import { getQueryClient } from './get-query-client'

export function Providers({ children }: { children: React.ReactNode }) {
  const queryClient = getQueryClient()
  return (
    <QueryClientProvider client={queryClient}>
      <ReactQueryStreamedHydration>
        {children}
      </ReactQueryStreamedHydration>
    </QueryClientProvider>
  )
}
```

Allows `useSuspenseQuery` in Client Components without explicit prefetching in Server Components. Results stream from server as Suspense boundaries resolve.

**Tradeoff**: Creates request waterfalls on page navigations (no server-side prefetch means client initiates the request).

### Data Ownership Warning

Avoid rendering prefetched data directly in Server Components AND in Client Components via useQuery. The Server Component can't revalidate, causing stale UI. Treat Server Components as a place to prefetch data, not to render query results.

### Nested Prefetching

You can use `HydrationBoundary` at any level of the component tree, not just at the page level:

```tsx
// app/posts/[id]/page.tsx
export default async function PostPage({ params }) {
  const queryClient = getQueryClient()
  await queryClient.prefetchQuery(postOptions(params.id))
  await queryClient.prefetchQuery(commentsOptions(params.id))

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <Post id={params.id} />
      <Suspense fallback={<CommentsSkeleton />}>
        <Comments id={params.id} />
      </Suspense>
    </HydrationBoundary>
  )
}
```

## Testing

### Setup

```tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,       // don't retry in tests
        gcTime: Infinity,   // prevent "did not exit" errors in Jest
      },
    },
  })

function createWrapper() {
  const queryClient = createTestQueryClient()
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}
```

### Testing hooks

```tsx
const { result } = renderHook(() => useCustomHook(), {
  wrapper: createWrapper(),
})
await waitFor(() => expect(result.current.isSuccess).toBe(true))
expect(result.current.data).toEqual('Hello')
```

### Testing components

```tsx
function renderWithQueryClient(ui: React.ReactElement) {
  const queryClient = createTestQueryClient()
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  )
}

test('renders todos', async () => {
  renderWithQueryClient(<Todos />)
  await screen.findByText('Todo 1')
})
```

### Key testing tips

- **Turn off retries** for tests (`retry: false`).
- **Set `gcTime: Infinity`** for Jest to avoid "did not exit" warnings.
- **Create a new QueryClient per test** to avoid shared state.
- Use **`msw`** (Mock Service Worker) or **`nock`** for mocking network requests.
- Use **`waitFor`** for async assertions.
- With **Vitest**, prefer `vi.fn()` over `jest.fn()`.
- For SSR testing, mock `window` or use `isServer` from `@tanstack/react-query`.

### Mocking with MSW (recommended)

```tsx
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'

const server = setupServer(
  http.get('/api/todos', () => {
    return HttpResponse.json([
      { id: 1, title: 'Todo 1' },
      { id: 2, title: 'Todo 2' },
    ])
  }),
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```
