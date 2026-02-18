---
name: tanstack-query
description: "TanStack Query (React Query) v5 best practices for fetching, caching and updating server state in React. Use when: (1) Data fetching with useQuery, useSuspenseQuery, useInfiniteQuery, (2) Mutations with useMutation and cache invalidation, (3) Query key design and queryOptions/infiniteQueryOptions, (4) Optimistic updates and cache manipulation, (5) Pagination and infinite scroll, (6) Prefetching with usePrefetchQuery and waterfall optimization, (7) SSR with Next.js pages/app router, Remix, streaming, (8) Suspense and Server Components integration, (9) TypeScript patterns (queryOptions, mutationOptions, type inference, global types), (10) Testing React Query hooks, (11) QueryClient setup, DevTools, staleTime, gcTime, retry, (12) useMutationState, (13) Any @tanstack/react-query imports."
---

# TanStack Query (React Query) v5

## Quick Setup

```tsx
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1 minute (avoid aggressive refetching)
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MyComponent />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

function MyComponent() {
  const { data, isPending, error } = useQuery({
    queryKey: ['todos'],
    queryFn: async () => {
      const res = await fetch('/api/todos')
      if (!res.ok) throw new Error('Network response was not ok')
      return res.json()
    },
  })
}
```

## Key API Patterns

### useQuery
```tsx
const { data, isPending, isError, error, isFetching } = useQuery({
  queryKey: ['todos', { status, page }],
  queryFn: () => fetchTodos({ status, page }),
  staleTime: 5 * 60 * 1000,
  enabled: !!userId,
})
```

### useMutation + Invalidation
```tsx
const queryClient = useQueryClient()
const mutation = useMutation({
  mutationFn: (newTodo) => axios.post('/todos', newTodo),
  onSuccess: () => queryClient.invalidateQueries({ queryKey: ['todos'] }),
})
mutation.mutate({ title: 'New Todo' })
```

### queryOptions (type-safe shared options)
```tsx
const todosOptions = (filters: Filters) => queryOptions({
  queryKey: ['todos', filters],
  queryFn: () => fetchTodos(filters),
  staleTime: 5000,
})

useQuery(todosOptions(filters))
useSuspenseQuery(todosOptions(filters))
queryClient.prefetchQuery(todosOptions(filters))
```

### Infinite Queries
```tsx
const { data, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteQuery({
  queryKey: ['projects'],
  queryFn: ({ pageParam }) => fetchProjects(pageParam),
  initialPageParam: 0,
  getNextPageParam: (lastPage) => lastPage.nextCursor,
})
```

### Suspense
```tsx
const { data } = useSuspenseQuery({ queryKey: ['todos'], queryFn: fetchTodos })
// data is always defined - errors/loading handled by Suspense/ErrorBoundary
```

### Prefetching (flatten waterfalls)
```tsx
// In event handlers
queryClient.prefetchQuery(todosOptions(filters))

// In components with Suspense
usePrefetchQuery(todosOptions(filters))
usePrefetchInfiniteQuery(projectsInfiniteOptions())
```

### Tracking Mutation State
```tsx
const pendingVariables = useMutationState({
  filters: { mutationKey: ['addTodo'], status: 'pending' },
  select: (mutation) => mutation.state.variables,
})
```

## Critical Defaults to Remember

- `staleTime: 0` - Data considered stale immediately (refetches on mount/focus/reconnect)
- `staleTime: 'static'` - Data never considered stale, not even on manual invalidation
- `gcTime: 5 * 60 * 1000` - Unused queries garbage collected after 5 min (`Infinity` during SSR)
- `retry: 3` on client, `0` on server - Exponential backoff
- `refetchOnWindowFocus: true` - Refetches stale queries on tab focus
- `structuralSharing: true` - Preserves referential identity for unchanged data
- Error type defaults to `Error` (configurable via Register interface)
- Mutations do **not** retry by default (set `retry` explicitly)

## Reference Documentation

Consult these files for detailed patterns and examples:

- **Core concepts** (queries, mutations, keys, functions, invalidation, filters): See [references/core-concepts.md](references/core-concepts.md)
- **Advanced patterns** (pagination, infinite queries, optimistic updates, prefetching, dependent queries, cancellation, render optimizations): See [references/advanced-patterns.md](references/advanced-patterns.md)
- **SSR, Suspense & Server Components** (Next.js pages/app router, Remix, hydration, streaming, testing): See [references/ssr-and-suspense.md](references/ssr-and-suspense.md)
- **TypeScript** (type inference, global types, queryOptions, infiniteQueryOptions, skipToken, GraphQL): See [references/typescript.md](references/typescript.md)

## Common Pitfalls

- **Don't create QueryClient at module level** in SSR - leaks data between requests. Use `useState` (Pages Router) or `getQueryClient()` singleton pattern (App Router).
- **`fetch` doesn't throw on HTTP errors** - must check `response.ok` and throw manually in `queryFn`.
- **Query keys are arrays** - `['todos']` not `'todos'`. Object key order doesn't matter, array order does.
- **`enabled: false`** prevents all automatic fetching. Use `skipToken` for type-safe conditional disabling.
- **Mutations don't retry** by default (unlike queries). Set `retry` explicitly if needed.
- **`setQueryData` requires immutability** - never mutate cached data directly, always return new objects/arrays.
- **Suspense queries run serially** in same component - use `useSuspenseQueries` for parallel execution.
- **`placeholderData: keepPreviousData`** replaces the removed `keepPreviousData` option from v4.
- **Don't render prefetched data in Server Components** - they can't revalidate. Server Components should prefetch; Client Components should render via useQuery/useSuspenseQuery.
- **Set `staleTime > 0` in SSR** - avoid refetching immediately on the client after hydration.
- **Don't use `queryClient.getQueryData` for rendering** - it won't trigger re-renders. Use `useQuery` or `useSuspenseQuery` instead.
