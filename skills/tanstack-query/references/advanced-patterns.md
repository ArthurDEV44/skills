# TanStack Query Advanced Patterns

## Table of Contents
- [Parallel Queries](#parallel-queries)
- [Dependent Queries](#dependent-queries)
- [Paginated Queries](#paginated-queries)
- [Infinite Queries](#infinite-queries)
- [Optimistic Updates](#optimistic-updates)
- [useMutationState](#usemutationstate)
- [Query Cancellation](#query-cancellation)
- [Disabling Queries](#disabling-queries)
- [Query Retries](#query-retries)
- [Initial Query Data](#initial-query-data)
- [Placeholder Query Data](#placeholder-query-data)
- [Default Query Function](#default-query-function)
- [Render Optimizations](#render-optimizations)
- [Window Focus Refetching](#window-focus-refetching)
- [Network Mode](#network-mode)
- [Background Fetching Indicators](#background-fetching-indicators)
- [Prefetching](#prefetching)

## Parallel Queries

Use multiple `useQuery` hooks side-by-side:
```tsx
const usersQuery = useQuery({ queryKey: ['users'], queryFn: fetchUsers })
const teamsQuery = useQuery({ queryKey: ['teams'], queryFn: fetchTeams })
```

For dynamic number of parallel queries, use `useQueries`:
```tsx
const userQueries = useQueries({
  queries: users.map((user) => ({
    queryKey: ['user', user.id],
    queryFn: () => fetchUserById(user.id),
  })),
})
```

With suspense, use `useSuspenseQueries` instead (individual `useSuspenseQuery` calls run serially).

`useQueries` also supports a `combine` option to merge results:
```tsx
const { data, pending } = useQueries({
  queries: ids.map((id) => ({
    queryKey: ['item', id],
    queryFn: () => fetchItem(id),
  })),
  combine: (results) => ({
    data: results.map((r) => r.data),
    pending: results.some((r) => r.isPending),
  }),
})
```

## Dependent Queries

Use `enabled` to create serial/dependent queries:

```tsx
const { data: user } = useQuery({
  queryKey: ['user', email],
  queryFn: getUserByEmail,
})

const { data: projects } = useQuery({
  queryKey: ['projects', user?.id],
  queryFn: getProjectsByUser,
  enabled: !!user?.id,
})
```

Dependent `useQueries`:
```tsx
const { data: userIds } = useQuery({
  queryKey: ['users'],
  queryFn: getUsersData,
  select: (users) => users.map((user) => user.id),
})

const usersMessages = useQueries({
  queries: userIds
    ? userIds.map((id) => ({
        queryKey: ['messages', id],
        queryFn: () => getMessagesByUsers(id),
      }))
    : [],
})
```

**Performance note**: Dependent queries create request waterfalls. Prefer restructuring APIs (e.g., `getProjectsByUserEmail`) to flatten waterfalls when possible.

## Paginated Queries

Include page in query key. Use `placeholderData` with `keepPreviousData` for smooth transitions:

```tsx
import { keepPreviousData, useQuery } from '@tanstack/react-query'

const { data, isPlaceholderData } = useQuery({
  queryKey: ['projects', page],
  queryFn: () => fetchProjects(page),
  placeholderData: keepPreviousData,
})

// Disable next page button while showing placeholder
<button
  disabled={isPlaceholderData || !data?.hasMore}
  onClick={() => setPage(old => old + 1)}
>
  Next Page
</button>
```

## Infinite Queries

```tsx
const {
  data,
  fetchNextPage,
  hasNextPage,
  isFetchingNextPage,
} = useInfiniteQuery({
  queryKey: ['projects'],
  queryFn: ({ pageParam }) => fetchProjects(pageParam),
  initialPageParam: 0,
  getNextPageParam: (lastPage, pages) => lastPage.nextCursor,
})

// data.pages = array of fetched pages
// data.pageParams = array of page params
```

**Bi-directional**: Add `getPreviousPageParam`, use `fetchPreviousPage`, `hasPreviousPage`.

**Reversed order**: Use `select`:
```tsx
select: (data) => ({
  pages: [...data.pages].reverse(),
  pageParams: [...data.pageParams].reverse(),
})
```

**Limit pages in memory** with `maxPages`:
```tsx
useInfiniteQuery({
  queryKey: ['projects'],
  queryFn: fetchProjects,
  initialPageParam: 0,
  getNextPageParam: (lastPage, pages) => lastPage.nextCursor,
  getPreviousPageParam: (firstPage, pages) => firstPage.prevCursor,
  maxPages: 3,
})
```

**Manual updates**:
```tsx
// Remove first page
queryClient.setQueryData(['projects'], (data) => ({
  pages: data.pages.slice(1),
  pageParams: data.pageParams.slice(1),
}))
```

**No cursor from API?** Use pageParam as cursor:
```tsx
getNextPageParam: (lastPage, allPages, lastPageParam) => {
  if (lastPage.length === 0) return undefined
  return lastPageParam + 1
},
```

**Type-safe with infiniteQueryOptions**:
```tsx
import { infiniteQueryOptions } from '@tanstack/react-query'

const projectsOptions = infiniteQueryOptions({
  queryKey: ['projects'],
  queryFn: ({ pageParam }) => fetchProjects(pageParam),
  initialPageParam: 0,
  getNextPageParam: (lastPage) => lastPage.nextCursor,
})

useInfiniteQuery(projectsOptions)
queryClient.prefetchInfiniteQuery(projectsOptions)
```

## Optimistic Updates

### Via the UI (simpler, recommended for single location)

```tsx
const { isPending, variables, mutate, isError } = useMutation({
  mutationFn: (newTodo: string) => axios.post('/api/data', { text: newTodo }),
  onSettled: () => queryClient.invalidateQueries({ queryKey: ['todos'] }),
})

// Render optimistic item:
{isPending && <li style={{ opacity: 0.5 }}>{variables}</li>}
{isError && (
  <li style={{ color: 'red' }}>
    {variables}
    <button onClick={() => mutate(variables)}>Retry</button>
  </li>
)}
```

Access mutation variables from other components via `useMutationState`:
```tsx
const variables = useMutationState<string>({
  filters: { mutationKey: ['addTodo'], status: 'pending' },
  select: (mutation) => mutation.state.variables,
})
```

### Via the cache (automatic multi-location updates)

```tsx
const queryClient = useQueryClient()

useMutation({
  mutationFn: updateTodo,
  onMutate: async (newTodo) => {
    // Cancel outgoing refetches
    await queryClient.cancelQueries({ queryKey: ['todos'] })
    // Snapshot previous value
    const previousTodos = queryClient.getQueryData(['todos'])
    // Optimistically update cache
    queryClient.setQueryData(['todos'], (old) => [...old, newTodo])
    // Return context with rollback data
    return { previousTodos }
  },
  onError: (err, newTodo, context) => {
    // Rollback on error
    queryClient.setQueryData(['todos'], context.previousTodos)
  },
  onSettled: () => {
    // Always refetch after error or success
    queryClient.invalidateQueries({ queryKey: ['todos'] })
  },
})
```

### Optimistic update for a single entity

```tsx
useMutation({
  mutationFn: (updatedTodo) => axios.put(`/todos/${updatedTodo.id}`, updatedTodo),
  onMutate: async (updatedTodo) => {
    await queryClient.cancelQueries({ queryKey: ['todos', updatedTodo.id] })
    const previousTodo = queryClient.getQueryData(['todos', updatedTodo.id])
    queryClient.setQueryData(['todos', updatedTodo.id], updatedTodo)
    return { previousTodo }
  },
  onError: (err, updatedTodo, context) => {
    queryClient.setQueryData(['todos', updatedTodo.id], context.previousTodo)
  },
  onSettled: (data, error, variables) => {
    queryClient.invalidateQueries({ queryKey: ['todos', variables.id] })
  },
})
```

## useMutationState

Track mutation state across components. Useful for showing global loading states or pending mutations.

```tsx
import { useMutationState } from '@tanstack/react-query'

// Get all pending mutation variables for a specific key
const pendingTodos = useMutationState({
  filters: { mutationKey: ['addTodo'], status: 'pending' },
  select: (mutation) => mutation.state.variables,
})

// Get count of pending mutations
const pendingCount = useMutationState({
  filters: { status: 'pending' },
  select: (mutation) => mutation.state.status,
}).length

// Get last mutation result
const lastResult = useMutationState({
  filters: { mutationKey: ['addTodo'], status: 'success' },
  select: (mutation) => mutation.state.data,
}).at(-1)
```

## Query Cancellation

TanStack Query provides an `AbortSignal` via `QueryFunctionContext`:

```tsx
useQuery({
  queryKey: ['todos'],
  queryFn: async ({ signal }) => {
    const response = await fetch('/todos', { signal })
    return response.json()
  },
})
```

Works with axios:
```tsx
queryFn: ({ signal }) => axios.get('/todos', { signal })
```

Manual cancellation:
```tsx
queryClient.cancelQueries({ queryKey: ['todos'] })
```

**Limitation**: Cancellation does not work with Suspense hooks.

## Disabling Queries

```tsx
// With enabled (allows refetch)
useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodoList,
  enabled: false,
})

// With skipToken (type-safe, refetch won't work)
import { skipToken } from '@tanstack/react-query'

useQuery({
  queryKey: ['todos', filter],
  queryFn: filter ? () => fetchTodos(filter) : skipToken,
})
```

**`enabled` can be a function** receiving the query:
```tsx
useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
  enabled: (query) => query.state.data !== undefined, // only refetch, never initial fetch
})
```

**Lazy queries** with conditional `enabled`:
```tsx
const { data } = useQuery({
  queryKey: ['todos', filter],
  queryFn: () => fetchTodos(filter),
  enabled: !!filter,
})
```

`isLoading` = `isPending && isFetching` (true only when actually fetching for the first time).

## Query Retries

```tsx
const result = useQuery({
  queryKey: ['todos', 1],
  queryFn: fetchTodoListPage,
  retry: 10, // default: 3 (0 on server)
  retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
})
```

Options: `false` (disable), number, `true` (infinite), `(failureCount, error) => boolean`.

## Initial Query Data

Persist to cache with `initialData`:
```tsx
const result = useQuery({
  queryKey: ['todos'],
  queryFn: () => fetch('/todos'),
  initialData: initialTodos,
  // Optional: when was this data last fresh?
  initialDataUpdatedAt: initialTodosUpdatedTimestamp,
})
```

From another query's cache:
```tsx
const result = useQuery({
  queryKey: ['todo', todoId],
  queryFn: () => fetch('/todos'),
  initialData: () =>
    queryClient.getQueryData(['todos'])?.find((d) => d.id === todoId),
  initialDataUpdatedAt: () =>
    queryClient.getQueryState(['todos'])?.dataUpdatedAt,
})
```

## Placeholder Query Data

Not persisted to cache. Query starts in `success` state with `isPlaceholderData: true`:

```tsx
const result = useQuery({
  queryKey: ['todos'],
  queryFn: () => fetch('/todos'),
  placeholderData: placeholderTodos,
})

// Function form (access previous data for transitions):
const result = useQuery({
  queryKey: ['todos', id],
  queryFn: () => fetch(`/todos/${id}`),
  placeholderData: (previousData, previousQuery) => previousData,
})
```

## Default Query Function

```tsx
const defaultQueryFn = async ({ queryKey }) => {
  const { data } = await axios.get(`https://api.example.com${queryKey[0]}`)
  return data
}

const queryClient = new QueryClient({
  defaultOptions: { queries: { queryFn: defaultQueryFn } },
})

// Then just pass keys:
useQuery({ queryKey: ['/posts'] })
```

## Render Optimizations

- **Structural sharing**: Unchanged data keeps same reference.
- **Tracked properties**: Only re-renders when accessed properties change (via Proxy). Don't use rest destructuring (`const { data, ...rest } = useQuery(...)` disables tracking).
- **`select`**: Subscribe to subset of data:

```tsx
const { data } = useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
  select: (data) => data.length,
})
// Only re-renders when length changes
```

Memoize select for stability:
```tsx
const selectTodoCount = (data) => data.length
const { data } = useQuery({ queryKey: ['todos'], queryFn: fetchTodos, select: selectTodoCount })
```

## Window Focus Refetching

Disable globally:
```tsx
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
})
```

Disable per-query:
```tsx
useQuery({ queryKey: ['todos'], queryFn: fetchTodos, refetchOnWindowFocus: false })
```

Custom focus events with `focusManager.setEventListener`.
React Native: Use `AppState` with `focusManager.setFocused`.

## Network Mode

- `'online'` (default): Queries/mutations don't fire without network. Retries pause offline.
- `'always'`: Ignores online/offline state. Good for AsyncStorage or non-network queries.
- `'offlineFirst'`: Run queryFn once, pause retries if offline. Good for service workers / HTTP caching.

## Background Fetching Indicators

Per-query:
```tsx
const { isFetching } = useQuery({ queryKey: ['todos'], queryFn: fetchTodos })
```

Global:
```tsx
import { useIsFetching } from '@tanstack/react-query'
const isFetching = useIsFetching()
```

For mutations:
```tsx
import { useIsMutating } from '@tanstack/react-query'
const isMutating = useIsMutating()
```

## Prefetching

### Basic prefetch
```tsx
await queryClient.prefetchQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
})
```

### With queryOptions (recommended)
```tsx
queryClient.prefetchQuery(todosOptions(filters))
queryClient.prefetchInfiniteQuery(projectsInfiniteOptions())
```

### Prefetch infinite query with multiple pages
```tsx
await queryClient.prefetchInfiniteQuery({
  queryKey: ['projects'],
  queryFn: fetchProjects,
  initialPageParam: 0,
  getNextPageParam: (lastPage, pages) => lastPage.nextCursor,
  pages: 3, // prefetch first 3 pages
})
```

### In event handlers
```tsx
const prefetch = () => {
  queryClient.prefetchQuery({
    queryKey: ['details'],
    queryFn: getDetailsData,
    staleTime: 60000,
  })
}
<button onMouseEnter={prefetch} onFocus={prefetch}>Show Details</button>
```

### In components (flatten waterfalls)

With Suspense, use `usePrefetchQuery` to start fetching without subscribing:
```tsx
// Parent component starts the fetch
usePrefetchQuery(todosOptions(filters))
usePrefetchInfiniteQuery(projectsInfiniteOptions())

// Child component (inside Suspense) consumes the data
const { data } = useSuspenseQuery(todosOptions(filters))
```

Without Suspense, use `useQuery` with `notifyOnChangeProps: []` to prefetch silently:
```tsx
useQuery({
  queryKey: ['article-comments', id],
  queryFn: getArticleCommentsById,
  notifyOnChangeProps: [],
})
```

### Manual priming
```tsx
queryClient.setQueryData(['todos'], todos)
```

### Prefetching on router transitions
```tsx
// React Router loader
export const loader = async () => {
  const queryClient = getQueryClient()
  await queryClient.ensureQueryData(todosOptions())
  return null
}
```

`ensureQueryData` only fetches if the data doesn't already exist or is stale.
