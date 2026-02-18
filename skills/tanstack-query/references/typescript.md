# TanStack Query TypeScript Guide

## Table of Contents
- [Type Inference](#type-inference)
- [Type Narrowing](#type-narrowing)
- [Typing Errors](#typing-errors)
- [Registering Global Types](#registering-global-types)
- [Typing Query Options](#typing-query-options)
- [Typing Infinite Query Options](#typing-infinite-query-options)
- [Typing Mutation Options](#typing-mutation-options)
- [skipToken for Typesafe Disabling](#skiptoken-for-typesafe-disabling)
- [GraphQL Integration](#graphql-integration)

## Type Inference

Types flow through automatically from `queryFn`:

```tsx
const { data } = useQuery({
  //    ^? const data: number | undefined
  queryKey: ['test'],
  queryFn: () => Promise.resolve(5),
})
```

With `select`:
```tsx
const { data } = useQuery({
  //    ^? const data: string | undefined
  queryKey: ['test'],
  queryFn: () => Promise.resolve(5),
  select: (data) => data.toString(),
})
```

Ensure `queryFn` has a well-defined return type (most fetch libraries return `any`):
```tsx
const fetchGroups = (): Promise<Group[]> =>
  axios.get('/groups').then((response) => response.data)

const { data } = useQuery({ queryKey: ['groups'], queryFn: fetchGroups })
//      ^? const data: Group[] | undefined
```

## Type Narrowing

Uses discriminated union on `status` field:
```tsx
const { data, isSuccess } = useQuery({
  queryKey: ['test'],
  queryFn: () => Promise.resolve(5),
})

if (isSuccess) {
  data // ^? const data: number
}
```

With `useSuspenseQuery`, `data` is always defined:
```tsx
const { data } = useSuspenseQuery({
  queryKey: ['test'],
  queryFn: () => Promise.resolve(5),
})
// data: number (never undefined)
```

## Typing Errors

Default error type is `Error`. For custom error types:
```tsx
const { error } = useQuery<Group[], string>(['groups'], fetchGroups)
//      ^? const error: string | null
```

Better: use type narrowing with `AxiosError`:
```tsx
const { error } = useQuery({ queryKey: ['groups'], queryFn: fetchGroups })
if (axios.isAxiosError(error)) {
  error // ^? const error: AxiosError
}
```

### Register Global Error Type

```tsx
import '@tanstack/react-query'

declare module '@tanstack/react-query' {
  interface Register {
    defaultError: AxiosError // or Error, or custom type
  }
}
```

## Registering Global Types

### Global Meta
```ts
import '@tanstack/react-query'

interface MyMeta extends Record<string, unknown> {
  // your fields
}

declare module '@tanstack/react-query' {
  interface Register {
    queryMeta: MyMeta
    mutationMeta: MyMeta
  }
}
```

### Global Query/Mutation Keys
```ts
import '@tanstack/react-query'

type QueryKey = ['dashboard' | 'marketing', ...ReadonlyArray<unknown>]

declare module '@tanstack/react-query' {
  interface Register {
    queryKey: QueryKey
    mutationKey: QueryKey
  }
}
```

## Typing Query Options

Use `queryOptions` helper for shared options with full inference:
```ts
import { queryOptions } from '@tanstack/react-query'

function groupOptions(id: number) {
  return queryOptions({
    queryKey: ['groups', id] as const,
    queryFn: () => fetchGroups(id),
    staleTime: 5 * 1000,
  })
}

// Full type inference everywhere:
useQuery(groupOptions(1))
useSuspenseQuery(groupOptions(5))
queryClient.prefetchQuery(groupOptions(23))

// Type-safe cache access:
const data = queryClient.getQueryData(groupOptions(42).queryKey)
//     ^? const data: Group[] | undefined

// Type-safe cache update:
queryClient.setQueryData(groupOptions(42).queryKey, newGroups)
```

Override per-component:
```ts
const query = useQuery({
  ...groupOptions(1),
  select: (data) => data.groupName,
})
```

## Typing Infinite Query Options

Use `infiniteQueryOptions` for type-safe infinite queries:
```ts
import { infiniteQueryOptions } from '@tanstack/react-query'

function projectsInfiniteOptions(filters?: Filters) {
  return infiniteQueryOptions({
    queryKey: ['projects', filters] as const,
    queryFn: ({ pageParam }) => fetchProjects({ ...filters, cursor: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.nextCursor,
  })
}

// Full inference:
useInfiniteQuery(projectsInfiniteOptions())
useSuspenseInfiniteQuery(projectsInfiniteOptions({ status: 'active' }))
queryClient.prefetchInfiniteQuery(projectsInfiniteOptions())

// Type-safe cache access:
const data = queryClient.getQueryData(projectsInfiniteOptions().queryKey)
//     ^? InfiniteData<ProjectPage, number> | undefined
```

## Typing Mutation Options

Use `mutationOptions` for type-safe shared mutation options:
```ts
import { mutationOptions } from '@tanstack/react-query'

function addTodoMutationOptions() {
  return mutationOptions({
    mutationKey: ['addTodo'] as const,
    mutationFn: (newTodo: NewTodo) => api.addTodo(newTodo),
    onSuccess: (data) => {
      // data is fully typed
    },
  })
}

useMutation(addTodoMutationOptions())
```

## skipToken for Typesafe Disabling

```tsx
import { skipToken, useQuery } from '@tanstack/react-query'

function Todos() {
  const [filter, setFilter] = React.useState<string | undefined>()

  const { data } = useQuery({
    queryKey: ['todos', filter],
    // Type-safe: when filter is undefined, skip the query
    queryFn: filter ? () => fetchTodos(filter) : skipToken,
  })
}
```

**Why skipToken over `enabled: false`?**
- `enabled: false` doesn't narrow the queryFn parameter types
- With `skipToken`, TypeScript knows that when queryFn runs, `filter` is defined
- `skipToken` prevents calling `refetch()` (which would fail anyway without the param)

Note: `refetch()` does not work with `skipToken`. Use `enabled: false` if you need manual refetch.

## GraphQL Integration

With `graphql-request` v5+ and GraphQL Code Generator:

```tsx
import request from 'graphql-request'
import { useQuery } from '@tanstack/react-query'
import { graphql } from './gql/gql'

const allFilmsQuery = graphql(`
  query allFilmsWithVariablesQuery($first: Int!) {
    allFilms(first: $first) {
      edges { node { id, title } }
    }
  }
`)

function App() {
  const { data } = useQuery({
    queryKey: ['films'],
    queryFn: async () =>
      request(
        'https://swapi-graphql.netlify.app/.netlify/functions/index',
        allFilmsQuery,
        { first: 10 }, // variables are type-checked
      ),
  })
}
```

### Type-safe query key factories

```ts
const todoKeys = {
  all: ['todos'] as const,
  lists: () => [...todoKeys.all, 'list'] as const,
  list: (filters: TodoFilters) => [...todoKeys.lists(), filters] as const,
  details: () => [...todoKeys.all, 'detail'] as const,
  detail: (id: number) => [...todoKeys.details(), id] as const,
}

// Usage:
useQuery({ queryKey: todoKeys.list({ status: 'done' }), queryFn: ... })
queryClient.invalidateQueries({ queryKey: todoKeys.lists() }) // invalidates all lists
queryClient.invalidateQueries({ queryKey: todoKeys.all })     // invalidates everything
```
