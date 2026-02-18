# TanStack Table State Management (React)

> Official docs: https://tanstack.com/table/latest/docs/guide/table-state

## Internal State (Default)

By default, the table manages all state internally. Access it via `table.getState()`:

```tsx
const table = useReactTable({ columns, data, getCoreRowModel: getCoreRowModel() })
console.log(table.getState()) // entire state
console.log(table.getState().sorting) // specific state
```

## Initial State

Set initial values without managing state yourself:

```tsx
const table = useReactTable({
  columns, data,
  getCoreRowModel: getCoreRowModel(),
  initialState: {
    columnOrder: ['age', 'firstName', 'lastName'],
    columnVisibility: { id: false },
    expanded: true,
    sorting: [{ id: 'age', desc: true }],
    pagination: { pageIndex: 0, pageSize: 25 },
  },
})
```

> Do NOT pass the same state to both `initialState` and `state`. `state` takes precedence.

## Supported State Properties

`initialState` and `state` support all of these:

- `VisibilityTableState` - Column visibility settings
- `ColumnOrderTableState` - Column order configuration
- `ColumnPinningTableState` - Pinned columns (left/right)
- `FiltersTableState` - Column filters and global filter
- `SortingTableState` - Column sorting
- `ExpandedTableState` - Expanded row groups
- `GroupingTableState` - Row grouping configuration
- `ColumnSizingTableState` - Column sizes
- `PaginationTableState` - Pagination settings (pageIndex, pageSize)
- `RowSelectionTableState` - Selected rows
- `RowPinningState` - Pinned rows (top/bottom)

## Controlled State (Individual)

Control only the state you need. Requires both `state.X` and `onXChange`:

```tsx
const [sorting, setSorting] = useState<SortingState>([])
const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 15 })

const table = useReactTable({
  columns, data,
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
  getPaginationRowModel: getPaginationRowModel(),
  state: { sorting, columnFilters, pagination },
  onSortingChange: setSorting,
  onColumnFiltersChange: setColumnFilters,
  onPaginationChange: setPagination,
})
```

## Fully Controlled State

Control entire state with `onStateChange`:

```tsx
const table = useReactTable({
  columns, data,
  getCoreRowModel: getCoreRowModel(),
})
const [state, setState] = useState({ ...table.initialState })

table.setOptions(prev => ({
  ...prev,
  state,
  onStateChange: setState,
}))
```

## Updater Callbacks

`on[State]Change` callbacks receive either a value or an updater function (same API as React's `useState`):

```tsx
onSortingChange: (updater) => {
  const newValue = updater instanceof Function ? updater(sorting) : updater
  // custom logic here (e.g., sync to URL params, log, etc.)
  setSorting(newValue)
}
```

This pattern allows intercepting state changes:

```tsx
// Sync pagination to URL search params
onPaginationChange: (updater) => {
  const next = updater instanceof Function ? updater(pagination) : updater
  setPagination(next)
  const params = new URLSearchParams(window.location.search)
  params.set('page', String(next.pageIndex))
  params.set('size', String(next.pageSize))
  window.history.replaceState({}, '', `?${params}`)
}
```

## State Types

Import TypeScript types for type-safe state:

```tsx
import type {
  SortingState,
  ColumnFiltersState,
  PaginationState,
  RowSelectionState,
  ExpandedState,
  ColumnVisibilityState,
  ColumnPinningState,
  ColumnSizingState,
  GroupingState,
  RowPinningState,
} from '@tanstack/react-table'
```

## Common Pattern: Server-Side Data Fetching

```tsx
const [sorting, setSorting] = useState<SortingState>([])
const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 15 })

const query = useQuery({
  queryKey: ['data', sorting, columnFilters, pagination],
  queryFn: () => fetchData(sorting, columnFilters, pagination),
})

const table = useReactTable({
  columns,
  data: query.data?.rows ?? [],
  manualSorting: true,
  manualFiltering: true,
  manualPagination: true,
  rowCount: query.data?.totalCount,
  state: { sorting, columnFilters, pagination },
  onSortingChange: setSorting,
  onColumnFiltersChange: setColumnFilters,
  onPaginationChange: setPagination,
  getCoreRowModel: getCoreRowModel(),
})
```

## Common Pattern: URL-Synced State

```tsx
import { useSearchParams } from 'react-router-dom'

const [searchParams, setSearchParams] = useSearchParams()

const sorting: SortingState = searchParams.get('sort')
  ? [{ id: searchParams.get('sort')!, desc: searchParams.get('desc') === 'true' }]
  : []

const pagination: PaginationState = {
  pageIndex: Number(searchParams.get('page') ?? 0),
  pageSize: Number(searchParams.get('size') ?? 20),
}

const table = useReactTable({
  // ...
  state: { sorting, pagination },
  onSortingChange: (updater) => {
    const next = updater instanceof Function ? updater(sorting) : updater
    setSearchParams(prev => {
      if (next.length > 0) {
        prev.set('sort', next[0].id)
        prev.set('desc', String(next[0].desc))
      } else {
        prev.delete('sort')
        prev.delete('desc')
      }
      return prev
    })
  },
  onPaginationChange: (updater) => {
    const next = updater instanceof Function ? updater(pagination) : updater
    setSearchParams(prev => {
      prev.set('page', String(next.pageIndex))
      prev.set('size', String(next.pageSize))
      return prev
    })
  },
})
```
