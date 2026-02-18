---
name: tanstack-table
description: "TanStack Table v8 headless UI library for building type-safe tables and datagrids in React, Vue, Solid, Svelte, Qwik, Angular, and Lit. Use when writing, reviewing, or refactoring code involving TanStack Table: (1) Creating tables with useReactTable, createColumnHelper, flexRender, (2) Implementing sorting, filtering, pagination, grouping, expanding, row selection, column pinning, column sizing/resizing, column ordering, column visibility, (3) Managing table state with onSortingChange, onPaginationChange, etc., (4) Using row models (getCoreRowModel, getSortedRowModel, getFilteredRowModel, getPaginationRowModel), (5) Server-side operations with manualSorting/manualFiltering/manualPagination, (6) Sticky column pinning with getCommonPinningStyles, row pinning, virtualization with TanStack Virtual, (7) Custom features with _features and declaration merging, column meta types, faceting, fuzzy filtering."
---

# TanStack Table v8

Headless UI library for building tables and datagrids. Provides state management, data processing, and APIs without markup or styles.

## Quick Start (React)

```tsx
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
} from '@tanstack/react-table'

type Person = { firstName: string; lastName: string; age: number }

const columnHelper = createColumnHelper<Person>()

const columns = [
  columnHelper.accessor('firstName', {
    header: 'First Name',
    cell: info => info.getValue(),
  }),
  columnHelper.accessor('lastName', { header: 'Last Name' }),
  columnHelper.accessor('age', { header: 'Age' }),
]

function MyTable({ data }: { data: Person[] }) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map(hg => (
          <tr key={hg.id}>
            {hg.headers.map(h => (
              <th key={h.id} colSpan={h.colSpan}>
                {h.isPlaceholder
                  ? null
                  : flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map(row => (
          <tr key={row.id}>
            {row.getVisibleCells().map(cell => (
              <td key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

## Critical Rules

1. **Stable references**: `data` and `columns` MUST use `useState`, `useMemo`, or be defined outside the component. Inline arrays cause infinite re-renders.
2. **Import only needed row models**: Tree-shakable. Only import `getCoreRowModel`, `getSortedRowModel`, etc. as needed.
3. **Use `flexRender`** for rendering headers and cells (handles string, JSX, and function column defs).
4. **Use `getVisibleCells()`** (not `getAllCells()`) when column visibility is used.
5. **Row model execution order**: Core -> Filtered -> Grouped -> Sorted -> Expanded -> Paginated.
6. **Use `getRowId`** for meaningful row IDs when data has a unique key (needed for stable row selection, pinning, expansion).
7. **Use `defaultColumn`** to set shared column defaults (size, minSize, maxSize, enableSorting, etc.) instead of repeating per column.
8. **Type-safe column meta**: Extend `ColumnMeta` via declaration merging for custom column metadata (e.g., `filterVariant`).
9. **Check `header.isPlaceholder`**: Always check before rendering header content to avoid rendering empty placeholder headers in grouped columns.

## Feature Quick Reference

| Feature | Row Model Import | Key State | Key API |
|---------|-----------------|-----------|---------|
| Sorting | `getSortedRowModel` | `SortingState` | `column.getToggleSortingHandler()` |
| Column Filter | `getFilteredRowModel` | `ColumnFiltersState` | `column.setFilterValue()` |
| Global Filter | `getFilteredRowModel` | `globalFilter` | `table.setGlobalFilter()` |
| Pagination | `getPaginationRowModel` | `PaginationState` | `table.nextPage()` / `table.previousPage()` |
| Row Selection | (none) | `RowSelectionState` | `row.getToggleSelectedHandler()` |
| Expanding | `getExpandedRowModel` | `ExpandedState` | `row.getToggleExpandedHandler()` |
| Grouping | `getGroupedRowModel` | `GroupingState` | `table.setGrouping()` |
| Col Visibility | (none) | `ColumnVisibilityState` | `column.getToggleVisibilityHandler()` |
| Col Pinning | (none) | `ColumnPinningState` | `column.pin('left'|'right'|false)` |
| Col Sizing | (none) | `ColumnSizingState` | `header.getResizeHandler()` |
| Col Ordering | (none) | `columnOrder: string[]` | `table.setColumnOrder()` |
| Row Pinning | (none) | `RowPinningState` | `row.pin('top'|'bottom'|false)` |
| Faceting | `getFacetedRowModel` + `getFacetedUniqueValues`/`getFacetedMinMaxValues` | (none) | `column.getFacetedUniqueValues()` |

## Server-Side Operations

Disable client-side processing with `manual*` options:

```tsx
const table = useReactTable({
  data,
  columns,
  getCoreRowModel: getCoreRowModel(),
  manualSorting: true, // data is pre-sorted
  manualFiltering: true, // data is pre-filtered
  manualPagination: true, // data is pre-paginated
  rowCount: totalRows, // tell table total count for pagination
  state: { sorting, columnFilters, pagination },
  onSortingChange: setSorting,
  onColumnFiltersChange: setColumnFilters,
  onPaginationChange: setPagination,
})
```

## Type-Safe Column Meta

Extend `ColumnMeta` to add custom metadata to column definitions:

```tsx
// Extend the module
declare module '@tanstack/react-table' {
  interface ColumnMeta<TData extends RowData, TValue> {
    filterVariant?: 'text' | 'range' | 'select'
    headerClassName?: string
  }
}

// Use in column defs
columnHelper.accessor('status', {
  header: 'Status',
  meta: { filterVariant: 'select' },
})

// Access in rendering
const filterVariant = column.columnDef.meta?.filterVariant
```

## Common Reusable Components

### IndeterminateCheckbox (for row selection)

```tsx
function IndeterminateCheckbox({
  indeterminate,
  ...rest
}: { indeterminate?: boolean } & React.InputHTMLAttributes<HTMLInputElement>) {
  const ref = React.useRef<HTMLInputElement>(null!)
  React.useEffect(() => {
    ref.current.indeterminate = !!indeterminate
  }, [indeterminate])
  return <input type="checkbox" ref={ref} {...rest} />
}
```

### Sticky Column Pinning Styles

```tsx
import { type Column, type CSSProperties } from '@tanstack/react-table'

function getCommonPinningStyles<TData>(column: Column<TData>): CSSProperties {
  const isPinned = column.getIsPinned()
  const isLastLeft = isPinned === 'left' && column.getIsLastColumn('left')
  const isFirstRight = isPinned === 'right' && column.getIsFirstColumn('right')
  return {
    boxShadow: isLastLeft
      ? '-4px 0 4px -4px gray inset'
      : isFirstRight
        ? '4px 0 4px -4px gray inset'
        : undefined,
    left: isPinned === 'left' ? `${column.getStart('left')}px` : undefined,
    right: isPinned === 'right' ? `${column.getAfter('right')}px` : undefined,
    opacity: isPinned ? 0.95 : 1,
    position: isPinned ? 'sticky' : 'relative',
    width: column.getSize(),
    zIndex: isPinned ? 1 : 0,
  }
}

// Apply to both <th> and <td>:
<th style={{ ...getCommonPinningStyles(header.column) }}>
<td style={{ ...getCommonPinningStyles(cell.column) }}>
```

## References

- **Core concepts, data setup, column defs, table instance, row models, APIs**: See [references/core-concepts.md](references/core-concepts.md)
- **All features (sorting, filtering, pagination, selection, pinning, sizing, grouping, expanding, faceting, row pinning, virtualization, custom features)**: See [references/features.md](references/features.md)
- **State management patterns (controlled, uncontrolled, fully controlled, server-side)**: See [references/state-management.md](references/state-management.md)
- **Migration from react-table v7 to v8**: See [references/migration-v8.md](references/migration-v8.md)
