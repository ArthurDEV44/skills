# TanStack Table Features Guide

> Official docs: https://tanstack.com/table/latest/docs/guide/sorting

## Table of Contents

- [Sorting](#sorting)
- [Column Filtering](#column-filtering)
- [Global Filtering](#global-filtering)
- [Fuzzy Filtering](#fuzzy-filtering)
- [Pagination](#pagination)
- [Row Selection](#row-selection)
- [Column Visibility](#column-visibility)
- [Column Ordering](#column-ordering)
- [Column Pinning](#column-pinning)
- [Column Sizing & Resizing](#column-sizing--resizing)
- [Expanding](#expanding)
- [Grouping](#grouping)
- [Column Faceting](#column-faceting)
- [Row Pinning](#row-pinning)
- [Virtualization](#virtualization)
- [Custom Features](#custom-features)
- [Common Reusable Components](#common-reusable-components)

---

## Sorting

> API: https://tanstack.com/table/latest/docs/api/features/sorting

### Setup

```tsx
import { getSortedRowModel } from '@tanstack/react-table'

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
})
```

### State

```ts
type SortingState = { id: string; desc: boolean }[]

const [sorting, setSorting] = useState<SortingState>([])
// Pass to table:
state: { sorting },
onSortingChange: setSorting,
```

### Built-in Sorting Functions

- `alphanumeric` / `alphanumericCaseSensitive` - Mixed alphanumeric (slower, more accurate)
- `text` / `textCaseSensitive` - String values (faster)
- `datetime` - Date objects
- `basic` - Simple `a > b` comparison (fastest)

### Custom Sorting Function

```ts
const mySort: SortingFn<TData> = (rowA, rowB, columnId) => {
  return // -1, 0, or 1 in ascending order
}
// In column def:
{ accessorKey: 'field', sortingFn: mySort }
// Or as global:
sortingFns: { mySort }
```

### Key Options

- `enableSorting` (table/column) - Enable/disable sorting
- `sortDescFirst` (table/column) - First sort direction
- `invertSorting` (column) - Invert sort order (for rankings)
- `sortUndefined` (column) - `'first'` | `'last'` | `false` | `-1` | `1`
- `enableSortingRemoval` (table) - Allow removing sort (default: true)
- `enableMultiSort` (table/column) - Multi-column sorting (Shift+click)
- `maxMultiSortColCount` (table) - Limit multi-sort columns
- `manualSorting` (table) - Disable client-side sorting for server-side

### Key APIs

- `column.getToggleSortingHandler()` - Click handler for sorting UI
- `column.getIsSorted()` - Returns `'asc'` | `'desc'` | `false`
- `column.toggleSorting(desc?, multi?)` - Toggle sorting programmatically
- `column.clearSorting()` - Clear sorting on this column
- `column.getCanSort()` - Whether this column can be sorted

### Sortable Header Pattern

```tsx
<th
  onClick={header.column.getToggleSortingHandler()}
  style={{ cursor: header.column.getCanSort() ? 'pointer' : 'default' }}
>
  {flexRender(header.column.columnDef.header, header.getContext())}
  {{ asc: ' ^', desc: ' v' }[header.column.getIsSorted() as string] ?? null}
</th>
```

---

## Column Filtering

> API: https://tanstack.com/table/latest/docs/api/features/column-filtering

### Setup

```tsx
import { getFilteredRowModel } from '@tanstack/react-table'

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
})
```

### State

```ts
type ColumnFiltersState = { id: string; value: unknown }[]

const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
state: { columnFilters },
onColumnFiltersChange: setColumnFilters,
```

### Built-in Filter Functions

- `includesString` / `includesStringSensitive`
- `equalsString` / `equalsStringSensitive`
- `arrIncludes` / `arrIncludesAll` / `arrIncludesSome`
- `equals` / `weakEquals`
- `inNumberRange`

### Custom Filter Function

```ts
const myFilter: FilterFn<any> = (row, columnId, filterValue, addMeta) => {
  return true // or false
}
// Attach optional behaviors:
myFilter.autoRemove = (val) => !val
myFilter.resolveFilterValue = (val) => val.toString().toLowerCase().trim()
```

### Key Options

- `filterFn` (column) - Specify filter function per column
- `enableColumnFilter` (column) / `enableColumnFilters` (table)
- `manualFiltering` (table) - For server-side filtering
- `filterFromLeafRows` (table) - Filter from leaf rows up (for expanding)
- `maxLeafRowFilterDepth` (table) - Max depth for leaf row filtering

### Key APIs

- `column.getFilterValue()` / `column.setFilterValue(value)`
- `column.getCanFilter()` / `column.getIsFiltered()`
- `table.getPreFilteredRowModel()` - Row model before filtering (useful for detecting column data types)

### Reusable Filter Component

```tsx
function Filter({ column, table }: { column: Column<any, any>; table: Table<any> }) {
  const firstValue = table.getPreFilteredRowModel().flatRows[0]?.getValue(column.id)
  const columnFilterValue = column.getFilterValue()

  return typeof firstValue === 'number' ? (
    <div onClick={e => e.stopPropagation()}>
      <input
        type="number"
        value={(columnFilterValue as [number, number])?.[0] ?? ''}
        onChange={e =>
          column.setFilterValue((old: [number, number]) => [e.target.value, old?.[1]])
        }
        placeholder="Min"
      />
      <input
        type="number"
        value={(columnFilterValue as [number, number])?.[1] ?? ''}
        onChange={e =>
          column.setFilterValue((old: [number, number]) => [old?.[0], e.target.value])
        }
        placeholder="Max"
      />
    </div>
  ) : (
    <input
      type="text"
      value={(columnFilterValue ?? '') as string}
      onChange={e => column.setFilterValue(e.target.value)}
      onClick={e => e.stopPropagation()}
      placeholder="Search..."
    />
  )
}
```

### Meta-Driven Filter Component (with faceting)

```tsx
function Filter({ column }: { column: Column<any, any> }) {
  const { filterVariant } = column.columnDef.meta ?? {}
  const sortedUniqueValues = React.useMemo(
    () =>
      filterVariant === 'range'
        ? []
        : Array.from(column.getFacetedUniqueValues().keys()).sort().slice(0, 5000),
    [column.getFacetedUniqueValues(), filterVariant],
  )

  if (filterVariant === 'range') {
    const [min, max] = column.getFacetedMinMaxValues() ?? [0, 0]
    return (
      <div className="flex gap-1">
        <DebouncedInput type="number" min={min} max={max}
          value={(column.getFilterValue() as [number, number])?.[0] ?? ''}
          onChange={val => column.setFilterValue((old: [number, number]) => [val, old?.[1]])}
          placeholder={`Min (${min})`} />
        <DebouncedInput type="number" min={min} max={max}
          value={(column.getFilterValue() as [number, number])?.[1] ?? ''}
          onChange={val => column.setFilterValue((old: [number, number]) => [old?.[0], val])}
          placeholder={`Max (${max})`} />
      </div>
    )
  }

  if (filterVariant === 'select') {
    return (
      <select value={column.getFilterValue() as string ?? ''}
        onChange={e => column.setFilterValue(e.target.value)}>
        <option value="">All</option>
        {sortedUniqueValues.map(value => (
          <option key={value} value={value}>{value}</option>
        ))}
      </select>
    )
  }

  return (
    <>
      <datalist id={column.id + '-list'}>
        {sortedUniqueValues.map(value => <option key={value} value={value} />)}
      </datalist>
      <DebouncedInput
        value={(column.getFilterValue() ?? '') as string}
        onChange={val => column.setFilterValue(val)}
        placeholder={`Search... (${column.getFacetedUniqueValues().size})`}
        list={column.id + '-list'}
      />
    </>
  )
}
```

---

## Global Filtering

> API: https://tanstack.com/table/latest/docs/api/features/global-filtering

### Setup

Same `getFilteredRowModel` as column filtering. Set `globalFilterFn` for the filter function.

```tsx
const [globalFilter, setGlobalFilter] = useState('')

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
  globalFilterFn: 'includesString',
  state: { globalFilter },
  onGlobalFilterChange: setGlobalFilter,
})
```

### UI

```tsx
<input
  value={globalFilter ?? ''}
  onChange={e => table.setGlobalFilter(String(e.target.value))}
  placeholder="Search all columns..."
/>
```

---

## Fuzzy Filtering

> Example: https://tanstack.com/table/latest/docs/framework/react/examples/filters-fuzzy

Requires `@tanstack/match-sorter-utils`:

```tsx
import { rankItem, compareItems } from '@tanstack/match-sorter-utils'

const fuzzyFilter: FilterFn<any> = (row, columnId, value, addMeta) => {
  const itemRank = rankItem(row.getValue(columnId), value)
  addMeta({ itemRank })
  return itemRank.passed
}

// Optional: fuzzy sort by rank
const fuzzySort: SortingFn<any> = (rowA, rowB, columnId) => {
  let dir = 0
  if (rowA.columnFiltersMeta[columnId]) {
    dir = compareItems(
      rowA.columnFiltersMeta[columnId]?.itemRank!,
      rowB.columnFiltersMeta[columnId]?.itemRank!
    )
  }
  return dir === 0 ? sortingFns.alphanumeric(rowA, rowB, columnId) : dir
}
```

---

## Pagination

> API: https://tanstack.com/table/latest/docs/api/features/pagination

### Client-Side

```tsx
import { getPaginationRowModel } from '@tanstack/react-table'

const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 10 })

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  getPaginationRowModel: getPaginationRowModel(),
  state: { pagination },
  onPaginationChange: setPagination,
})
```

### Server-Side

```tsx
const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  manualPagination: true,
  rowCount: totalRowCount, // or pageCount
  state: { pagination },
  onPaginationChange: setPagination,
})
```

### Key APIs

- `table.firstPage()` / `table.lastPage()` / `table.previousPage()` / `table.nextPage()`
- `table.getCanPreviousPage()` / `table.getCanNextPage()`
- `table.setPageIndex(index)` / `table.setPageSize(size)`
- `table.getPageCount()` / `table.getRowCount()`

### Options

- `autoResetPageIndex` (table) - Reset pageIndex on data changes (default: true, auto-disabled with manualPagination)

### Pagination Controls Pattern

```tsx
<div className="flex items-center gap-2">
  <button onClick={() => table.firstPage()} disabled={!table.getCanPreviousPage()}>{'<<'}</button>
  <button onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>{'<'}</button>
  <button onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>{'>'}</button>
  <button onClick={() => table.lastPage()} disabled={!table.getCanNextPage()}>{'>>'}</button>
  <span>
    Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount().toLocaleString()}
  </span>
  <span>| Go to page:
    <input type="number" min={1} max={table.getPageCount()}
      defaultValue={table.getState().pagination.pageIndex + 1}
      onChange={e => { const page = e.target.value ? Number(e.target.value) - 1 : 0; table.setPageIndex(page) }}
    />
  </span>
  <select value={table.getState().pagination.pageSize}
    onChange={e => table.setPageSize(Number(e.target.value))}>
    {[10, 20, 30, 50, 100].map(size => (
      <option key={size} value={size}>Show {size}</option>
    ))}
  </select>
</div>
```

---

## Row Selection

> API: https://tanstack.com/table/latest/docs/api/features/row-selection

### State

```ts
const [rowSelection, setRowSelection] = useState<RowSelectionState>({})

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  state: { rowSelection },
  onRowSelectionChange: setRowSelection,
  getRowId: row => row.uuid, // recommended for meaningful IDs
})
```

### Checkbox Column

```tsx
columnHelper.display({
  id: 'select',
  header: ({ table }) => (
    <IndeterminateCheckbox
      checked={table.getIsAllRowsSelected()}
      indeterminate={table.getIsSomeRowsSelected()}
      onChange={table.getToggleAllRowsSelectedHandler()}
    />
  ),
  cell: ({ row }) => (
    <IndeterminateCheckbox
      checked={row.getIsSelected()}
      disabled={!row.getCanSelect()}
      indeterminate={row.getIsSomeSelected()}
      onChange={row.getToggleSelectedHandler()}
    />
  ),
})
```

### IndeterminateCheckbox Component

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

### Key Options

- `enableRowSelection` (table) - Boolean or `(row) => boolean`
- `enableMultiRowSelection` (table) - Set `false` for single selection (radio buttons)
- `enableSubRowSelection` (table) - Auto-select sub-rows with parent

### Key APIs

- `table.getSelectedRowModel().rows` - Selected rows
- `table.getFilteredSelectedRowModel().rows` - Filtered selected rows
- `row.getIsSelected()` / `row.toggleSelected()`

---

## Column Visibility

> API: https://tanstack.com/table/latest/docs/api/features/column-visibility

### State

```ts
const [columnVisibility, setColumnVisibility] = useState({ colId: false })

state: { columnVisibility },
onColumnVisibilityChange: setColumnVisibility,
```

### Toggle UI

```tsx
{table.getAllColumns()
  .filter(column => column.getCanHide())
  .map(column => (
    <label key={column.id}>
      <input
        checked={column.getIsVisible()}
        onChange={column.getToggleVisibilityHandler()}
        type="checkbox"
      />
      {column.id}
    </label>
  ))}
```

Use `row.getVisibleCells()` (not `row.getAllCells()`) when rendering.

---

## Column Ordering

> API: https://tanstack.com/table/latest/docs/api/features/column-ordering

### State

```ts
const [columnOrder, setColumnOrder] = useState<string[]>(['col1', 'col2', 'col3'])

state: { columnOrder },
onColumnOrderChange: setColumnOrder,
```

Order is affected by: Column Pinning -> Manual Column Ordering -> Grouping.

### DnD Recommendations (React)

1. Use `@dnd-kit/core` (recommended)
2. Avoid `react-dnd` with React 18+
3. Consider native browser drag events for lightweight solutions

---

## Column Pinning

> API: https://tanstack.com/table/latest/docs/api/features/column-pinning

### State

```ts
const [columnPinning, setColumnPinning] = useState<ColumnPinningState>({
  left: ['expand-column'],
  right: ['actions-column'],
})

state: { columnPinning },
onColumnPinningChange: setColumnPinning,
```

### Key APIs

- `column.pin('left' | 'right' | false)` - Pin/unpin column
- `column.getIsPinned()` - Returns `'left'` | `'right'` | `false`
- `column.getCanPin()` - Whether column can be pinned
- `column.getStart('left')` / `column.getAfter('right')` - CSS position values for sticky
- `column.getIsLastColumn('left')` / `column.getIsFirstColumn('right')` - For box-shadow borders
- `column.getIndex(position)` - Column index within pinned/center group

### Sticky CSS Approach (Recommended)

Use `position: sticky` on pinned columns with calculated left/right offsets:

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
```

Apply to both headers and cells:

```tsx
// Header
<th style={{ ...getCommonPinningStyles(header.column) }}>

// Cell
<td style={{ ...getCommonPinningStyles(cell.column) }}>
```

Required CSS for the container:

```css
.table-container {
  overflow-x: auto;
}
table {
  border-collapse: separate;
  border-spacing: 0;
}
th, td {
  background-color: white; /* Required for sticky to work properly */
}
```

### Split Table Approach

Use separate table elements for left-pinned, center, and right-pinned columns:

```tsx
// Headers
table.getLeftHeaderGroups()    // Left-pinned column headers
table.getCenterHeaderGroups()  // Unpinned column headers
table.getRightHeaderGroups()   // Right-pinned column headers

// Cells
row.getLeftVisibleCells()      // Left-pinned cells
row.getCenterVisibleCells()    // Unpinned cells
row.getRightVisibleCells()     // Right-pinned cells
```

### Pin/Unpin Toggle UI

```tsx
{!header.isPlaceholder && header.column.getCanPin() && (
  <div className="flex gap-1">
    {header.column.getIsPinned() !== 'left' && (
      <button onClick={() => header.column.pin('left')}>{'<='}</button>
    )}
    {header.column.getIsPinned() && (
      <button onClick={() => header.column.pin(false)}>X</button>
    )}
    {header.column.getIsPinned() !== 'right' && (
      <button onClick={() => header.column.pin('right')}>{'=>'}</button>
    )}
  </div>
)}
```

---

## Column Sizing & Resizing

> API: https://tanstack.com/table/latest/docs/api/features/column-sizing

### Defaults

```ts
{ size: 150, minSize: 20, maxSize: Number.MAX_SAFE_INTEGER }
```

Override per-column or with `defaultColumn` table option:

```tsx
const table = useReactTable({
  defaultColumn: { size: 200, minSize: 50, maxSize: 500 },
})
```

### Resize Mode

- `"onEnd"` (default) - Size updates after drag ends (better React performance)
- `"onChange"` - Immediate size updates during drag

```tsx
const table = useReactTable({
  columnResizeMode: 'onChange',
  columnResizeDirection: 'ltr', // or 'rtl'
})
```

### Resize Handler

```tsx
<div
  onMouseDown={header.getResizeHandler()}
  onTouchStart={header.getResizeHandler()}
  className={`resizer ${header.column.getIsResizing() ? 'isResizing' : ''}`}
/>
```

### Performance Tips

1. Calculate all column widths once upfront, memoized
2. Memoize table body while resizing
3. Use CSS variables for column widths instead of inline styles:

```tsx
// Generate CSS variables from table state
const columnSizeVars = React.useMemo(() => {
  const headers = table.getFlatHeaders()
  const colSizes: Record<string, number> = {}
  for (const header of headers) {
    colSizes[`--header-${header.id}-size`] = header.getSize()
    colSizes[`--col-${header.column.id}-size`] = header.column.getSize()
  }
  return colSizes
}, [table.getState().columnSizingInfo, table.getState().columnSizing])

// Apply to table element
<table style={{ ...columnSizeVars, width: table.getTotalSize() }}>

// Use in CSS
th { width: calc(var(--header-firstName-size) * 1px); }
td { width: calc(var(--col-firstName-size) * 1px); }
```

---

## Expanding

> API: https://tanstack.com/table/latest/docs/api/features/expanding

### Sub-Rows

```tsx
const table = useReactTable({
  data, columns,
  getSubRows: row => row.children,
  getCoreRowModel: getCoreRowModel(),
  getExpandedRowModel: getExpandedRowModel(),
})
```

### Custom Expanded UI (Detail Panels)

```tsx
const table = useReactTable({
  getRowCanExpand: row => true,
  getCoreRowModel: getCoreRowModel(),
  getExpandedRowModel: getExpandedRowModel(),
})

// In render:
{table.getRowModel().rows.map(row => (
  <React.Fragment key={row.id}>
    <tr>
      {row.getVisibleCells().map(cell => (
        <td key={cell.id}>
          {flexRender(cell.column.columnDef.cell, cell.getContext())}
        </td>
      ))}
    </tr>
    {row.getIsExpanded() && (
      <tr>
        <td colSpan={row.getVisibleCells().length}>
          {/* Custom detail panel content */}
        </td>
      </tr>
    )}
  </React.Fragment>
))}
```

### State

```ts
type ExpandedState = true | Record<string, boolean>
// true = all expanded, Record = specific rows

const [expanded, setExpanded] = useState<ExpandedState>({})
state: { expanded },
onExpandedChange: setExpanded,
```

### Toggle UI

```tsx
<button onClick={row.getToggleExpandedHandler()}>
  {row.getIsExpanded() ? '▼' : '▶'}
</button>

// With depth-based indentation for sub-rows
<div style={{ paddingLeft: `${row.depth * 2}rem` }}>
  {row.getCanExpand() ? (
    <button onClick={row.getToggleExpandedHandler()}>
      {row.getIsExpanded() ? '▼' : '▶'}
    </button>
  ) : '●'}{' '}
  {row.getValue('name')}
</div>
```

### Expand All

```tsx
<button onClick={table.getToggleAllRowsExpandedHandler()}>
  {table.getIsAllRowsExpanded() ? 'Collapse All' : 'Expand All'}
</button>
```

### Options

- `paginateExpandedRows` (table) - Default true. Set false to keep expanded rows on parent's page.
- `filterFromLeafRows` / `maxLeafRowFilterDepth` - Control filtering of expanded rows.

---

## Grouping

> API: https://tanstack.com/table/latest/docs/api/features/grouping

### Setup

```tsx
import { getGroupedRowModel, getExpandedRowModel } from '@tanstack/react-table'

const table = useReactTable({
  data, columns,
  getCoreRowModel: getCoreRowModel(),
  getGroupedRowModel: getGroupedRowModel(),
  getExpandedRowModel: getExpandedRowModel(),
})
```

### State

```ts
const [grouping, setGrouping] = useState<GroupingState>([])
state: { grouping },
onGroupingChange: setGrouping,
```

### Aggregation

```ts
columnHelper.accessor('amount', {
  aggregationFn: 'sum', // built-in: sum, count, min, max, extent, mean, median, unique, uniqueCount
})
```

Custom:
```ts
aggregationFns: {
  myAgg: (columnId, leafRows, childRows) => { return aggregatedValue },
}
```

### Rendering Grouped Cells

```tsx
{row.getVisibleCells().map(cell => (
  <td key={cell.id}>
    {cell.getIsGrouped() ? (
      // Grouped cell: render expander + aggregated value
      <button onClick={row.getToggleExpandedHandler()}>
        {row.getIsExpanded() ? '▼' : '▶'}{' '}
        {flexRender(cell.column.columnDef.cell, cell.getContext())} ({row.subRows.length})
      </button>
    ) : cell.getIsAggregated() ? (
      // Aggregated cell: render aggregated value
      flexRender(cell.column.columnDef.aggregatedCell ?? cell.column.columnDef.cell, cell.getContext())
    ) : cell.getIsPlaceholder() ? null : (
      // Regular cell
      flexRender(cell.column.columnDef.cell, cell.getContext())
    )}
  </td>
))}
```

### Options

- `groupedColumnMode`: `'reorder'` | `'remove'` | `false`
- `manualGrouping`: `true` for server-side grouping

---

## Column Faceting

> API: https://tanstack.com/table/latest/docs/api/features/column-faceting

### Setup

```tsx
const table = useReactTable({
  getCoreRowModel: getCoreRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
  getFacetedRowModel: getFacetedRowModel(),
  getFacetedUniqueValues: getFacetedUniqueValues(),
  getFacetedMinMaxValues: getFacetedMinMaxValues(),
})
```

### Usage

```ts
// Unique values for autocomplete/select dropdowns
const suggestions = Array.from(column.getFacetedUniqueValues().keys()).sort().slice(0, 5000)

// Min/max for range slider/input
const [min, max] = column.getFacetedMinMaxValues() ?? [0, 1]
```

---

## Row Pinning

> API: https://tanstack.com/table/latest/docs/api/features/row-pinning

### State

```ts
const [rowPinning, setRowPinning] = useState<RowPinningState>({
  top: [],    // row IDs pinned to top
  bottom: [], // row IDs pinned to bottom
})

state: { rowPinning },
onRowPinningChange: setRowPinning,
```

### Key APIs

- `row.pin('top' | 'bottom' | false)` - Pin/unpin a row
- `row.getIsPinned()` - Returns `'top'` | `'bottom'` | `false`
- `row.getCanPin()` - Whether the row can be pinned
- `table.getTopRows()` - Rows pinned to top
- `table.getCenterRows()` - Unpinned rows only
- `table.getBottomRows()` - Rows pinned to bottom

### Options

- `keepPinnedRows` (table) - Keep pinned rows visible even when they would be filtered out (default: true)
- `enableRowPinning` (table) - Boolean or `(row) => boolean`

### Rendering Pinned Rows

```tsx
<tbody>
  {/* Top-pinned rows */}
  {table.getTopRows().map(row => (
    <tr key={row.id} className="bg-blue-50 sticky top-0">
      {row.getVisibleCells().map(cell => (
        <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
      ))}
    </tr>
  ))}
  {/* Center (unpinned) rows */}
  {table.getCenterRows().map(row => (
    <tr key={row.id}>
      {row.getVisibleCells().map(cell => (
        <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
      ))}
    </tr>
  ))}
  {/* Bottom-pinned rows */}
  {table.getBottomRows().map(row => (
    <tr key={row.id} className="bg-blue-50 sticky bottom-0">
      {row.getVisibleCells().map(cell => (
        <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
      ))}
    </tr>
  ))}
</tbody>
```

---

## Virtualization

TanStack Table does not include virtualization but integrates with [TanStack Virtual](https://tanstack.com/virtual/latest).

### Row Virtualization

```tsx
import { useVirtualizer } from '@tanstack/react-virtual'

const { rows } = table.getRowModel()
const parentRef = React.useRef<HTMLDivElement>(null)

const virtualizer = useVirtualizer({
  count: rows.length,
  getScrollElement: () => parentRef.current,
  estimateSize: () => 35, // estimated row height in px
  overscan: 5,
})

return (
  <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
    <table>
      <thead>{/* ... standard header rendering ... */}</thead>
      <tbody style={{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }}>
        {virtualizer.getVirtualItems().map(virtualRow => {
          const row = rows[virtualRow.index]
          return (
            <tr key={row.id}
              style={{
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
                position: 'absolute',
                width: '100%',
              }}>
              {row.getVisibleCells().map(cell => (
                <td key={cell.id}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          )
        })}
      </tbody>
    </table>
  </div>
)
```

### Column Virtualization

```tsx
const columnVirtualizer = useVirtualizer({
  count: visibleColumns.length,
  estimateSize: index => visibleColumns[index].getSize(),
  getScrollElement: () => parentRef.current,
  horizontal: true,
  overscan: 3,
})
```

### Virtualized + Infinite Scrolling (with TanStack Query)

Combine `useInfiniteQuery` with virtualization for infinite scroll tables. Use `fetchNextPage` when the virtualizer scrolls near the end of loaded data.

### Examples

- [virtualized-rows](https://tanstack.com/table/latest/docs/framework/react/examples/virtualized-rows)
- [virtualized-columns](https://tanstack.com/table/latest/docs/framework/react/examples/virtualized-columns)
- [virtualized-infinite-scrolling](https://tanstack.com/table/latest/docs/framework/react/examples/virtualized-infinite-scrolling)

---

## Custom Features

> Docs: https://tanstack.com/table/latest/docs/guide/custom-features

Use `_features` table option to add custom features:

```ts
import { makeStateUpdater, type TableFeature, type RowData } from '@tanstack/react-table'

// 1. Define types
type DensityState = 'sm' | 'md' | 'lg'
interface DensityTableState { density: DensityState }
interface DensityOptions { enableDensity?: boolean; onDensityChange?: OnChangeFn<DensityState> }
interface DensityInstance {
  setDensity: (updater: Updater<DensityState>) => void
  toggleDensity: (value?: DensityState) => void
}

// 2. Extend TanStack Table types via declaration merging
declare module '@tanstack/react-table' {
  interface TableState extends DensityTableState {}
  interface TableOptionsResolved<TData extends RowData> extends DensityOptions {}
  interface Table<TData extends RowData> extends DensityInstance {}
}

// 3. Implement the feature
export const DensityFeature: TableFeature<any> = {
  getInitialState: (state) => ({ density: 'md' as DensityState, ...state }),
  getDefaultOptions: <TData extends RowData>(table: Table<TData>) => ({
    enableDensity: true,
    onDensityChange: makeStateUpdater('density', table),
  }),
  createTable: <TData extends RowData>(table: Table<TData>) => {
    table.setDensity = (updater) => {
      const safeUpdater: Updater<DensityState> = (old) => {
        return typeof updater === 'function' ? updater(old) : updater
      }
      table.options.onDensityChange?.(safeUpdater)
    }
    table.toggleDensity = (value) => {
      table.setDensity((old) => {
        if (value) return value
        return old === 'lg' ? 'md' : old === 'md' ? 'sm' : 'lg'
      })
    }
  },
}

// 4. Pass to table
const table = useReactTable({
  _features: [DensityFeature],
  state: { density },
  onDensityChange: setDensity,
  // ...
})
```
