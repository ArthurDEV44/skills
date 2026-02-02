# coss ui - Migration from Radix / shadcn/ui

coss ui is intentionally similar to shadcn/ui but uses Base UI internally instead of Radix.

## Key Pattern: asChild to render

The most significant change. Replace Radix `asChild` with Base UI `render` prop:

```tsx
// Before (Radix/shadcn)
<DropdownMenuTrigger asChild>
  <Button>Edit</Button>
</DropdownMenuTrigger>

// After (coss ui)
<MenuTrigger render={<Button />}>Edit</MenuTrigger>
```

## Component Renaming

| Radix/shadcn | coss ui (preferred) | Legacy alias (still works) |
|---|---|---|
| `*Content` | `*Popup` or `*Panel` | `*Content` |
| `DropdownMenu*` | `Menu*` | `DropdownMenu*` |
| `HoverCard*` | `PreviewCard*` | `HoverCard*` |
| `ButtonGroup*` | `Group*` | `ButtonGroup*` |
| `TabsTrigger` | `TabsTab` | `TabsTrigger` |
| `TabsContent` | `TabsPanel` | `TabsContent` |
| `RadioGroupItem` | `Radio` | `RadioGroupItem` |
| `ToggleGroupItem` | `Toggle` | `ToggleGroupItem` |

## Import Changes

| Before | After |
|---|---|
| `@/components/ui/dropdown-menu` | `@/components/ui/menu` |
| `@/components/ui/hover-card` | `@/components/ui/preview-card` |

## Per-Component Migration Notes

### Accordion
- `type="multiple"` becomes `multiple={true}`; remove `type="single"` and `collapsible`
- Use arrays for `defaultValue`
- Prefer `AccordionPanel` over `AccordionContent`

### Alert
- New variants: info, success, warning, error (require CSS variables)

### AlertDialog
- Replace `asChild` with `render`; use `AlertDialogClose` instead of separate Action/Cancel
- Prefer `AlertDialogPopup`; wrap content in `AlertDialogPanel`

### Badge
- New sizes: sm, default, lg (lg matches original shadcn/ui)
- New variants: info, success, warning, error

### Button
- More compact sizing; new sizes: xs (24px), xl (40px), icon-sm, icon-lg
- New variant: `destructive-outline`

### Card
- Use `CardPanel` instead of `CardContent`

### Command
- No cmdk dependency; built with Base UI Autocomplete + Dialog
- Data-driven approach using `items` array
- `CommandGroup` uses `<CommandGroupLabel>` child instead of `heading` prop

### Dialog
- Replace `asChild` with `render`; prefer `DialogPopup`
- Use `DialogPanel` for scrollable content between header/footer

### Group
- `GroupSeparator` is **always required** between controls

### Input / Select / Textarea
- More compact: sm (28px), default (32px), lg (36px). Use `lg` for shadcn/ui parity.

### Input Group
- No `InputGroupButton`; use regular `Button` inside `InputGroupAddon`

### Menu
- `onSelect` becomes `onClick`
- `MenuGroupLabel` instead of `DropdownMenuLabel`
- `MenuPopup` instead of `DropdownMenuContent`

### Popover
- Prefer `PopoverPopup`; adds `PopoverTitle`, `PopoverDescription`, `PopoverClose`

### Select
- **Must pass `items` prop** (array or record) on `Select` root
- Remove `placeholder` from `Select`; prefer `SelectPopup` over `SelectContent`

### Sheet
- Prefer `SheetPopup`; use `SheetPanel` for scrollable content

### Slider
- Always pass values as arrays: `value={[50]}`
- `onValueChange` receives array of numbers

### Tabs
- Use `TabsTab` instead of `TabsTrigger`; `TabsList` adds `variant` prop (default|underline)

### Toast
- Replace `<Toaster />` with `<ToastProvider>` wrapper
- Uses `toastManager.add()` with `title`, `description`, `type`, `actionProps`

### ToggleGroup
- `type="multiple"` becomes `multiple` prop; remove `type="single"`
- Always use arrays for `defaultValue`

### Tooltip
- Prefer `TooltipPopup` over `TooltipContent`

## Documentation
- Full migration guide: https://coss.com/ui/docs/radix-migration
- Base UI docs: https://base-ui.com
