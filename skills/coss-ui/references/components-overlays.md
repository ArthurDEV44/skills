# coss ui - Overlay Components Reference

## Table of Contents
- [Dialog](#dialog)
- [AlertDialog](#alert-dialog)
- [Sheet](#sheet)
- [Popover](#popover)
- [Tooltip](#tooltip)
- [PreviewCard](#preview-card)
- [Menu](#menu)
- [Command](#command)
- [Toast](#toast)

---

## Dialog

CLI: `pnpm dlx shadcn@latest add @coss/dialog`

```tsx
import {
  Dialog, DialogClose, DialogDescription, DialogFooter, DialogHeader,
  DialogPanel, DialogPopup, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog"
```

**DialogPopup props:**
| Prop | Type | Default |
|------|------|---------|
| `showCloseButton` | `boolean` | `true` |
| `bottomStickOnMobile` | `boolean` | `true` |

**DialogFooter props:** `variant` ("default" | "bare")
**DialogPanel props:** `scrollFade` (boolean, default true)

`DialogContent` is a legacy alias for `DialogPopup`.

```tsx
<Dialog>
  <DialogTrigger render={<Button variant="outline" />}>Open</DialogTrigger>
  <DialogPopup className="sm:max-w-sm">
    <DialogHeader>
      <DialogTitle>Edit profile</DialogTitle>
      <DialogDescription>Make changes here.</DialogDescription>
    </DialogHeader>
    <DialogPanel className="grid gap-4">
      <Field>
        <FieldLabel>Name</FieldLabel>
        <Input defaultValue="Margaret Welsh" type="text" />
      </Field>
    </DialogPanel>
    <DialogFooter>
      <DialogClose render={<Button variant="ghost" />}>Cancel</DialogClose>
      <Button type="submit">Save</Button>
    </DialogFooter>
  </DialogPopup>
</Dialog>
```

Supports: nested dialogs, close confirmation (with AlertDialog), controlled state, bare footer variant.

Docs: https://coss.com/ui/docs/components/dialog

---

## Alert Dialog

CLI: `pnpm dlx shadcn@latest add @coss/alert-dialog`

```tsx
import {
  AlertDialog, AlertDialogClose, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogPopup, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
```

**AlertDialogPopup props:** `bottomStickOnMobile` (boolean, default true)
**AlertDialogFooter props:** `variant` ("default" | "bare")

`AlertDialogContent` is a legacy alias for `AlertDialogPopup`.

```tsx
<AlertDialog>
  <AlertDialogTrigger render={<Button variant="destructive-outline" />}>
    Delete Account
  </AlertDialogTrigger>
  <AlertDialogPopup>
    <AlertDialogHeader>
      <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
      <AlertDialogDescription>
        This action cannot be undone.
      </AlertDialogDescription>
    </AlertDialogHeader>
    <AlertDialogFooter>
      <AlertDialogClose render={<Button variant="ghost" />}>Cancel</AlertDialogClose>
      <AlertDialogClose render={<Button variant="destructive" />}>Delete</AlertDialogClose>
    </AlertDialogFooter>
  </AlertDialogPopup>
</AlertDialog>
```

Docs: https://coss.com/ui/docs/components/alert-dialog

---

## Sheet

CLI: `pnpm dlx shadcn@latest add @coss/sheet`

```tsx
import {
  Sheet, SheetClose, SheetDescription, SheetFooter, SheetHeader,
  SheetPanel, SheetPopup, SheetTitle, SheetTrigger,
} from "@/components/ui/sheet"
```

**SheetPopup props:**
| Prop | Type | Default |
|------|------|---------|
| `side` | `"right" \| "left" \| "top" \| "bottom"` | `"right"` |
| `variant` | `"default" \| "inset"` | `"default"` |
| `showCloseButton` | `boolean` | `true` |

**SheetFooter props:** `variant` ("default" | "bare")
**SheetPanel props:** `scrollFade` (boolean, default true)

`SheetContent` is a legacy alias for `SheetPopup`.

```tsx
<Sheet>
  <SheetTrigger render={<Button variant="outline" />}>Open Sheet</SheetTrigger>
  <SheetPopup side="right">
    <SheetHeader>
      <SheetTitle>Edit profile</SheetTitle>
      <SheetDescription>Make changes here.</SheetDescription>
    </SheetHeader>
    <SheetPanel className="grid gap-4">
      {/* Content */}
    </SheetPanel>
    <SheetFooter>
      <SheetClose render={<Button variant="ghost" />}>Cancel</SheetClose>
      <Button type="submit">Save</Button>
    </SheetFooter>
  </SheetPopup>
</Sheet>
```

Docs: https://coss.com/ui/docs/components/sheet

---

## Popover

CLI: `pnpm dlx shadcn@latest add @coss/popover`

```tsx
import {
  Popover, PopoverClose, PopoverDescription,
  PopoverPopup, PopoverTitle, PopoverTrigger,
} from "@/components/ui/popover"
```

**PopoverPopup props:**
| Prop | Type | Default |
|------|------|---------|
| `tooltipStyle` | `boolean` | `false` |
| `side` | `"top" \| "bottom" \| "left" \| "right"` | `"bottom"` |
| `align` | `"start" \| "center" \| "end"` | `"center"` |
| `sideOffset` | `number` | `4` |

`PopoverContent` is a legacy alias for `PopoverPopup`.

```tsx
<Popover>
  <PopoverTrigger>Open Popover</PopoverTrigger>
  <PopoverPopup>
    <PopoverTitle>Title</PopoverTitle>
    <PopoverDescription>Description</PopoverDescription>
    <PopoverClose>Close</PopoverClose>
  </PopoverPopup>
</Popover>
```

Use `tooltipStyle` for info icon scenarios (smaller padding, arrow).

Animated popovers across multiple triggers: use `PopoverCreateHandle`.

Docs: https://coss.com/ui/docs/components/popover

---

## Tooltip

CLI: `pnpm dlx shadcn@latest add @coss/tooltip`

```tsx
import {
  Tooltip, TooltipCreateHandle, TooltipPopup, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip"
```

**TooltipPopup props:**
| Prop | Type | Default |
|------|------|---------|
| `side` | `"top" \| "bottom" \| "left" \| "right"` | `"top"` |
| `align` | `"start" \| "center" \| "end"` | `"center"` |
| `sideOffset` | `number` | `4` |

`TooltipContent` is a legacy alias for `TooltipPopup`.

```tsx
<Tooltip>
  <TooltipTrigger render={<Button variant="outline" />}>Hover me</TooltipTrigger>
  <TooltipPopup>Helpful hint</TooltipPopup>
</Tooltip>
```

Grouped (instant appearance after first): wrap in `TooltipProvider`.
Animated across triggers: use `TooltipCreateHandle`.

Docs: https://coss.com/ui/docs/components/tooltip

---

## Preview Card

CLI: `pnpm dlx shadcn@latest add @coss/preview-card`

```tsx
import { PreviewCard, PreviewCardPopup, PreviewCardTrigger } from "@/components/ui/preview-card"
```

`HoverCardContent` is a legacy alias for `PreviewCardPopup`.

**PreviewCardPopup props:** `align` ("start"|"center"|"end"), `sideOffset` (number, default 4)

```tsx
<PreviewCard>
  <PreviewCardTrigger render={<Button variant="ghost" />}>coss.com/ui</PreviewCardTrigger>
  <PreviewCardPopup>
    <h4 className="font-medium text-sm">coss.com/ui</h4>
    <p className="text-muted-foreground text-sm">Beautiful components.</p>
  </PreviewCardPopup>
</PreviewCard>
```

Docs: https://coss.com/ui/docs/components/preview-card

---

## Menu

CLI: `pnpm dlx shadcn@latest add @coss/menu`

```tsx
import {
  Menu, MenuCheckboxItem, MenuGroup, MenuGroupLabel, MenuItem, MenuPopup,
  MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuShortcut,
  MenuSub, MenuSubPopup, MenuSubTrigger, MenuTrigger,
} from "@/components/ui/menu"
```

`DropdownMenu*` names are legacy aliases for `Menu*`.

**MenuPopup props:** `side`, `align`, `sideOffset` (default 4), `alignOffset`
**MenuItem props:** `inset` (boolean), `variant` ("default"|"destructive"), `closeOnClick`
**MenuCheckboxItem props:** `variant` ("default"|"switch"), `defaultChecked`

```tsx
<Menu>
  <MenuTrigger render={<Button variant="outline" />}>Open menu</MenuTrigger>
  <MenuPopup>
    <MenuItem>Profile</MenuItem>
    <MenuItem>Settings</MenuItem>
    <MenuSeparator />
    <MenuItem variant="destructive">Delete</MenuItem>
  </MenuPopup>
</Menu>
```

Submenus:
```tsx
<MenuSub>
  <MenuSubTrigger>More</MenuSubTrigger>
  <MenuSubPopup>
    <MenuItem>Sub item</MenuItem>
  </MenuSubPopup>
</MenuSub>
```

Checkbox items, radio groups, grouped items, shortcuts, links (via `render` prop), open on hover.

Docs: https://coss.com/ui/docs/components/menu

---

## Command

CLI: `pnpm dlx shadcn@latest add @coss/command`

```tsx
import {
  Command, CommandCollection, CommandDialog, CommandDialogPopup, CommandDialogTrigger,
  CommandEmpty, CommandFooter, CommandGroup, CommandGroupLabel,
  CommandInput, CommandItem, CommandList, CommandPanel, CommandSeparator, CommandShortcut,
} from "@/components/ui/command"
```

**Command props:** `items`, `open`, `autoHighlight` (default "always"), `keepHighlight` (default true)

No cmdk dependency. Built with Base UI Autocomplete and Dialog.

```tsx
<CommandDialog>
  <CommandDialogTrigger render={<Button variant="outline" />}>
    Open Command Palette
  </CommandDialogTrigger>
  <CommandDialogPopup>
    <Command items={items}>
      <CommandInput placeholder="Search..." />
      <CommandEmpty>No results found.</CommandEmpty>
      <CommandList>
        {(item) => <CommandItem key={item.value} value={item.value}>{item.label}</CommandItem>}
      </CommandList>
    </Command>
  </CommandDialogPopup>
</CommandDialog>
```

Grouped:
```tsx
<CommandList>
  {(group, index) => (
    <Fragment key={group.value}>
      <CommandGroup items={group.items}>
        <CommandGroupLabel>{group.value}</CommandGroupLabel>
        <CommandCollection>
          {(item) => <CommandItem key={item.value} value={item.value}>{item.label}</CommandItem>}
        </CommandCollection>
      </CommandGroup>
      {index < groupedItems.length - 1 && <CommandSeparator />}
    </Fragment>
  )}
</CommandList>
```

Standalone (without dialog): `<Command open items={items}>...`

Keyboard shortcut: listen for Cmd+K / Ctrl+K, toggle `open` state.

Docs: https://coss.com/ui/docs/components/command

---

## Toast

CLI: `pnpm dlx shadcn@latest add @coss/toast`

```tsx
import {
  ToastProvider, AnchoredToastProvider, toastManager, anchoredToastManager,
  ToastViewport, Toast, ToastTitle, ToastDescription, ToastAction, ToastClose,
} from "@/components/ui/toast"
```

**ToastProvider props:** `position` ("top-left"|"top-center"|"top-right"|"bottom-left"|"bottom-center"|"bottom-right", default "bottom-right")

Wrap app in `<ToastProvider>` (replaces `<Toaster />`).

```tsx
// Basic
toastManager.add({
  title: "Event created",
  description: "Monday, January 3rd at 6:00pm",
})

// With type
toastManager.add({ title: "Success!", type: "success" })
// Types: "success", "error", "info", "warning", "loading"

// With action
const id = toastManager.add({
  title: "Action performed",
  description: "You can undo this.",
  actionProps: { children: "Undo", onClick: () => toastManager.close(id) },
  timeout: 1000000,
})

// Promise-based
toastManager.promise(fetchData(), {
  loading: { title: "Loading..." },
  success: (data) => ({ title: "Done!", description: data }),
  error: () => ({ title: "Error" }),
})

// Anchored (positioned relative to element)
anchoredToastManager.add({
  title: "Copied!",
  positionerProps: { anchor: buttonRef.current },
  data: { tooltipStyle: true },
})
```

Docs: https://coss.com/ui/docs/components/toast
