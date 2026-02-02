# coss ui - Layout & Display Components Reference

## Table of Contents
- [Accordion](#accordion)
- [Alert](#alert)
- [Avatar](#avatar)
- [Badge](#badge)
- [Breadcrumb](#breadcrumb)
- [Button](#button)
- [Card](#card)
- [Collapsible](#collapsible)
- [Empty](#empty)
- [Frame](#frame)
- [Group](#group)
- [InputGroup](#input-group)
- [Kbd](#kbd)
- [Label](#label)
- [Meter](#meter)
- [Pagination](#pagination)
- [Progress](#progress)
- [ScrollArea](#scroll-area)
- [Separator](#separator)
- [Skeleton](#skeleton)
- [Spinner](#spinner)
- [Table](#table)
- [Tabs](#tabs)
- [Toggle](#toggle)
- [ToggleGroup](#toggle-group)
- [Toolbar](#toolbar)

---

## Accordion

CLI: `pnpm dlx shadcn@latest add @coss/accordion`

```tsx
import { Accordion, AccordionItem, AccordionPanel, AccordionTrigger } from "@/components/ui/accordion"
```

`AccordionContent` is a legacy alias for `AccordionPanel`.

**Accordion props:** `defaultValue` (string|string[]), `value`, `multiple` (boolean), `onValueChange`

```tsx
<Accordion>
  <AccordionItem value="item-1">
    <AccordionTrigger>Is it accessible?</AccordionTrigger>
    <AccordionPanel>Yes. WAI-ARIA compliant.</AccordionPanel>
  </AccordionItem>
</Accordion>

// Multiple open
<Accordion multiple>...</Accordion>

// Controlled
<Accordion onValueChange={setValue} value={value}>...</Accordion>
```

Docs: https://coss.com/ui/docs/components/accordion

---

## Alert

CLI: `pnpm dlx shadcn@latest add @coss/alert`

```tsx
import { Alert, AlertDescription, AlertTitle, AlertAction } from "@/components/ui/alert"
```

**Alert props:** `variant` ("default"|"error"|"info"|"success"|"warning")

```tsx
<Alert variant="info">
  <InfoIcon />
  <AlertTitle>Heads up!</AlertTitle>
  <AlertDescription>Description here.</AlertDescription>
  <AlertAction>
    <Button size="xs" variant="ghost">Dismiss</Button>
    <Button size="xs">Ok</Button>
  </AlertAction>
</Alert>
```

Requires CSS variables for color variants (installed with `@coss/colors-neutral`).

Docs: https://coss.com/ui/docs/components/alert

---

## Avatar

CLI: `pnpm dlx shadcn@latest add @coss/avatar`

```tsx
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
```

Default size: 32px (size-8). Customize with className.

```tsx
<Avatar>
  <AvatarImage src="/avatar.png" alt="User" />
  <AvatarFallback>JD</AvatarFallback>
</Avatar>

// Custom size and radius
<Avatar className="size-12 rounded-md">
  <AvatarImage src="/avatar.png" alt="User" />
  <AvatarFallback>JD</AvatarFallback>
</Avatar>

// Grouped
<div className="-space-x-[0.6rem] flex">
  <Avatar className="ring-2 ring-background">...</Avatar>
  <Avatar className="ring-2 ring-background">...</Avatar>
</div>
```

Docs: https://coss.com/ui/docs/components/avatar

---

## Badge

CLI: `pnpm dlx shadcn@latest add @coss/badge`

```tsx
import { Badge } from "@/components/ui/badge"
```

**Props:**
| Prop | Type | Default |
|------|------|---------|
| `variant` | `"default"\|"destructive"\|"error"\|"info"\|"outline"\|"secondary"\|"success"\|"warning"` | `"default"` |
| `size` | `"default"\|"sm"\|"lg"` | `"default"` |
| `render` | `React.ReactElement` | -- |

```tsx
<Badge>Default</Badge>
<Badge variant="info">Info</Badge>
<Badge variant="success">Success</Badge>
<Badge variant="warning">Warning</Badge>
<Badge variant="error">Error</Badge>
<Badge variant="outline"><CheckIcon /> Verified</Badge>
<Badge render={<Link href="/" />}>Link Badge</Badge>
<Badge className="rounded-full">7</Badge>
```

Docs: https://coss.com/ui/docs/components/badge

---

## Breadcrumb

CLI: `pnpm dlx shadcn@latest add @coss/breadcrumb`

```tsx
import {
  Breadcrumb, BreadcrumbEllipsis, BreadcrumbItem, BreadcrumbLink,
  BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
```

```tsx
<Breadcrumb>
  <BreadcrumbList>
    <BreadcrumbItem>
      <BreadcrumbLink href="/">Home</BreadcrumbLink>
    </BreadcrumbItem>
    <BreadcrumbSeparator />
    <BreadcrumbItem>
      <BreadcrumbLink href="/components">Components</BreadcrumbLink>
    </BreadcrumbItem>
    <BreadcrumbSeparator />
    <BreadcrumbItem>
      <BreadcrumbPage>Current Page</BreadcrumbPage>
    </BreadcrumbItem>
  </BreadcrumbList>
</Breadcrumb>
```

Custom separator: `<BreadcrumbSeparator> / </BreadcrumbSeparator>`
With Next.js Link: `<BreadcrumbLink render={<Link href="/" />}>Home</BreadcrumbLink>`

Docs: https://coss.com/ui/docs/components/breadcrumb

---

## Button

CLI: `pnpm dlx shadcn@latest add @coss/button`

```tsx
import { Button } from "@/components/ui/button"
```

**Props:**
| Prop | Type | Default |
|------|------|---------|
| `variant` | `"default"\|"destructive"\|"destructive-outline"\|"ghost"\|"link"\|"outline"\|"secondary"` | `"default"` |
| `size` | `"default"\|"xs"\|"sm"\|"lg"\|"xl"\|"icon"\|"icon-sm"\|"icon-lg"\|"icon-xl"\|"icon-xs"` | `"default"` |
| `render` | `React.ReactElement` | -- |

```tsx
<Button>Default</Button>
<Button variant="outline">Outline</Button>
<Button variant="destructive">Delete</Button>
<Button variant="destructive-outline">Destructive Outline</Button>
<Button size="xs">Tiny</Button>
<Button size="xl">Extra Large</Button>
<Button render={<Link href="/login" />}>Login</Button>
<Button><DownloadIcon /> Download</Button>
<Button disabled><Spinner /> Loading...</Button>
```

Docs: https://coss.com/ui/docs/components/button

---

## Card

CLI: `pnpm dlx shadcn@latest add @coss/card`

```tsx
import { Card, CardDescription, CardFooter, CardHeader, CardPanel, CardTitle } from "@/components/ui/card"
```

`CardContent` is a legacy alias for `CardPanel`. Also has `CardAction` for header actions.

```tsx
<Card>
  <CardHeader>
    <CardTitle>Create project</CardTitle>
    <CardDescription>Deploy in one-click.</CardDescription>
  </CardHeader>
  <CardPanel>
    {/* Content */}
  </CardPanel>
  <CardFooter>
    <Button>Deploy</Button>
  </CardFooter>
</Card>
```

Docs: https://coss.com/ui/docs/components/card

---

## Collapsible

CLI: `pnpm dlx shadcn@latest add @coss/collapsible`

```tsx
import { Collapsible, CollapsiblePanel, CollapsibleTrigger } from "@/components/ui/collapsible"
```

`CollapsibleContent` is a legacy alias for `CollapsiblePanel`.

```tsx
<Collapsible>
  <CollapsibleTrigger>Show details</CollapsibleTrigger>
  <CollapsiblePanel>Hidden content here.</CollapsiblePanel>
</Collapsible>
```

Docs: https://coss.com/ui/docs/components/collapsible

---

## Empty

CLI: `pnpm dlx shadcn@latest add @coss/empty`

```tsx
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
```

**EmptyMedia props:** `variant` ("default"|"icon")

```tsx
<Empty>
  <EmptyHeader>
    <EmptyMedia variant="icon"><RouteIcon /></EmptyMedia>
    <EmptyTitle>No upcoming meetings</EmptyTitle>
    <EmptyDescription>Create a meeting to get started.</EmptyDescription>
  </EmptyHeader>
  <EmptyContent>
    <Button size="sm">Create meeting</Button>
  </EmptyContent>
</Empty>
```

Docs: https://coss.com/ui/docs/components/empty

---

## Frame

CLI: `pnpm dlx shadcn@latest add @coss/frame`

```tsx
import { Frame, FrameDescription, FrameFooter, FrameHeader, FramePanel, FrameTitle } from "@/components/ui/frame"
```

```tsx
<Frame>
  <FrameHeader>
    <FrameTitle>Section header</FrameTitle>
    <FrameDescription>Description</FrameDescription>
  </FrameHeader>
  <FramePanel>Content</FramePanel>
  <FrameFooter>Footer</FrameFooter>
</Frame>
```

Multiple FramePanels render with spacing between them.

Docs: https://coss.com/ui/docs/components/frame

---

## Group

CLI: `pnpm dlx shadcn@latest add @coss/group`

```tsx
import { Button } from "@/components/ui/button"
import { Group, GroupSeparator } from "@/components/ui/group"
```

`ButtonGroup*` names are legacy aliases for `Group*`. Also has `GroupText`.

**Group props:** `orientation` ("horizontal"|"vertical")
**GroupSeparator** is **always required** between controls.

```tsx
<Group>
  <Button>Button</Button>
  <GroupSeparator />
  <Button>Button</Button>
</Group>
```

For action-performing controls (not state-toggling; use ToggleGroup for that).

Docs: https://coss.com/ui/docs/components/group

---

## Input Group

CLI: `pnpm dlx shadcn@latest add @coss/input-group`

```tsx
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText, InputGroupTextarea } from "@/components/ui/input-group"
```

**InputGroupAddon props:** `align` ("inline-start"|"inline-end"|"block-start"|"block-end", default "inline-start")

Place addons after input in DOM for proper focus. No InputGroupButton; use regular Button in InputGroupAddon.

```tsx
<InputGroup>
  <InputGroupInput placeholder="Search..." type="search" />
  <InputGroupAddon align="inline-end">
    <Kbd>⌘K</Kbd>
  </InputGroupAddon>
</InputGroup>

<InputGroup>
  <InputGroupAddon>
    <InputGroupText>https://</InputGroupText>
  </InputGroupAddon>
  <InputGroupInput placeholder="example.com" type="url" />
</InputGroup>
```

Docs: https://coss.com/ui/docs/components/input-group

---

## Kbd

CLI: `pnpm dlx shadcn@latest add @coss/kbd`

```tsx
import { Kbd, KbdGroup } from "@/components/ui/kbd"
```

```tsx
<Kbd>K</Kbd>
<KbdGroup><Kbd>⌘</Kbd><Kbd>K</Kbd></KbdGroup>
```

Docs: https://coss.com/ui/docs/components/kbd

---

## Label

CLI: `pnpm dlx shadcn@latest add @coss/label`

```tsx
import { Label } from "@/components/ui/label"
```

**Props:** `render` (React.ReactElement)

```tsx
<Label htmlFor={id}>Email</Label>
```

For form validation, prefer `FieldLabel` from the Field component.

Docs: https://coss.com/ui/docs/components/label

---

## Meter

CLI: `pnpm dlx shadcn@latest add @coss/meter`

```tsx
import { Meter, MeterIndicator, MeterLabel, MeterTrack, MeterValue } from "@/components/ui/meter"
```

**Props:** `value`, `min` (default 0), `max` (default 100)

```tsx
<Meter value={75}>
  <div className="flex items-center justify-between gap-2">
    <MeterLabel>Storage usage</MeterLabel>
    <MeterValue />
  </div>
  <MeterTrack><MeterIndicator /></MeterTrack>
</Meter>

// Minimal
<Meter value={50} />

// Custom format
<MeterValue>{(_formatted, value) => `${value} / 5`}</MeterValue>
```

Docs: https://coss.com/ui/docs/components/meter

---

## Pagination

CLI: `pnpm dlx shadcn@latest add @coss/pagination`

```tsx
import {
  Pagination, PaginationContent, PaginationEllipsis, PaginationItem,
  PaginationLink, PaginationNext, PaginationPrevious,
} from "@/components/ui/pagination"
```

**PaginationLink props:** `isActive` (boolean), `size` ("default"|"icon"), `render`

```tsx
<Pagination>
  <PaginationContent>
    <PaginationItem><PaginationPrevious href="#" /></PaginationItem>
    <PaginationItem><PaginationLink href="#">1</PaginationLink></PaginationItem>
    <PaginationItem><PaginationLink href="#" isActive>2</PaginationLink></PaginationItem>
    <PaginationItem><PaginationLink href="#">3</PaginationLink></PaginationItem>
    <PaginationItem><PaginationEllipsis /></PaginationItem>
    <PaginationItem><PaginationNext href="#" /></PaginationItem>
  </PaginationContent>
</Pagination>
```

Docs: https://coss.com/ui/docs/components/pagination

---

## Progress

CLI: `pnpm dlx shadcn@latest add @coss/progress`

```tsx
import { Progress, ProgressLabel, ProgressValue, ProgressTrack, ProgressIndicator } from "@/components/ui/progress"
```

**Props:** `value` (number), `max` (default 100)

When rendering children, must include both ProgressTrack and ProgressIndicator.

```tsx
// Simple
<Progress value={40} />

// With label
<Progress value={60}>
  <div className="flex items-center justify-between gap-2">
    <ProgressLabel>Export data</ProgressLabel>
    <ProgressValue />
  </div>
  <ProgressTrack><ProgressIndicator /></ProgressTrack>
</Progress>
```

Docs: https://coss.com/ui/docs/components/progress

---

## Scroll Area

CLI: `pnpm dlx shadcn@latest add @coss/scroll-area`

```tsx
import { ScrollArea } from "@/components/ui/scroll-area"
```

**Props:** `scrollFade` (boolean, default false), `scrollbarGutter` (boolean, default false)

```tsx
<ScrollArea className="h-64 rounded-md border">
  <div className="p-4">{/* Scrollable content */}</div>
</ScrollArea>

<ScrollArea scrollFade className="h-64">...</ScrollArea>
```

Docs: https://coss.com/ui/docs/components/scroll-area

---

## Separator

CLI: `pnpm dlx shadcn@latest add @coss/separator`

```tsx
import { Separator } from "@/components/ui/separator"
```

**Props:** `orientation` ("horizontal"|"vertical")

```tsx
<Separator />
<Separator orientation="vertical" />
```

Docs: https://coss.com/ui/docs/components/separator

---

## Skeleton

CLI: `pnpm dlx shadcn@latest add @coss/skeleton`

```tsx
import { Skeleton } from "@/components/ui/skeleton"
```

Simple div with animated pulse. Style via className.

```tsx
<Skeleton className="size-10 rounded-full" />
<Skeleton className="h-4 w-full" />
```

Docs: https://coss.com/ui/docs/components/skeleton

---

## Spinner

CLI: `pnpm dlx shadcn@latest add @coss/spinner`

```tsx
import { Spinner } from "@/components/ui/spinner"
```

```tsx
<Spinner />
<Button disabled><Spinner /> Loading...</Button>
```

Docs: https://coss.com/ui/docs/components/spinner

---

## Table

CLI: `pnpm dlx shadcn@latest add @coss/table`

```tsx
import {
  Table, TableBody, TableCaption, TableCell, TableFooter, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
```

```tsx
<Table>
  <TableCaption>Caption</TableCaption>
  <TableHeader>
    <TableRow>
      <TableHead>Header</TableHead>
      <TableHead>Header</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    <TableRow>
      <TableCell>Cell</TableCell>
      <TableCell>Cell</TableCell>
    </TableRow>
  </TableBody>
</Table>
```

Framed variant: wrap in `<Frame>`.

Docs: https://coss.com/ui/docs/components/table

---

## Tabs

CLI: `pnpm dlx shadcn@latest add @coss/tabs`

```tsx
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs"
```

`TabsTrigger` is a legacy alias for `TabsTab`. `TabsContent` is a legacy alias for `TabsPanel`.

**Tabs props:** `variant` ("default"|"underline"), `orientation` ("horizontal"|"vertical"), `defaultValue`

```tsx
<Tabs defaultValue="tab-1">
  <TabsList>
    <TabsTab value="tab-1">Tab 1</TabsTab>
    <TabsTab value="tab-2">Tab 2</TabsTab>
  </TabsList>
  <TabsPanel value="tab-1">Content 1</TabsPanel>
  <TabsPanel value="tab-2">Content 2</TabsPanel>
</Tabs>

<Tabs defaultValue="tab-1" variant="underline">...</Tabs>
<Tabs defaultValue="tab-1" orientation="vertical">...</Tabs>
```

Docs: https://coss.com/ui/docs/components/tabs

---

## Toggle

CLI: `pnpm dlx shadcn@latest add @coss/toggle`

```tsx
import { Toggle } from "@/components/ui/toggle"
```

**Props:** `variant` ("default"|"outline"), `size` ("default"|"sm"|"lg")

```tsx
<Toggle>Toggle</Toggle>
<Toggle variant="outline"><BoldIcon /></Toggle>
<Toggle size="sm" variant="outline">Small</Toggle>
<Toggle disabled variant="outline">Disabled</Toggle>
```

Docs: https://coss.com/ui/docs/components/toggle

---

## Toggle Group

CLI: `pnpm dlx shadcn@latest add @coss/toggle-group`

```tsx
import { Toggle, ToggleGroup } from "@/components/ui/toggle-group"
```

`ToggleGroupItem` is a legacy alias for `Toggle`. Also has `ToggleGroupSeparator`.

**ToggleGroup props:** `variant` ("default"|"outline"), `size` ("default"|"sm"|"lg"), `orientation`, `disabled`, `multiple`, `defaultValue` (string[])

```tsx
<ToggleGroup defaultValue={["bold"]}>
  <Toggle aria-label="Bold" value="bold"><BoldIcon /></Toggle>
  <Toggle aria-label="Italic" value="italic"><ItalicIcon /></Toggle>
</ToggleGroup>

// Outline with separators
<ToggleGroup defaultValue={["bold"]} variant="outline">
  <Toggle value="bold"><BoldIcon /></Toggle>
  <ToggleGroupSeparator />
  <Toggle value="italic"><ItalicIcon /></Toggle>
</ToggleGroup>

// Multiple selection
<ToggleGroup defaultValue={["bold"]} multiple>...</ToggleGroup>
```

Docs: https://coss.com/ui/docs/components/toggle-group

---

## Toolbar

CLI: `pnpm dlx shadcn@latest add @coss/toolbar`

```tsx
import { Toolbar, ToolbarButton, ToolbarGroup, ToolbarSeparator } from "@/components/ui/toolbar"
```

Also has `ToolbarLink`.

```tsx
<Toolbar>
  <ToolbarGroup>
    <ToolbarButton render={<Toggle />}>Bold</ToolbarButton>
    <ToolbarButton render={<Toggle />}>Underline</ToolbarButton>
  </ToolbarGroup>
  <ToolbarSeparator />
  <ToolbarButton render={<Button />}>Save</ToolbarButton>
</Toolbar>
```

Docs: https://coss.com/ui/docs/components/toolbar
