---
name: coss-ui
description: "coss ui component library built on Base UI and Tailwind CSS for React applications. Copy-paste ownership model with 50+ accessible, composable components. Use when: (1) Building React UI with coss ui components (Accordion, Alert, Button, Dialog, Select, Tabs, Toast, etc.), (2) Installing or setting up coss ui in a project, (3) Styling coss ui components with CSS variables and color tokens, (4) Migrating from shadcn/ui or Radix UI to coss ui (asChild to render prop, component renaming), (5) Using Base UI primitives with coss ui wrappers, (6) Implementing forms with Field, Fieldset, Form, and validation, (7) Working with any @coss/* component package, (8) Writing React code that imports from @/components/ui/*."
---

# coss ui

## Overview

coss ui is a modern UI component library built on Base UI and Tailwind CSS for React. It uses a copy-paste ownership model: components are copied into your project at `components/ui/`, giving full control. Built by Cal.com.

**Architecture layers:**
- **Primitives** - Unstyled, accessible Base UI building blocks
- **Particles** - Pre-assembled components combining multiple primitives
- **Atoms** - API-enhanced particles integrating external data services

## Quick Start

### Install all components
```bash
pnpm dlx shadcn@latest add @coss/ui @coss/colors-neutral
```

### Install a single component
```bash
pnpm dlx shadcn@latest add @coss/<component-name>
```

### Base UI setup
Add `isolation: isolate` to root wrapper div (prevents z-index conflicts with portals). Add `position: relative` to body for iOS Safari 26+.

## Key Patterns

### render prop (replaces Radix asChild)
```tsx
// Radix: <Trigger asChild><Button>Click</Button></Trigger>
// coss ui:
<MenuTrigger render={<Button />}>Click</MenuTrigger>
<DialogClose render={<Button variant="ghost" />}>Cancel</DialogClose>
<Badge render={<Link href="/" />}>Badge</Badge>
```

### Component naming
- `*Popup` preferred over `*Content` (legacy alias kept)
- `*Panel` preferred over `*Content` for main content areas
- Legacy names (`DialogContent`, `SheetContent`, etc.) still exported

### Sizing
Components are more compact than shadcn/ui. Use `size="lg"` to match original shadcn/ui heights.
- Input/Select/Textarea: sm (28px), default (32px), lg (36px)
- Button: xs (24px), sm, default, lg, xl (40px), icon variants

### Color tokens
Extended beyond shadcn/ui palette:
- `--destructive-foreground` - destructive buttons, menu items, badges, field errors
- `--info` / `--info-foreground` - info badges, toasts, alerts
- `--success` / `--success-foreground` - success badges, toasts, alerts
- `--warning` / `--warning-foreground` - warning badges, toasts, alerts

Install color tokens: `pnpm dlx shadcn@latest add @coss/colors-neutral`

## Component Reference

For detailed component API, props, and examples, read the appropriate reference file:

- **Form components** (Input, Textarea, Field, Fieldset, Form, Checkbox, Radio, Select, Combobox, Autocomplete, Switch, Slider, NumberField): See [references/components-forms.md](references/components-forms.md)
- **Overlay components** (Dialog, AlertDialog, Sheet, Popover, Tooltip, PreviewCard, Menu, Command, Toast): See [references/components-overlays.md](references/components-overlays.md)
- **Layout & display components** (Accordion, Alert, Avatar, Badge, Breadcrumb, Button, Card, Collapsible, Empty, Frame, Group, InputGroup, Kbd, Label, Meter, Pagination, Progress, ScrollArea, Separator, Skeleton, Spinner, Table, Tabs, Toggle, ToggleGroup, Toolbar): See [references/components-display.md](references/components-display.md)
- **Migration guide** (Radix/shadcn to coss ui): See [references/migration.md](references/migration.md)

## Documentation Links

- Overview: https://coss.com/ui/docs
- Get Started: https://coss.com/ui/docs/get-started
- Styling: https://coss.com/ui/docs/styling
- Migration: https://coss.com/ui/docs/radix-migration
- Roadmap: https://coss.com/ui/docs/roadmap
- LLM docs: https://coss.com/ui/llms.txt
- Origin UI: https://coss.com/origin
- Component docs: https://coss.com/ui/docs/components/<name>
