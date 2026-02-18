---
name: primeng
description: >-
  PrimeNG UI component library for Angular with 80+ components, theming, and accessibility.
  Use when: (1) Setting up PrimeNG in an Angular project with providePrimeNG, theme presets,
  and animations, (2) Using PrimeNG components — Table, Dialog, Toast, Menu, Form inputs,
  Tree, Accordion, Tabs, DataView, (3) Theming with Aura/Lara/Nora/Material presets,
  definePreset, design tokens, dark mode, styled/unstyled modes, CSS layers,
  (4) Building data tables with sorting, filtering, pagination, selection, lazy loading,
  virtual scroll, row expansion, inline editing, frozen columns, CSV export,
  (5) Creating forms with InputText, Select, AutoComplete, DatePicker, MultiSelect,
  TreeSelect, Checkbox, RadioButton, InputNumber, Slider, Rating, Password, Textarea,
  (6) Using overlays — Dialog, ConfirmDialog, ConfirmPopup, Drawer, Popover, Toast,
  (7) Navigation — Menubar, TieredMenu, ContextMenu, MegaMenu, Breadcrumb, TabMenu, Steps,
  (8) PrimeNG templates (ng-template) with #header, #body, #footer, #caption, #filter,
  (9) Accessibility and WCAG compliance with PrimeNG components,
  (10) PrimeFlex utility CSS or Tailwind CSS integration with PrimeNG.
---

# PrimeNG

PrimeNG is a comprehensive UI component library for Angular, offering 80+ feature-rich, accessible, and themeable components. It uses a design-token-based theming architecture with built-in presets (Aura, Lara, Nora, Material).

## Installation

```bash
npm install primeng @primeuix/themes primeicons
# Animations (required)
npm install @angular/animations
```

## Configuration

Configure PrimeNG in `app.config.ts` via `providePrimeNG`:

```typescript
import { ApplicationConfig } from '@angular/core';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { providePrimeNG } from 'primeng/config';
import Aura from '@primeuix/themes/aura';

export const appConfig: ApplicationConfig = {
  providers: [
    provideAnimationsAsync(),
    providePrimeNG({
      theme: {
        preset: Aura,
        options: {
          prefix: 'p',                    // CSS variable prefix
          darkModeSelector: 'system',     // 'system' | '.my-dark-class' | false
          cssLayer: false                 // Enable CSS @layer for specificity control
        }
      },
      ripple: true,                       // Material ripple effect
      inputVariant: 'outlined',           // 'outlined' | 'filled'
      zIndex: {
        modal: 1100,
        overlay: 1000,
        menu: 1000,
        tooltip: 1100
      }
    })
  ]
};
```

### Available Presets

| Preset | Import | Style |
|--------|--------|-------|
| Aura | `@primeuix/themes/aura` | Modern, clean |
| Lara | `@primeuix/themes/lara` | Bootstrap-inspired |
| Nora | `@primeuix/themes/nora` | Enterprise, compact |
| Material | `@primeuix/themes/material` | Material Design 3 |

### PrimeIcons

```typescript
// In styles.scss or angular.json
@import 'primeicons/primeicons.css';
```

Usage: `<i class="pi pi-check"></i>` — see [PrimeIcons catalog](https://primeng.org/icons)

## Importing Components

PrimeNG v20 uses standalone components. Import each component directly:

```typescript
import { Component } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { TableModule } from 'primeng/table';
import { DialogModule } from 'primeng/dialog';

@Component({
  selector: 'app-demo',
  standalone: true,
  imports: [ButtonModule, TableModule, DialogModule],
  template: `...`
})
export class DemoComponent {}
```

## Core Component Categories

### Button

```html
<!-- Variants -->
<p-button label="Primary" />
<p-button label="Secondary" severity="secondary" />
<p-button label="Success" severity="success" />
<p-button label="Danger" severity="danger" />
<p-button label="Warning" severity="warn" />

<!-- Styles -->
<p-button label="Outlined" [outlined]="true" />
<p-button label="Text" [text]="true" />
<p-button label="Raised" [raised]="true" />
<p-button label="Rounded" [rounded]="true" />
<p-button icon="pi pi-check" label="Save" [loading]="saving" (onClick)="save()" />
<p-button icon="pi pi-trash" severity="danger" [rounded]="true" [outlined]="true" />
```

### Table (p-table)

The most feature-rich component. Supports sorting, filtering, pagination, selection, lazy loading, virtual scroll, row expansion, inline editing, frozen columns, and CSV export.

```html
<p-table
  [value]="products"
  [paginator]="true"
  [rows]="10"
  [rowsPerPageOptions]="[10, 25, 50]"
  [globalFilterFields]="['name', 'category']"
  dataKey="id"
  [rowHover]="true"
  [showCurrentPageReport]="true"
  currentPageReportTemplate="Showing {first} to {last} of {totalRecords}"
>
  <ng-template #caption>
    <div class="flex justify-between items-center">
      <h5>Products</h5>
      <p-iconfield iconPosition="left">
        <p-inputicon class="pi pi-search" />
        <input pInputText type="text" (input)="dt.filterGlobal($event.target.value, 'contains')" placeholder="Search..." />
      </p-iconfield>
    </div>
  </ng-template>

  <ng-template #header>
    <tr>
      <th pSortableColumn="name">Name <p-sortIcon field="name" /></th>
      <th pSortableColumn="price">Price <p-sortIcon field="price" /></th>
      <th>Actions</th>
    </tr>
  </ng-template>

  <ng-template #body let-product>
    <tr>
      <td>{{ product.name }}</td>
      <td>{{ product.price | currency }}</td>
      <td>
        <p-button icon="pi pi-pencil" [rounded]="true" [outlined]="true" (click)="edit(product)" />
      </td>
    </tr>
  </ng-template>
</p-table>
```

For advanced Table patterns (lazy loading, virtual scroll, column filters, selection, expansion, frozen columns, export), see [references/data-components.md](references/data-components.md).

### Dialog

```html
<p-button label="Open" (onClick)="visible = true" />

<p-dialog
  header="Edit Item"
  [(visible)]="visible"
  [modal]="true"
  [style]="{ width: '450px' }"
  [closable]="true"
  [draggable]="false"
>
  <ng-template #content>
    <div class="flex flex-col gap-4">
      <input pInputText [(ngModel)]="item.name" placeholder="Name" />
    </div>
  </ng-template>
  <ng-template #footer>
    <p-button label="Cancel" severity="secondary" (onClick)="visible = false" />
    <p-button label="Save" (onClick)="save()" />
  </ng-template>
</p-dialog>
```

### Toast

```typescript
import { MessageService } from 'primeng/api';

@Component({
  providers: [MessageService],
  template: `
    <p-toast />
    <p-button label="Show" (onClick)="show()" />
  `
})
export class Demo {
  constructor(private messageService: MessageService) {}

  show() {
    this.messageService.add({
      severity: 'success',  // 'success' | 'info' | 'warn' | 'error' | 'secondary' | 'contrast'
      summary: 'Saved',
      detail: 'Record updated successfully',
      life: 3000
    });
  }
}
```

### ConfirmDialog

```typescript
import { ConfirmationService } from 'primeng/api';

@Component({
  providers: [ConfirmationService],
  template: `
    <p-confirmdialog />
    <p-button label="Delete" severity="danger" (onClick)="confirmDelete()" />
  `
})
export class Demo {
  constructor(private confirmationService: ConfirmationService) {}

  confirmDelete() {
    this.confirmationService.confirm({
      message: 'Are you sure you want to delete this record?',
      header: 'Confirm Delete',
      icon: 'pi pi-exclamation-triangle',
      acceptButtonProps: { severity: 'danger', label: 'Delete' },
      rejectButtonProps: { severity: 'secondary', label: 'Cancel' },
      accept: () => this.delete(),
      reject: () => {}
    });
  }
}
```

### Form Inputs

```html
<!-- Text -->
<input pInputText [(ngModel)]="value" placeholder="Name" />
<textarea pTextarea [(ngModel)]="text" [rows]="5" [autoResize]="true"></textarea>
<p-password [(ngModel)]="pw" [toggleMask]="true" [feedback]="true" />

<!-- Numbers -->
<p-inputnumber [(ngModel)]="price" mode="currency" currency="USD" locale="en-US" />
<p-inputnumber [(ngModel)]="qty" [showButtons]="true" [min]="0" [max]="100" />

<!-- Select -->
<p-select [(ngModel)]="city" [options]="cities" optionLabel="name" placeholder="Select City" />
<p-multiselect [(ngModel)]="selected" [options]="cities" optionLabel="name" display="chip" />
<p-autocomplete [(ngModel)]="query" [suggestions]="results" (completeMethod)="search($event)" field="name" />

<!-- Date -->
<p-datepicker [(ngModel)]="date" [showIcon]="true" dateFormat="yy-mm-dd" />
<p-datepicker [(ngModel)]="range" selectionMode="range" [readonlyInput]="true" />

<!-- Boolean / Choice -->
<p-checkbox [(ngModel)]="checked" [binary]="true" label="Accept terms" />
<p-radiobutton [(ngModel)]="category" value="A" label="Option A" />
<p-toggleswitch [(ngModel)]="active" />
<p-slider [(ngModel)]="volume" [min]="0" [max]="100" />
<p-rating [(ngModel)]="score" [cancel]="false" />

<!-- Special -->
<p-treeselect [(ngModel)]="selectedNodes" [options]="nodes" selectionMode="checkbox" display="chip" />
<p-colorpicker [(ngModel)]="color" />
<p-inputotp [(ngModel)]="otp" [length]="6" />
```

For detailed form patterns with validation, see [references/form-components.md](references/form-components.md).

### Navigation / Menu

```html
<!-- Top menubar -->
<p-menubar [model]="menuItems">
  <ng-template #end>
    <p-avatar image="avatar.png" shape="circle" />
  </ng-template>
</p-menubar>

<!-- Breadcrumb -->
<p-breadcrumb [model]="breadcrumbItems" [home]="home" />

<!-- Tab navigation -->
<p-tabmenu [model]="tabItems" />

<!-- Step indicator -->
<p-steps [model]="stepItems" [(activeIndex)]="activeStep" />
```

Menu items follow the `MenuItem` interface:

```typescript
import { MenuItem } from 'primeng/api';

items: MenuItem[] = [
  {
    label: 'File',
    icon: 'pi pi-file',
    items: [
      { label: 'New', icon: 'pi pi-plus', command: () => this.create() },
      { label: 'Open', icon: 'pi pi-folder-open', routerLink: '/files' },
      { separator: true },
      { label: 'Quit', icon: 'pi pi-times' }
    ]
  }
];
```

For all menu types, see [references/menu-components.md](references/menu-components.md).

## Template System (ng-template)

PrimeNG uses Angular's `ng-template` with named references for customization:

| Template | Used In | Purpose |
|----------|---------|---------|
| `#header` | Table, Dialog, Card, Panel | Header content |
| `#body` | Table | Row template |
| `#footer` | Table, Dialog, Card | Footer content |
| `#caption` | Table | Table caption / toolbar area |
| `#filter` | ColumnFilter | Custom filter UI |
| `#content` | Dialog, Drawer | Main content area |
| `#item` | Select, MultiSelect, Listbox | Custom option rendering |
| `#selectedItem` | Select | Selected value display |
| `#empty` | Table, Select | Empty state |
| `#loading` | Table | Loading state row |
| `#loadingbody` | Table (virtual scroll) | Skeleton row template |
| `#start` / `#end` | Toolbar, Menubar | Left/right sections |

## Theming

PrimeNG v20 uses a design-token architecture. Customize via `definePreset`:

```typescript
import { definePreset } from '@primeuix/themes';
import Aura from '@primeuix/themes/aura';

const MyPreset = definePreset(Aura, {
  semantic: {
    primary: {
      50: '{indigo.50}', 100: '{indigo.100}', 200: '{indigo.200}',
      300: '{indigo.300}', 400: '{indigo.400}', 500: '{indigo.500}',
      600: '{indigo.600}', 700: '{indigo.700}', 800: '{indigo.800}',
      900: '{indigo.900}', 950: '{indigo.950}'
    },
    colorScheme: {
      light: {
        surface: { 0: '#ffffff', 50: '{zinc.50}', /* ...shades */ 950: '{zinc.950}' }
      },
      dark: {
        surface: { 0: '#ffffff', 50: '{slate.50}', /* ...shades */ 950: '{slate.950}' }
      }
    }
  }
});
```

For detailed theming (token layers, component tokens, unstyled mode, CSS layers, runtime switching), see [references/theming.md](references/theming.md).

## Accessibility

- All PrimeNG components are WCAG 2.1 compliant
- Keyboard navigation built-in (arrow keys, Enter, Escape, Tab)
- ARIA attributes auto-applied (`role`, `aria-label`, `aria-expanded`, etc.)
- Add custom labels: `ariaLabel`, `ariaLabelledBy` props on most components
- Screen reader support for dynamic content (Toast, Dialog, live regions)

## CSS Integration

### With Tailwind CSS

PrimeNG works alongside Tailwind. Enable CSS layers to avoid specificity conflicts:

```typescript
providePrimeNG({
  theme: {
    preset: Aura,
    options: { cssLayer: { name: 'primeng', order: 'tailwind-base, primeng, tailwind-utilities' } }
  }
})
```

```css
/* styles.css */
@layer tailwind-base, primeng, tailwind-utilities;
@layer tailwind-base { @tailwind base; }
@layer tailwind-utilities { @tailwind components; @tailwind utilities; }
```

### With PrimeFlex

```bash
npm install primeflex
```

```scss
@import 'primeflex/primeflex.scss';
```

## Reference Guides

- **Theming**: See [references/theming.md](references/theming.md) — design tokens, definePreset, dark mode, unstyled mode, CSS layers, runtime theme switching
- **Data Components**: See [references/data-components.md](references/data-components.md) — Table (lazy, virtual scroll, filters, selection, frozen columns, export), TreeTable, DataView, Paginator
- **Form Components**: See [references/form-components.md](references/form-components.md) — all input types, reactive/template forms, validation patterns, InputGroup
- **Overlay Components**: See [references/overlay-components.md](references/overlay-components.md) — Dialog, ConfirmDialog, ConfirmPopup, Drawer, Popover, Toast, Tooltip
- **Menu Components**: See [references/menu-components.md](references/menu-components.md) — Menubar, TieredMenu, ContextMenu, MegaMenu, Breadcrumb, TabMenu, Steps, Dock
- **Layout & Misc**: See [references/layout-misc.md](references/layout-misc.md) — Accordion, Tabs, Panel, Splitter, Card, Toolbar, Tree, Tag, Badge, Avatar, ProgressBar, Skeleton, Chart
