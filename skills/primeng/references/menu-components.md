# PrimeNG Menu Components

All menu components share the `MenuItem` interface from `primeng/api`.

## MenuItem Interface

```typescript
import { MenuItem } from 'primeng/api';

const items: MenuItem[] = [
  {
    label: 'File',
    icon: 'pi pi-file',
    items: [                              // Sub-items
      { label: 'New', icon: 'pi pi-plus', command: () => this.newFile() },
      { label: 'Open', icon: 'pi pi-folder-open', routerLink: '/files' },
      { separator: true },               // Divider
      { label: 'Quit', icon: 'pi pi-times', disabled: true }
    ]
  },
  {
    label: 'Profile',
    icon: 'pi pi-user',
    routerLink: '/profile',              // Angular router navigation
    routerLinkActiveOptions: { exact: true },
    badge: '3',                           // Badge count
    badgeSeverity: 'danger'
  },
  {
    label: 'External',
    icon: 'pi pi-external-link',
    url: 'https://example.com',          // External URL
    target: '_blank'
  }
];
```

### Key MenuItem Properties

| Property | Type | Purpose |
|----------|------|---------|
| `label` | string | Display text |
| `icon` | string | PrimeIcon class |
| `command` | function | Click handler `(event) => void` |
| `routerLink` | string / any[] | Angular router link |
| `url` | string | External URL |
| `items` | MenuItem[] | Sub-menu items |
| `separator` | boolean | Divider line |
| `disabled` | boolean | Disable item |
| `visible` | boolean | Show/hide item |
| `badge` | string | Badge text |
| `badgeSeverity` | string | Badge color |
| `expanded` | boolean | Expanded state (PanelMenu) |
| `styleClass` | string | CSS class |
| `target` | string | Link target |

## Menubar

Horizontal top navigation bar:

```html
<p-menubar [model]="items">
  <ng-template #start>
    <img src="logo.svg" height="40" />
  </ng-template>
  <ng-template #end>
    <div class="flex items-center gap-2">
      <p-button icon="pi pi-bell" [rounded]="true" [text]="true" />
      <p-avatar image="avatar.png" shape="circle" />
    </div>
  </ng-template>
</p-menubar>
```

## Menu

Simple popup or inline menu:

```html
<!-- Popup (toggle on button click) -->
<p-button icon="pi pi-bars" (click)="menu.toggle($event)" />
<p-menu #menu [model]="items" [popup]="true" />

<!-- Inline (always visible) -->
<p-menu [model]="items" />
```

## TieredMenu

Multi-level popup menu:

```html
<p-button label="Menu" (click)="tieredMenu.toggle($event)" />
<p-tieredmenu #tieredMenu [model]="items" [popup]="true" />
```

## ContextMenu

Right-click menu:

```html
<!-- On a target element -->
<div (contextmenu)="cm.show($event)">
  Right-click here
</div>
<p-contextmenu #cm [model]="contextItems" />

<!-- On a table row -->
<p-table [value]="products" [(contextMenuSelection)]="selectedProduct" [contextMenu]="cm">
  <ng-template #body let-product>
    <tr [pContextMenuRow]="product">
      <td>{{ product.name }}</td>
    </tr>
  </ng-template>
</p-table>
<p-contextmenu #cm [model]="contextItems" />
```

## MegaMenu

Full-width dropdown with multi-column layout:

```html
<p-megamenu [model]="megaItems" orientation="horizontal">
  <ng-template #start>
    <img src="logo.svg" height="40" />
  </ng-template>
</p-megamenu>
```

```typescript
megaItems: MegaMenuItem[] = [
  {
    label: 'Products',
    items: [
      [
        { label: 'Category A', items: [
          { label: 'Item A1', routerLink: '/a1' },
          { label: 'Item A2', routerLink: '/a2' }
        ]},
      ],
      [
        { label: 'Category B', items: [
          { label: 'Item B1', routerLink: '/b1' },
          { label: 'Item B2', routerLink: '/b2' }
        ]}
      ]
    ]
  }
];
```

## Breadcrumb

```html
<p-breadcrumb [model]="breadcrumbs" [home]="homeItem" />
```

```typescript
homeItem: MenuItem = { icon: 'pi pi-home', routerLink: '/' };

breadcrumbs: MenuItem[] = [
  { label: 'Products', routerLink: '/products' },
  { label: 'Electronics', routerLink: '/products/electronics' },
  { label: 'Laptops' }
];
```

## TabMenu

Tab-style horizontal navigation:

```html
<p-tabmenu [model]="tabs" [(activeItem)]="activeTab" />
```

```typescript
tabs: MenuItem[] = [
  { label: 'Dashboard', icon: 'pi pi-home', routerLink: '/dashboard' },
  { label: 'Users', icon: 'pi pi-users', routerLink: '/users' },
  { label: 'Settings', icon: 'pi pi-cog', routerLink: '/settings' }
];
```

## Steps

Step-by-step wizard navigation:

```html
<p-steps [model]="steps" [(activeIndex)]="activeStep" [readonly]="false" />
```

```typescript
steps: MenuItem[] = [
  { label: 'Personal', routerLink: '/wizard/personal' },
  { label: 'Payment', routerLink: '/wizard/payment' },
  { label: 'Confirmation', routerLink: '/wizard/confirm' }
];
activeStep = 0;
```

## Dock

macOS-style dock:

```html
<p-dock [model]="dockItems" position="bottom">
  <ng-template #item let-item>
    <img [src]="item.icon" [alt]="item.label" width="100%" />
  </ng-template>
</p-dock>
```

## PanelMenu

Vertical accordion-style menu:

```html
<p-panelmenu [model]="panelItems" [multiple]="true" />
```

Items with `items` children are expandable panels. Supports `[expanded]="true"` for default-open panels.
