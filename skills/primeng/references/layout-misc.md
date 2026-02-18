# PrimeNG Layout & Miscellaneous Components

## Accordion

```html
<p-accordion [multiple]="true" [activeIndex]="0">
  <p-accordionpanel header="Section 1" value="0">
    <p>Content 1</p>
  </p-accordionpanel>
  <p-accordionpanel header="Section 2" value="1">
    <p>Content 2</p>
  </p-accordionpanel>
</p-accordion>
```

## Tabs

```html
<p-tabs value="0">
  <p-tablist>
    <p-tab value="0">General</p-tab>
    <p-tab value="1">Details</p-tab>
    <p-tab value="2">Settings</p-tab>
  </p-tablist>
  <p-tabpanels>
    <p-tabpanel value="0"><p>General content</p></p-tabpanel>
    <p-tabpanel value="1"><p>Detail content</p></p-tabpanel>
    <p-tabpanel value="2"><p>Settings content</p></p-tabpanel>
  </p-tabpanels>
</p-tabs>

<!-- Closable / Scrollable -->
<p-tabs value="0" [scrollable]="true">
  <p-tablist>
    @for (tab of tabs; track tab.value) {
      <p-tab [value]="tab.value" [closable]="true">{{ tab.label }}</p-tab>
    }
  </p-tablist>
</p-tabs>
```

## Panel

```html
<p-panel header="User Details" [toggleable]="true" [collapsed]="false">
  <p>Panel content</p>
</p-panel>
```

## Card

```html
<p-card header="Title" subheader="Subtitle" [style]="{ width: '350px' }">
  <ng-template #header>
    <img src="image.jpg" alt="Header" />
  </ng-template>
  <p>Card content</p>
  <ng-template #footer>
    <div class="flex gap-2">
      <p-button label="Save" icon="pi pi-check" />
      <p-button label="Cancel" severity="secondary" [outlined]="true" />
    </div>
  </ng-template>
</p-card>
```

## Splitter

Resizable split panels:

```html
<p-splitter [style]="{ height: '300px' }">
  <p-splitterpanel [size]="40" [minSize]="20">
    <p>Left Panel</p>
  </p-splitterpanel>
  <p-splitterpanel [size]="60">
    <p>Right Panel</p>
  </p-splitterpanel>
</p-splitter>

<!-- Nested vertical -->
<p-splitter layout="vertical">
  <p-splitterpanel>Top</p-splitterpanel>
  <p-splitterpanel>Bottom</p-splitterpanel>
</p-splitter>
```

## Toolbar

```html
<p-toolbar>
  <ng-template #start>
    <p-button label="New" icon="pi pi-plus" class="mr-2" />
    <p-button label="Delete" icon="pi pi-trash" severity="danger" [outlined]="true" />
  </ng-template>
  <ng-template #center>
    <p-iconfield>
      <p-inputicon class="pi pi-search" />
      <input pInputText placeholder="Search" />
    </p-iconfield>
  </ng-template>
  <ng-template #end>
    <p-button icon="pi pi-upload" label="Export" severity="secondary" />
  </ng-template>
</p-toolbar>
```

## Tree

```html
<p-tree
  [value]="nodes"
  selectionMode="checkbox"
  [(selection)]="selectedNodes"
  [filter]="true"
  filterPlaceholder="Search..."
/>
```

```typescript
import { TreeNode } from 'primeng/api';

nodes: TreeNode[] = [
  {
    label: 'Documents',
    icon: 'pi pi-folder',
    expandedIcon: 'pi pi-folder-open',
    children: [
      { label: 'resume.pdf', icon: 'pi pi-file', data: { size: '250KB' } },
      { label: 'cover.docx', icon: 'pi pi-file', data: { size: '120KB' } }
    ]
  },
  {
    label: 'Pictures',
    icon: 'pi pi-folder',
    children: [
      { label: 'photo.jpg', icon: 'pi pi-image' }
    ]
  }
];
```

### Tree selection modes

- `'single'` — one node at a time
- `'multiple'` — multiple with Ctrl/Meta
- `'checkbox'` — checkbox selection with parent/child propagation

## Tag

```html
<p-tag value="New" />
<p-tag value="Active" severity="success" />
<p-tag value="Warning" severity="warn" />
<p-tag value="Error" severity="danger" />
<p-tag value="Info" severity="info" />
<p-tag value="Beta" severity="secondary" />
<p-tag value="Custom" icon="pi pi-star" [rounded]="true" />
```

## Badge

```html
<!-- Standalone -->
<p-badge value="4" />
<p-badge value="2" severity="danger" />

<!-- As overlay on other elements -->
<i class="pi pi-bell" pBadge value="3" severity="danger"></i>
<p-button label="Inbox" badge="5" badgeSeverity="danger" />
```

## Avatar

```html
<!-- Image -->
<p-avatar image="avatar.png" shape="circle" size="large" />

<!-- Label (initials) -->
<p-avatar label="JD" shape="circle" [style]="{ 'background-color': '#9c27b0', color: '#fff' }" />

<!-- Icon -->
<p-avatar icon="pi pi-user" shape="circle" />

<!-- Group -->
<p-avatargroup>
  <p-avatar image="user1.png" shape="circle" />
  <p-avatar image="user2.png" shape="circle" />
  <p-avatar image="user3.png" shape="circle" />
  <p-avatar label="+5" shape="circle" [style]="{ 'background-color': '#dee2e6' }" />
</p-avatargroup>
```

## ProgressBar

```html
<p-progressbar [value]="65" />
<p-progressbar [value]="40" [showValue]="false" />
<p-progressbar mode="indeterminate" [style]="{ height: '4px' }" />
```

## Skeleton

Loading placeholder:

```html
<p-skeleton width="100%" height="2rem" />
<p-skeleton width="10rem" height="1rem" />
<p-skeleton shape="circle" size="5rem" />
<p-skeleton width="100%" height="150px" borderRadius="16px" />
```

## Message / Messages

```html
<!-- Inline message -->
<p-message severity="success">Operation completed successfully.</p-message>
<p-message severity="error" size="small" variant="simple">Field is required</p-message>
<p-message severity="info" [closable]="true">This is dismissible</p-message>

<!-- Multiple messages -->
<p-messages [(value)]="messages" [closable]="true" />
```

```typescript
messages: Message[] = [
  { severity: 'info', summary: 'Info', detail: 'First message' },
  { severity: 'warn', summary: 'Warning', detail: 'Second message' }
];
```

## Chip

```html
<p-chip label="Angular" />
<p-chip label="John Doe" image="avatar.png" [removable]="true" (onRemove)="onRemove($event)" />
<p-chip label="Active" icon="pi pi-check" />
```

## ScrollTop

```html
<!-- Scroll-to-top button (appears when scrolling down) -->
<p-scrolltop />
<p-scrolltop [threshold]="200" icon="pi pi-arrow-up" behavior="smooth" />
```

## Chart

Wrapper around Chart.js:

```html
<p-chart type="bar" [data]="chartData" [options]="chartOptions" />
```

```typescript
// npm install chart.js
chartData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr'],
  datasets: [
    {
      label: 'Sales',
      data: [540, 325, 702, 620],
      backgroundColor: ['#42A5F5', '#66BB6A', '#FFA726', '#26C6DA']
    }
  ]
};

chartOptions = {
  responsive: true,
  plugins: { legend: { position: 'top' } }
};
```

Chart types: `bar`, `line`, `pie`, `doughnut`, `polarArea`, `radar`, `scatter`, `bubble`, `combo`

## FileUpload

```html
<!-- Advanced (with preview) -->
<p-fileupload
  name="files"
  url="/api/upload"
  [multiple]="true"
  accept="image/*"
  [maxFileSize]="1000000"
  (onUpload)="onUpload($event)"
  (onError)="onError($event)"
/>

<!-- Basic -->
<p-fileupload
  mode="basic"
  name="file"
  url="/api/upload"
  accept=".csv,.xlsx"
  chooseLabel="Import"
  [auto]="true"
/>

<!-- Custom (manual upload) -->
<p-fileupload
  [customUpload]="true"
  (uploadHandler)="customUpload($event)"
  [multiple]="true"
/>
```

## Editor (Rich Text)

```html
<p-editor [(ngModel)]="htmlContent" [style]="{ height: '320px' }" />
```

Requires Quill: `npm install quill`
