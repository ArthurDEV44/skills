# PrimeNG Data Components

## Table (p-table)

The most powerful PrimeNG component. Fully featured data grid with sorting, filtering, pagination, selection, lazy loading, virtual scroll, editing, frozen columns, and export.

### Sorting

```html
<p-table [value]="products">
  <ng-template #header>
    <tr>
      <th pSortableColumn="name">Name <p-sortIcon field="name" /></th>
      <th pSortableColumn="price">Price <p-sortIcon field="price" /></th>
    </tr>
  </ng-template>
  <ng-template #body let-product>
    <tr>
      <td>{{ product.name }}</td>
      <td>{{ product.price | currency }}</td>
    </tr>
  </ng-template>
</p-table>
```

Multi-sort: `[sortMode]="'multiple'"` — hold Ctrl/Meta to sort by multiple columns.

### Column Filters

```html
<th>
  <div class="flex items-center gap-2">
    Name
    <p-columnFilter type="text" field="name" display="menu" />
  </div>
</th>

<!-- Filter types: text, numeric, date, boolean -->
<!-- Display modes: menu (popup), row (inline) -->

<!-- Custom filter template -->
<p-columnFilter field="status" matchMode="equals" display="menu">
  <ng-template #filter let-value let-filter="filterCallback">
    <p-select
      [(ngModel)]="value"
      [options]="statuses"
      (onChange)="filter($event.value)"
      placeholder="Select Status"
    />
  </ng-template>
</p-columnFilter>
```

### Selection

```html
<!-- Single selection -->
<p-table [value]="products" selectionMode="single" [(selection)]="selectedProduct" dataKey="id">
  <ng-template #body let-product>
    <tr [pSelectableRow]="product">
      <td>{{ product.name }}</td>
    </tr>
  </ng-template>
</p-table>

<!-- Checkbox selection -->
<p-table [value]="products" [(selection)]="selectedProducts" dataKey="id">
  <ng-template #header>
    <tr>
      <th style="width: 4rem"><p-tableHeaderCheckbox /></th>
      <th>Name</th>
    </tr>
  </ng-template>
  <ng-template #body let-product>
    <tr>
      <td><p-tableCheckbox [value]="product" /></td>
      <td>{{ product.name }}</td>
    </tr>
  </ng-template>
</p-table>
```

### Pagination

```html
<p-table
  [value]="products"
  [paginator]="true"
  [rows]="10"
  [rowsPerPageOptions]="[10, 25, 50]"
  [showCurrentPageReport]="true"
  currentPageReportTemplate="Showing {first} to {last} of {totalRecords}"
  [showFirstLastIcon]="true"
  [showPageLinks]="true"
  [showJumpToPageDropdown]="true"
>
```

### Lazy Loading (Server-Side)

```html
<p-table
  [value]="products"
  [lazy]="true"
  [paginator]="true"
  [rows]="10"
  [totalRecords]="totalRecords"
  [loading]="loading"
  (onLazyLoad)="loadData($event)"
>
```

```typescript
loadData(event: TableLazyLoadEvent) {
  this.loading = true;
  const { first, rows, sortField, sortOrder, filters, globalFilter } = event;

  this.productService.getProducts({
    skip: first,
    take: rows,
    sortField,
    sortOrder,
    filters
  }).subscribe(result => {
    this.products = result.data;
    this.totalRecords = result.total;
    this.loading = false;
  });
}
```

### Virtual Scroll

For large datasets (thousands of rows) without pagination:

```html
<p-table
  [value]="products"
  [scrollable]="true"
  scrollHeight="400px"
  [virtualScroll]="true"
  [virtualScrollItemSize]="46"
  [rows]="100"
  [lazy]="true"
  (onLazyLoad)="loadLazy($event)"
>
  <ng-template #loadingbody let-columns="columns">
    <tr style="height: 46px">
      <td *ngFor="let col of columns">
        <p-skeleton [ngStyle]="{ width: '60%' }" />
      </td>
    </tr>
  </ng-template>
</p-table>
```

### Row Expansion

```html
<p-table [value]="products" dataKey="id">
  <ng-template #header>
    <tr>
      <th style="width: 3rem"></th>
      <th>Name</th>
    </tr>
  </ng-template>
  <ng-template #body let-product let-expanded="expanded">
    <tr>
      <td>
        <p-button
          type="button"
          pRipple
          [pRowToggler]="product"
          [text]="true"
          [rounded]="true"
          [plain]="true"
          [icon]="expanded ? 'pi pi-chevron-down' : 'pi pi-chevron-right'"
        />
      </td>
      <td>{{ product.name }}</td>
    </tr>
  </ng-template>
  <ng-template #rowexpansion let-product>
    <tr>
      <td colspan="2">
        <div class="p-4">
          <h5>Orders for {{ product.name }}</h5>
          <p-table [value]="product.orders" dataKey="id">
            <!-- Nested table -->
          </p-table>
        </div>
      </td>
    </tr>
  </ng-template>
</p-table>
```

### Inline Editing

```html
<p-table [value]="products" editMode="row" dataKey="id">
  <ng-template #body let-product let-editing="editing" let-ri="rowIndex">
    <tr [pEditableRow]="product">
      <td>
        <p-cellEditor>
          <ng-template #output>{{ product.name }}</ng-template>
          <ng-template #input>
            <input pInputText [(ngModel)]="product.name" />
          </ng-template>
        </p-cellEditor>
      </td>
      <td>
        <p-button *ngIf="!editing" icon="pi pi-pencil" pInitEditableRow (click)="onRowEditInit(product)" />
        <p-button *ngIf="editing" icon="pi pi-check" pSaveEditableRow (click)="onRowEditSave(product)" />
        <p-button *ngIf="editing" icon="pi pi-times" pCancelEditableRow (click)="onRowEditCancel(product, ri)" />
      </td>
    </tr>
  </ng-template>
</p-table>
```

### Frozen Columns

```html
<p-table [value]="products" [scrollable]="true" scrollHeight="400px" styleClass="mt-4">
  <ng-template #header>
    <tr>
      <th style="min-width: 200px" pFrozenColumn>Name</th>
      <th style="min-width: 200px">Col 1</th>
      <th style="min-width: 200px">Col 2</th>
      <!-- ... many columns ... -->
      <th style="min-width: 200px" pFrozenColumn alignFrozen="right">Actions</th>
    </tr>
  </ng-template>
</p-table>
```

### CSV Export

```html
<p-table #dt [value]="products" [columns]="cols">
  <ng-template #caption>
    <p-button icon="pi pi-download" label="Export" (click)="dt.exportCSV()" />
  </ng-template>
</p-table>
```

### Global Filter

```html
<p-table #dt [value]="products" [globalFilterFields]="['name', 'country.name', 'status']">
  <ng-template #caption>
    <p-iconfield iconPosition="left">
      <p-inputicon class="pi pi-search" />
      <input pInputText (input)="dt.filterGlobal($event.target.value, 'contains')" placeholder="Search..." />
    </p-iconfield>
  </ng-template>
</p-table>
```

## TreeTable

Hierarchical data table:

```html
<p-treetable [value]="files" [columns]="cols">
  <ng-template #header let-columns>
    <tr>
      <th *ngFor="let col of columns">{{ col.header }}</th>
    </tr>
  </ng-template>
  <ng-template #body let-rowNode let-rowData="rowData" let-columns="columns">
    <tr>
      <td *ngFor="let col of columns; let i = index">
        <p-treeTableToggler [rowNode]="rowNode" *ngIf="i === 0" />
        {{ rowData[col.field] }}
      </td>
    </tr>
  </ng-template>
</p-treetable>
```

## DataView

Grid/list layout for data display:

```html
<p-dataview [value]="products" [layout]="layout">
  <ng-template #header>
    <div class="flex justify-end">
      <p-selectbutton [(ngModel)]="layout" [options]="['list', 'grid']" />
    </div>
  </ng-template>
  <ng-template #list let-items>
    <div *ngFor="let product of items" class="flex gap-4 p-4 border-b">
      <img [src]="product.image" [alt]="product.name" width="80" />
      <div>
        <h5>{{ product.name }}</h5>
        <span>{{ product.price | currency }}</span>
      </div>
    </div>
  </ng-template>
  <ng-template #grid let-items>
    <div class="grid grid-cols-3 gap-4 p-4">
      <div *ngFor="let product of items" class="border rounded p-4">
        <img [src]="product.image" [alt]="product.name" class="w-full" />
        <h5>{{ product.name }}</h5>
        <span>{{ product.price | currency }}</span>
      </div>
    </div>
  </ng-template>
</p-dataview>
```
