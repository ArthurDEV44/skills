# PrimeNG Overlay Components

## Dialog

```html
<p-dialog
  header="Edit Profile"
  [(visible)]="dialogVisible"
  [modal]="true"
  [closable]="true"
  [draggable]="false"
  [resizable]="false"
  [style]="{ width: '500px' }"
  [breakpoints]="{ '960px': '75vw', '640px': '95vw' }"
  (onHide)="onClose()"
>
  <ng-template #content>
    <div class="flex flex-col gap-4">
      <div class="flex flex-col gap-2">
        <label for="name">Name</label>
        <input pInputText id="name" [(ngModel)]="profile.name" />
      </div>
      <div class="flex flex-col gap-2">
        <label for="email">Email</label>
        <input pInputText id="email" [(ngModel)]="profile.email" />
      </div>
    </div>
  </ng-template>
  <ng-template #footer>
    <p-button label="Cancel" severity="secondary" [text]="true" (onClick)="dialogVisible = false" />
    <p-button label="Save" icon="pi pi-check" (onClick)="save()" />
  </ng-template>
</p-dialog>
```

### Key Props

| Prop | Type | Default | Purpose |
|------|------|---------|---------|
| `[(visible)]` | boolean | false | Two-way visibility binding |
| `[modal]` | boolean | false | Overlay backdrop |
| `[closable]` | boolean | true | Show close button |
| `[draggable]` | boolean | true | Enable dragging |
| `[resizable]` | boolean | true | Enable resizing |
| `[maximizable]` | boolean | false | Maximize toggle |
| `[breakpoints]` | object | — | Responsive width breakpoints |
| `[position]` | string | 'center' | top, bottom, left, right, topleft, etc. |
| `[dismissableMask]` | boolean | false | Close on backdrop click |
| `[closeOnEscape]` | boolean | true | Close on Escape key |

## ConfirmDialog

```typescript
import { ConfirmationService } from 'primeng/api';
import { ConfirmDialog } from 'primeng/confirmdialog';

@Component({
  imports: [ConfirmDialog, ButtonModule],
  providers: [ConfirmationService],
  template: `
    <p-confirmdialog />
    <p-button label="Delete" severity="danger" icon="pi pi-trash" (onClick)="confirm()" />
  `
})
export class Demo {
  constructor(private confirmationService: ConfirmationService) {}

  confirm() {
    this.confirmationService.confirm({
      message: 'Do you want to delete this record?',
      header: 'Delete Confirmation',
      icon: 'pi pi-info-circle',
      acceptButtonProps: { label: 'Delete', severity: 'danger' },
      rejectButtonProps: { label: 'Cancel', severity: 'secondary', outlined: true },
      accept: () => {
        this.messageService.add({ severity: 'info', summary: 'Confirmed', detail: 'Record deleted' });
      },
      reject: () => {}
    });
  }
}
```

## ConfirmPopup

Inline confirmation attached to a target element:

```html
<p-confirmpopup />
<p-button label="Delete" (onClick)="confirmDelete($event)" />
```

```typescript
confirmDelete(event: Event) {
  this.confirmationService.confirm({
    target: event.target as EventTarget,
    message: 'Are you sure?',
    accept: () => this.delete()
  });
}
```

## Toast

```typescript
import { MessageService } from 'primeng/api';
import { Toast } from 'primeng/toast';

@Component({
  imports: [Toast],
  providers: [MessageService],
  template: `<p-toast />`
})
export class Demo {
  constructor(private messageService: MessageService) {}

  showSuccess() {
    this.messageService.add({
      severity: 'success',
      summary: 'Success',
      detail: 'Operation completed',
      life: 3000            // Auto-dismiss after 3s
    });
  }

  showSticky() {
    this.messageService.add({
      severity: 'warn',
      summary: 'Warning',
      detail: 'This stays until dismissed',
      sticky: true
    });
  }

  showMultiple() {
    this.messageService.addAll([
      { severity: 'success', summary: 'Saved', detail: 'Item 1' },
      { severity: 'success', summary: 'Saved', detail: 'Item 2' }
    ]);
  }

  clearAll() {
    this.messageService.clear();
  }
}
```

### Toast Positions

```html
<p-toast position="top-right" />          <!-- default -->
<p-toast position="top-left" />
<p-toast position="top-center" />
<p-toast position="bottom-right" />
<p-toast position="bottom-left" />
<p-toast position="bottom-center" />
<p-toast position="center" />
```

### Multiple Toast Groups

```html
<p-toast />
<p-toast position="top-left" key="tl" />
```

```typescript
// Target specific toast
this.messageService.add({ key: 'tl', severity: 'info', summary: 'Info', detail: 'Top left toast' });
```

### Severities

`success`, `info`, `warn`, `error`, `secondary`, `contrast`

## Drawer (Sidebar)

```html
<p-drawer
  [(visible)]="drawerVisible"
  [position]="'right'"
  [style]="{ width: '400px' }"
  header="Settings"
>
  <ng-template #content>
    <div class="flex flex-col gap-4">
      <p>Drawer content here</p>
    </div>
  </ng-template>
</p-drawer>
```

| Position | Value |
|----------|-------|
| Left (default) | `'left'` |
| Right | `'right'` |
| Top | `'top'` |
| Bottom | `'bottom'` |
| Full screen | `[fullScreen]="true"` |

## Popover

Overlay anchored to a target element:

```html
<p-button (onClick)="op.toggle($event)" label="Show" />

<p-popover #op>
  <div class="flex flex-col gap-4" style="width: 300px;">
    <h5>Quick Stats</h5>
    <div class="flex gap-4">
      <span>Users: 1,234</span>
      <span>Revenue: $12,345</span>
    </div>
  </div>
</p-popover>
```

## Tooltip

```html
<!-- Directive -->
<input pInputText pTooltip="Enter your name" tooltipPosition="top" />
<p-button label="Save" pTooltip="Save changes" tooltipPosition="bottom" />

<!-- Positions: top, bottom, left, right -->
<!-- Show on focus -->
<input pInputText pTooltip="Help text" tooltipEvent="focus" />
```
