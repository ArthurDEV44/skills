# PrimeNG Form Components

## InputText

```html
<input pInputText [(ngModel)]="value" placeholder="Enter text" />

<!-- Sizes -->
<input pInputText [(ngModel)]="value" pSize="small" />
<input pInputText [(ngModel)]="value" pSize="large" />

<!-- Disabled / Invalid -->
<input pInputText [(ngModel)]="value" [disabled]="true" />
<input pInputText [(ngModel)]="value" [ngClass]="{ 'ng-invalid ng-dirty': hasError }" />

<!-- Fluid (full width) -->
<input pInputText [(ngModel)]="value" fluid />

<!-- With IconField -->
<p-iconfield iconPosition="left">
  <p-inputicon class="pi pi-search" />
  <input pInputText [(ngModel)]="search" placeholder="Search" />
</p-iconfield>
```

## InputGroup

```html
<p-inputgroup>
  <p-inputgroup-addon><i class="pi pi-user"></i></p-inputgroup-addon>
  <input pInputText placeholder="Username" [(ngModel)]="username" />
</p-inputgroup>

<p-inputgroup>
  <p-inputgroup-addon>$</p-inputgroup-addon>
  <p-inputnumber [(ngModel)]="price" />
  <p-inputgroup-addon>.00</p-inputgroup-addon>
</p-inputgroup>

<p-inputgroup>
  <p-inputgroup-addon><p-checkbox [(ngModel)]="agreed" [binary]="true" /></p-inputgroup-addon>
  <input pInputText placeholder="I agree" />
</p-inputgroup>
```

## Textarea

```html
<textarea pTextarea [(ngModel)]="text" [rows]="5" [cols]="30" [autoResize]="true"></textarea>
```

## InputNumber

```html
<!-- Basic -->
<p-inputnumber [(ngModel)]="qty" />

<!-- Currency -->
<p-inputnumber [(ngModel)]="price" mode="currency" currency="USD" locale="en-US" />

<!-- Decimal -->
<p-inputnumber [(ngModel)]="val" mode="decimal" [minFractionDigits]="2" [maxFractionDigits]="5" />

<!-- With buttons -->
<p-inputnumber [(ngModel)]="qty" [showButtons]="true" [min]="0" [max]="100" />
<p-inputnumber [(ngModel)]="qty" [showButtons]="true" buttonLayout="horizontal"
  incrementButtonIcon="pi pi-plus" decrementButtonIcon="pi pi-minus" />

<!-- Prefix / suffix -->
<p-inputnumber [(ngModel)]="percent" prefix="%" />
<p-inputnumber [(ngModel)]="temp" suffix="°C" />
```

## Select (Dropdown)

```html
<p-select
  [(ngModel)]="selectedCity"
  [options]="cities"
  optionLabel="name"
  optionValue="code"
  placeholder="Select a city"
  [showClear]="true"
  [filter]="true"
  filterBy="name"
/>

<!-- Grouped options -->
<p-select [(ngModel)]="selected" [options]="groupedCities" [group]="true" optionLabel="name" optionGroupLabel="label" optionGroupChildren="items" />

<!-- Custom item template -->
<p-select [(ngModel)]="selected" [options]="countries" optionLabel="name">
  <ng-template #item let-country>
    <div class="flex items-center gap-2">
      <img [src]="country.flag" width="20" />
      <span>{{ country.name }}</span>
    </div>
  </ng-template>
  <ng-template #selectedItem let-country>
    <div class="flex items-center gap-2" *ngIf="country">
      <img [src]="country.flag" width="20" />
      <span>{{ country.name }}</span>
    </div>
  </ng-template>
</p-select>
```

## MultiSelect

```html
<p-multiselect
  [(ngModel)]="selectedCities"
  [options]="cities"
  optionLabel="name"
  placeholder="Select cities"
  display="chip"
  [showClear]="true"
  [filter]="true"
  [maxSelectedLabels]="3"
  selectedItemsLabel="{0} items selected"
/>
```

## AutoComplete

```html
<p-autocomplete
  [(ngModel)]="query"
  [suggestions]="results"
  (completeMethod)="search($event)"
  field="name"
  placeholder="Search..."
  [dropdown]="true"
/>

<!-- Multiple -->
<p-autocomplete
  [(ngModel)]="selectedItems"
  [suggestions]="results"
  (completeMethod)="search($event)"
  field="name"
  [multiple]="true"
/>
```

```typescript
search(event: AutoCompleteCompleteEvent) {
  this.results = this.allItems.filter(item =>
    item.name.toLowerCase().includes(event.query.toLowerCase())
  );
}
```

## DatePicker

```html
<!-- Basic -->
<p-datepicker [(ngModel)]="date" [showIcon]="true" dateFormat="yy-mm-dd" />

<!-- Range -->
<p-datepicker [(ngModel)]="rangeDates" selectionMode="range" [readonlyInput]="true" />

<!-- Multiple -->
<p-datepicker [(ngModel)]="dates" selectionMode="multiple" />

<!-- Time -->
<p-datepicker [(ngModel)]="datetime" [showTime]="true" [showSeconds]="true" />

<!-- Month / Year picker -->
<p-datepicker [(ngModel)]="month" view="month" dateFormat="mm/yy" />
<p-datepicker [(ngModel)]="year" view="year" dateFormat="yy" />

<!-- Min/Max -->
<p-datepicker [(ngModel)]="date" [minDate]="minDate" [maxDate]="maxDate" />

<!-- Inline -->
<p-datepicker [(ngModel)]="date" [inline]="true" />
```

## Checkbox

```html
<!-- Binary -->
<p-checkbox [(ngModel)]="accepted" [binary]="true" label="I accept the terms" inputId="accept" />

<!-- Multiple values -->
@for (cat of categories; track cat) {
  <p-checkbox [(ngModel)]="selectedCategories" [value]="cat" [label]="cat" [inputId]="cat" name="cats" />
}
```

## RadioButton

```html
@for (option of options; track option.key) {
  <div class="flex items-center gap-2">
    <p-radiobutton [(ngModel)]="selected" [value]="option.value" [inputId]="option.key" name="choice" />
    <label [for]="option.key">{{ option.label }}</label>
  </div>
}
```

## ToggleSwitch

```html
<p-toggleswitch [(ngModel)]="active" />
<p-toggleswitch [(ngModel)]="active" onLabel="Yes" offLabel="No" />
```

## Slider

```html
<p-slider [(ngModel)]="value" [min]="0" [max]="100" />
<p-slider [(ngModel)]="rangeValues" [range]="true" />
<p-slider [(ngModel)]="value" orientation="vertical" [style]="{ height: '200px' }" />
```

## Rating

```html
<p-rating [(ngModel)]="score" [cancel]="false" />
<p-rating [(ngModel)]="score" [stars]="10" [readonly]="true" />
```

## Password

```html
<p-password [(ngModel)]="pw" [toggleMask]="true" [feedback]="true" />
<p-password [(ngModel)]="pw" [toggleMask]="true" [feedback]="false" placeholder="No strength meter" />
```

## TreeSelect

```html
<p-treeselect
  [(ngModel)]="selectedNodes"
  [options]="nodes"
  selectionMode="checkbox"
  display="chip"
  placeholder="Select items"
/>
```

## ColorPicker

```html
<p-colorpicker [(ngModel)]="color" />
<p-colorpicker [(ngModel)]="color" [inline]="true" />
```

## InputOtp

```html
<p-inputotp [(ngModel)]="otp" [length]="6" />
<p-inputotp [(ngModel)]="otp" [length]="4" [mask]="true" [integerOnly]="true" />
```

## Reactive Forms

```typescript
@Component({
  imports: [ReactiveFormsModule, InputTextModule, SelectModule, ButtonModule, MessageModule],
  template: `
    <form [formGroup]="form" (ngSubmit)="onSubmit()">
      <div class="flex flex-col gap-2">
        <label for="name">Name</label>
        <input pInputText id="name" formControlName="name"
          [ngClass]="{ 'ng-invalid ng-dirty': form.get('name')?.invalid && form.get('name')?.touched }" />
        @if (form.get('name')?.invalid && form.get('name')?.touched) {
          <p-message severity="error" size="small" variant="simple">Name is required</p-message>
        }
      </div>

      <div class="flex flex-col gap-2">
        <label for="city">City</label>
        <p-select formControlName="city" [options]="cities" optionLabel="name" placeholder="Select" />
      </div>

      <p-button type="submit" label="Submit" [disabled]="form.invalid" />
    </form>
  `
})
export class FormDemo {
  form = new FormGroup({
    name: new FormControl('', Validators.required),
    city: new FormControl(null, Validators.required)
  });

  onSubmit() {
    if (this.form.valid) {
      console.log(this.form.value);
    } else {
      this.form.markAllAsTouched();
    }
  }
}
```

## Template-Driven Forms

```html
<form #myForm="ngForm" (ngSubmit)="onSubmit(myForm)">
  <div class="flex flex-col gap-2">
    <label for="email">Email</label>
    <input pInputText id="email" name="email" [(ngModel)]="email" required email
      #emailModel="ngModel"
      [ngClass]="{ 'ng-invalid ng-dirty': emailModel.invalid && (emailModel.touched || myForm.submitted) }" />
    @if (emailModel.invalid && (emailModel.touched || myForm.submitted)) {
      <p-message severity="error" size="small" variant="simple">Valid email required</p-message>
    }
  </div>

  <div class="flex flex-col gap-2">
    <label for="date">Date</label>
    <p-datepicker name="date" [(ngModel)]="date" required
      #dateModel="ngModel"
      [invalid]="dateModel.invalid && (dateModel.touched || myForm.submitted)" />
    @if (dateModel.invalid && (dateModel.touched || myForm.submitted)) {
      <p-message severity="error" size="small" variant="simple">Date is required</p-message>
    }
  </div>

  <p-button type="submit" label="Submit" severity="secondary" />
</form>
```

## Fluid Layout

Apply `fluid` to make form elements fill their container width:

```html
<div class="flex flex-col gap-4" style="max-width: 400px;">
  <input pInputText fluid placeholder="Full width" />
  <p-select [options]="cities" optionLabel="name" fluid />
  <p-datepicker fluid />
</div>
```
