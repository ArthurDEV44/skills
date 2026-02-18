# Forms

## Reactive Forms (Preferred)

### Basic Setup

```typescript
import { Component, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';

@Component({
  selector: 'app-login',
  imports: [ReactiveFormsModule],
  template: `
    <form [formGroup]="form" (ngSubmit)="onSubmit()">
      <input formControlName="email" placeholder="Email" />
      @if (form.controls.email.errors?.['required'] && form.controls.email.touched) {
        <span class="error">Email is required</span>
      }
      @if (form.controls.email.errors?.['email']) {
        <span class="error">Invalid email format</span>
      }

      <input formControlName="password" type="password" placeholder="Password" />

      <button type="submit" [disabled]="form.invalid">Login</button>
    </form>
  `,
})
export class LoginComponent {
  private fb = inject(FormBuilder);

  form = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(8)]],
  });

  onSubmit() {
    if (this.form.valid) {
      console.log(this.form.value);
      // { email: string | null, password: string | null }
    }
  }
}
```

### Typed Forms (Strict by Default)

```typescript
const form = this.fb.group({
  name: ['', Validators.required],       // FormControl<string | null>
  age: [0],                               // FormControl<number | null>
  active: this.fb.nonNullable.control(true), // FormControl<boolean> (non-nullable)
});

// Fully typed access
const name: string | null = form.value.name;
```

Use `fb.nonNullable` for controls that should never be null.

### Nested FormGroups

```typescript
form = this.fb.group({
  name: ['', Validators.required],
  address: this.fb.group({
    street: [''],
    city: [''],
    zip: ['', Validators.pattern(/^\d{5}$/)],
  }),
});
```

Template:

```html
<form [formGroup]="form">
  <input formControlName="name" />
  <div formGroupName="address">
    <input formControlName="street" />
    <input formControlName="city" />
    <input formControlName="zip" />
  </div>
</form>
```

### FormArray (Dynamic Fields)

```typescript
@Component({
  imports: [ReactiveFormsModule],
  template: `
    <form [formGroup]="form">
      <div formArrayName="aliases">
        @for (alias of aliases.controls; track $index) {
          <input [formControlName]="$index" />
          <button type="button" (click)="removeAlias($index)">Remove</button>
        }
      </div>
      <button type="button" (click)="addAlias()">Add Alias</button>
    </form>
  `,
})
export class ProfileComponent {
  private fb = inject(FormBuilder);

  form = this.fb.group({
    aliases: this.fb.array([this.fb.control('')]),
  });

  get aliases() {
    return this.form.controls.aliases;
  }

  addAlias() {
    this.aliases.push(this.fb.control(''));
  }

  removeAlias(index: number) {
    this.aliases.removeAt(index);
  }
}
```

### Custom Validators

```typescript
import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';

// Sync validator
export function forbiddenNameValidator(nameRe: RegExp): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const forbidden = nameRe.test(control.value);
    return forbidden ? { forbiddenName: { value: control.value } } : null;
  };
}

// Async validator
@Injectable({ providedIn: 'root' })
export class UniqueEmailValidator {
  private http = inject(HttpClient);

  validate(control: AbstractControl): Observable<ValidationErrors | null> {
    return this.http.get<boolean>(`/api/check-email?email=${control.value}`).pipe(
      map(exists => exists ? { emailTaken: true } : null),
      catchError(() => of(null)),
    );
  }
}

// Usage
email: ['', {
  validators: [Validators.required, forbiddenNameValidator(/admin/i)],
  asyncValidators: [this.emailValidator.validate.bind(this.emailValidator)],
  updateOn: 'blur',
}],
```

### Form Utilities

```typescript
// Patch partial values
form.patchValue({ name: 'Updated Name' });

// Reset form
form.reset();

// Mark all as touched (show validation)
form.markAllAsTouched();

// Get raw value (ignores disabled controls)
const raw = form.getRawValue();
```
