# coss ui - Form Components Reference

## Table of Contents
- [Input](#input)
- [Textarea](#textarea)
- [Field](#field)
- [Fieldset](#fieldset)
- [Form](#form)
- [Checkbox](#checkbox)
- [CheckboxGroup](#checkbox-group)
- [RadioGroup](#radio-group)
- [Select](#select)
- [Combobox](#combobox)
- [Autocomplete](#autocomplete)
- [Switch](#switch)
- [Slider](#slider)
- [NumberField](#number-field)

---

## Input

CLI: `pnpm dlx shadcn@latest add @coss/input`

```tsx
import { Input } from "@/components/ui/input"
```

**Props:**
| Prop | Type | Default |
|------|------|---------|
| `size` | `"sm" \| "default" \| "lg" \| number` | `"default"` |
| `unstyled` | `boolean` | `false` |

Sizes: sm (28px), default (32px), lg (36px). Use `lg` to match shadcn/ui.

```tsx
<Input aria-label="Email" placeholder="[email protected]" type="email" />
<Input size="sm" placeholder="Small" type="text" />
<Input disabled placeholder="Disabled" type="text" />
<Input type="file" />
```

Docs: https://coss.com/ui/docs/components/input

---

## Textarea

CLI: `pnpm dlx shadcn@latest add @coss/textarea`

```tsx
import { Textarea } from "@/components/ui/textarea"
```

**Props:**
| Prop | Type | Default |
|------|------|---------|
| `size` | `"sm" \| "default" \| "lg" \| number` | `"default"` |
| `unstyled` | `boolean` | `false` |

```tsx
<Textarea placeholder="Type your message here" />
<Textarea size="sm" placeholder="Small textarea" />
<Textarea size="lg" placeholder="Large textarea" />
```

Docs: https://coss.com/ui/docs/components/textarea

---

## Field

CLI: `pnpm dlx shadcn@latest add @coss/field`

```tsx
import { Field, FieldDescription, FieldError, FieldLabel, FieldValidity } from "@/components/ui/field"
```

**Sub-components:** Field, FieldLabel, FieldItem, FieldDescription, FieldError, FieldControl, FieldValidity

```tsx
<Field>
  <FieldLabel>Name</FieldLabel>
  <Input type="text" placeholder="Enter your name" />
  <FieldDescription>Visible on your profile</FieldDescription>
</Field>

// With validation error
<Field name="email">
  <FieldLabel>Email</FieldLabel>
  <Input required type="email" />
  <FieldError>Please enter a valid email.</FieldError>
</Field>
```

Works with: Input, Textarea, Select, Combobox, Autocomplete, Checkbox, CheckboxGroup, RadioGroup, Switch, Slider, NumberField.

Docs: https://coss.com/ui/docs/components/field

---

## Fieldset

CLI: `pnpm dlx shadcn@latest add @coss/fieldset`

```tsx
import { Fieldset, FieldsetLegend } from "@/components/ui/fieldset"
```

```tsx
<Fieldset>
  <FieldsetLegend>Billing Details</FieldsetLegend>
  <Field>
    <FieldLabel>Company</FieldLabel>
    <Input placeholder="Enter company name" type="text" />
  </Field>
</Fieldset>
```

Docs: https://coss.com/ui/docs/components/fieldset

---

## Form

CLI: `pnpm dlx shadcn@latest add @coss/form`

Dependencies: `@base-ui/react`, `zod`

```tsx
import { Form } from "@/components/ui/form"
```

**Props:** `onSubmit`, `errors` (Record<string, string | string[]>), `className`

```tsx
<Form onSubmit={(e) => { e.preventDefault(); /* handle */ }}>
  <Field name="email">
    <FieldLabel>Email</FieldLabel>
    <Input name="email" type="email" required />
    <FieldError>Please enter a valid email.</FieldError>
  </Field>
  <Button type="submit">Submit</Button>
</Form>
```

Supports Zod validation with `errors` prop for server-side error display.

Docs: https://coss.com/ui/docs/components/form

---

## Checkbox

CLI: `pnpm dlx shadcn@latest add @coss/checkbox`

```tsx
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
```

```tsx
<Label>
  <Checkbox />
  Accept terms and conditions
</Label>

<Checkbox defaultChecked disabled />
```

Card style:
```tsx
<Label className="flex items-center gap-6 rounded-lg border p-3 hover:bg-accent/50 has-data-checked:border-primary/48">
  <Checkbox />
  <span>Enable feature</span>
</Label>
```

Docs: https://coss.com/ui/docs/components/checkbox

---

## Checkbox Group

CLI: `pnpm dlx shadcn@latest add @coss/checkbox-group`

```tsx
import { Checkbox } from "@/components/ui/checkbox"
import { CheckboxGroup } from "@/components/ui/checkbox-group"
import { Label } from "@/components/ui/label"
```

**Props:** `aria-label`, `defaultValue` (string[]), `value` (string[]), `onValueChange`, `allValues` (for parent), `disabled`

```tsx
<CheckboxGroup aria-label="Select frameworks" defaultValue={["next"]}>
  <Label><Checkbox value="next" />Next.js</Label>
  <Label><Checkbox value="vite" />Vite</Label>
  <Label><Checkbox value="astro" />Astro</Label>
</CheckboxGroup>
```

Parent checkbox (select all): use `parent` prop on Checkbox and `allValues` on CheckboxGroup.

Docs: https://coss.com/ui/docs/components/checkbox-group

---

## Radio Group

CLI: `pnpm dlx shadcn@latest add @coss/radio-group`

```tsx
import { Label } from "@/components/ui/label"
import { Radio, RadioGroup } from "@/components/ui/radio-group"
```

`RadioGroupItem` is a legacy alias for `Radio`.

```tsx
<RadioGroup defaultValue="next">
  <Label><Radio value="next" /> Next.js</Label>
  <Label><Radio value="vite" /> Vite</Label>
  <Label><Radio value="astro" /> Astro</Label>
</RadioGroup>
```

Supports card style, disabled items, and form integration.

Docs: https://coss.com/ui/docs/components/radio-group

---

## Select

CLI: `pnpm dlx shadcn@latest add @coss/select`

```tsx
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "@/components/ui/select"
```

**Important:** Must pass `items` prop (array or record) on `Select` root.

**SelectTrigger props:** `size` ("sm" | "default" | "lg")
**SelectPopup props:** `alignItemWithTrigger` (boolean, default true), `sideOffset` (number)

```tsx
const items = [
  { label: "Next.js", value: "next" },
  { label: "Vite", value: "vite" },
];

<Select aria-label="Framework" defaultValue="next" items={items}>
  <SelectTrigger>
    <SelectValue />
  </SelectTrigger>
  <SelectPopup>
    {items.map(({ label, value }) => (
      <SelectItem key={value} value={value}>{label}</SelectItem>
    ))}
  </SelectPopup>
</Select>
```

Groups: `SelectGroup`, `SelectGroupLabel`, `SelectSeparator`
Multiple: `<Select multiple defaultValue={["js", "ts"]}>`

Docs: https://coss.com/ui/docs/components/select

---

## Combobox

CLI: `pnpm dlx shadcn@latest add @coss/combobox`

```tsx
import {
  Combobox, ComboboxEmpty, ComboboxInput, ComboboxItem, ComboboxList, ComboboxPopup,
  ComboboxChip, ComboboxChips, ComboboxValue,
  ComboboxGroup, ComboboxGroupLabel, ComboboxCollection, ComboboxSeparator, ComboboxTrigger,
} from "@/components/ui/combobox"
```

**Combobox props:** `items`, `multiple`, `open`, `disabled`, `autoHighlight`, `defaultValue`
**ComboboxInput props:** `size` ("sm"|"default"|"lg"), `startAddon`, `showTrigger` (default true), `showClear`

```tsx
<Combobox items={items}>
  <ComboboxInput placeholder="Select an item..." />
  <ComboboxPopup>
    <ComboboxEmpty>No results found.</ComboboxEmpty>
    <ComboboxList>
      {(item) => (
        <ComboboxItem key={item.value} value={item}>{item.label}</ComboboxItem>
      )}
    </ComboboxList>
  </ComboboxPopup>
</Combobox>
```

Multiple with chips:
```tsx
<Combobox items={items} multiple>
  <ComboboxChips>
    <ComboboxValue>
      {(value) => (
        <>
          {value?.map((item) => (
            <ComboboxChip key={item.value} aria-label={item.label}>{item.label}</ComboboxChip>
          ))}
          <ComboboxInput placeholder={value.length > 0 ? undefined : "Select..."} />
        </>
      )}
    </ComboboxValue>
  </ComboboxChips>
  <ComboboxPopup>
    <ComboboxEmpty>No items found.</ComboboxEmpty>
    <ComboboxList>
      {(item) => <ComboboxItem key={item.value} value={item}>{item.label}</ComboboxItem>}
    </ComboboxList>
  </ComboboxPopup>
</Combobox>
```

Custom trigger (button style):
```tsx
<Combobox defaultValue={items[0]} items={items}>
  <ComboboxTrigger render={<Button className="w-full justify-between" variant="outline" />}>
    <ComboboxValue />
    <ChevronsUpDownIcon />
  </ComboboxTrigger>
  <ComboboxPopup>
    <div className="border-b p-2">
      <ComboboxInput placeholder="Search..." showTrigger={false} startAddon={<SearchIcon />} />
    </div>
    <ComboboxEmpty>No items found.</ComboboxEmpty>
    <ComboboxList>
      {(item) => <ComboboxItem key={item.value} value={item}>{item.label}</ComboboxItem>}
    </ComboboxList>
  </ComboboxPopup>
</Combobox>
```

Docs: https://coss.com/ui/docs/components/combobox

---

## Autocomplete

CLI: `pnpm dlx shadcn@latest add @coss/autocomplete`

```tsx
import {
  Autocomplete, AutocompleteEmpty, AutocompleteInput, AutocompleteItem,
  AutocompleteList, AutocompletePopup, AutocompleteGroup, AutocompleteGroupLabel,
  AutocompleteCollection, AutocompleteSeparator, AutocompleteStatus, useAutocompleteFilter,
} from "@/components/ui/autocomplete"
```

**Autocomplete props:** `items`, `open`, `disabled`, `autoHighlight`, `mode` ("both" for inline), `filter`, `limit`, `value`, `onValueChange`, `itemToStringValue`
**AutocompleteInput props:** `size`, `startAddon`, `showTrigger`, `showClear`

Difference from Combobox: Autocomplete is for filtering/searching, Combobox is for selecting from a list.

```tsx
<Autocomplete items={items}>
  <AutocompleteInput placeholder="Search..." />
  <AutocompletePopup>
    <AutocompleteEmpty>No results found.</AutocompleteEmpty>
    <AutocompleteList>
      {(item) => <AutocompleteItem key={item.value} value={item}>{item.label}</AutocompleteItem>}
    </AutocompleteList>
  </AutocompletePopup>
</Autocomplete>
```

Async search: set `filter={null}`, manage items via state, use debounce.

Docs: https://coss.com/ui/docs/components/autocomplete

---

## Switch

CLI: `pnpm dlx shadcn@latest add @coss/switch`

```tsx
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
```

Size via CSS variable `--thumb-size`. Default: `[--thumb-size:--spacing(5)] sm:[--thumb-size:--spacing(4)]`

```tsx
<Label>
  <Switch />
  Marketing emails
</Label>

<Label>
  <Switch disabled />
  Disabled switch
</Label>
```

Card style:
```tsx
<Label className="flex items-center gap-6 rounded-lg border p-3 hover:bg-accent/50 has-data-checked:border-primary/48 has-data-checked:bg-accent/50">
  <div className="flex flex-col gap-1">
    <p>Enable notifications</p>
    <p className="text-muted-foreground text-xs">Description here.</p>
  </div>
  <Switch defaultChecked id={id} />
</Label>
```

Docs: https://coss.com/ui/docs/components/switch

---

## Slider

CLI: `pnpm dlx shadcn@latest add @coss/slider`

```tsx
import { Slider, SliderValue } from "@/components/ui/slider"
```

**Important:** Always pass values as arrays: `defaultValue={[50]}`. Uses Base UI multiple value approach.

```tsx
<Slider defaultValue={50} />

<Field>
  <Slider defaultValue={50}>
    <div className="mb-2 flex items-center justify-between gap-1">
      <FieldLabel>Opacity</FieldLabel>
      <SliderValue />
    </div>
  </Slider>
</Field>
```

Docs: https://coss.com/ui/docs/components/slider

---

## Number Field

CLI: `pnpm dlx shadcn@latest add @coss/number-field`

```tsx
import {
  NumberField, NumberFieldDecrement, NumberFieldGroup,
  NumberFieldIncrement, NumberFieldInput, NumberFieldScrubArea,
} from "@/components/ui/number-field"
```

**NumberField props:** `size` ("sm"|"default"|"lg"), `defaultValue`, `min`, `max`, `step`, `disabled`, `format`

```tsx
<NumberField defaultValue={0}>
  <NumberFieldGroup>
    <NumberFieldDecrement />
    <NumberFieldInput />
    <NumberFieldIncrement />
  </NumberFieldGroup>
</NumberField>

// With scrub area and range
<NumberField defaultValue={5} min={0} max={10}>
  <NumberFieldScrubArea label="Quantity" />
  <NumberFieldGroup>
    <NumberFieldDecrement />
    <NumberFieldInput />
    <NumberFieldIncrement />
  </NumberFieldGroup>
</NumberField>

// Currency formatting
<NumberField defaultValue={0} format={{ currency: "USD", style: "currency" }}>
  <NumberFieldGroup>
    <NumberFieldDecrement />
    <NumberFieldInput />
    <NumberFieldIncrement />
  </NumberFieldGroup>
</NumberField>
```

Docs: https://coss.com/ui/docs/components/number-field
