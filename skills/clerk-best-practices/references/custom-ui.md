# Custom UI and Appearance

## Appearance Prop

Customize any Clerk component with the `appearance` prop:

```tsx
<SignIn
  appearance={{
    variables: {
      colorPrimary: '#0000ff',
      borderRadius: '0.5rem',
    },
    layout: {
      logoImageUrl: '/logo.png',
      socialButtonsVariant: 'iconButton',
    },
  }}
/>
```

## Global Appearance (ClerkProvider)

Apply to all components at once:

```tsx
<ClerkProvider
  appearance={{
    variables: {
      colorPrimary: '#6366f1',
      colorBackground: '#ffffff',
      borderRadius: '0.5rem',
    },
    layout: {
      logoImageUrl: '/logo.png',
      socialButtonsVariant: 'iconButton',
      socialButtonsPlacement: 'bottom',
    },
  }}
>
  {children}
</ClerkProvider>
```

## Variables (Colors, Typography, Borders)

| Property | Description |
|----------|-------------|
| `colorPrimary` | Primary brand color |
| `colorBackground` | Background color |
| `colorText` | Primary text color |
| `colorInputBackground` | Input field background |
| `colorInputText` | Input field text |
| `borderRadius` | Border radius (default `0.375rem`) |
| `fontFamily` | Custom font family |
| `fontSize` | Base font size |

## Layout Options

| Property | Description |
|----------|-------------|
| `logoImageUrl` | Custom logo URL |
| `socialButtonsVariant` | `'blockButton'` \| `'iconButton'` \| `'auto'` |
| `socialButtonsPlacement` | `'top'` \| `'bottom'` |

## shadcn/ui Theme

If using shadcn/ui, apply the official shadcn theme:

```tsx
import { shadcn } from '@clerk/themes'

<ClerkProvider
  appearance={{
    baseTheme: shadcn,
  }}
>
  {children}
</ClerkProvider>
```

Import the shadcn CSS:

```css
/* global.css */
@import 'tailwindcss';
@import '@clerk/themes/shadcn.css';
```

## Dark Theme

```tsx
import { dark } from '@clerk/themes'

<ClerkProvider appearance={{ baseTheme: dark }}>
  {children}
</ClerkProvider>
```

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| Colors not applying | Use `colorPrimary` not `primaryColor` |
| Logo not showing | Put `logoImageUrl` inside `layout: {}` |
| Theme not applying | Use `baseTheme` not `theme` at ClerkProvider level |
| CSS conflicts | Use `appearance` prop, not direct CSS overrides |

[Docs](https://clerk.com/docs/nextjs/guides/customizing-clerk/appearance-prop/overview)
