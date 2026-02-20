---
name: react-email-best-practices
description: "React Email framework for building and sending emails with React components, Tailwind CSS styling, and cross-client compatibility. Use when: (1) Creating email templates with React Email components (Html, Head, Body, Container, Section, Row, Column, Text, Button, Link, Img, Heading, Preview, Hr, Font, Tailwind, Markdown, CodeBlock, CodeInline), (2) Styling emails with Tailwind CSS and pixelBasedPreset, (3) Rendering emails to HTML with render(), toPlainText(), pretty(), (4) Sending emails via Resend, Nodemailer, SendGrid, Postmark, AWS SES, MailerSend, Plunk, or Loops, (5) Building responsive email layouts with Row/Column, (6) Setting up React Email projects with CLI (dev, export, build), (7) Fixing email client compatibility issues (Outlook, Gmail clipping), (8) Creating reusable email layouts and shared components, (9) Any @react-email/components or @react-email/render imports."
---

# React Email Best Practices

## Quick Setup

```bash
# New standalone project
npx create-email@latest

# Add to existing project
npm install @react-email/components react react-dom
```

Add dev script to `package.json`:
```json
{
  "scripts": {
    "email": "email dev --dir emails --port 3000"
  }
}
```

## Project Structure

```
my-project/
  emails/
    welcome.tsx
    reset-password.tsx
    order-confirmation.tsx
    components/
      layout.tsx
      header.tsx
      footer.tsx
  package.json
  tsconfig.json
```

## Basic Email Template

```tsx
import {
  Html,
  Head,
  Preview,
  Body,
  Container,
  Section,
  Heading,
  Text,
  Button,
  Hr,
  Tailwind,
  pixelBasedPreset,
} from '@react-email/components';

interface WelcomeEmailProps {
  name: string;
  confirmUrl: string;
}

export default function WelcomeEmail({ name, confirmUrl }: WelcomeEmailProps) {
  return (
    <Html lang="en">
      <Head />
      <Preview>Welcome to Acme - Confirm your email</Preview>
      <Tailwind config={{ presets: [pixelBasedPreset] }}>
        <Body className="bg-gray-100 font-sans">
          <Container className="mx-auto my-10 max-w-xl bg-white rounded-lg p-8">
            <Heading className="text-2xl font-bold text-gray-900 m-0 mb-6">
              Welcome, {name}!
            </Heading>
            <Text className="text-base leading-6 text-gray-600 m-0 mb-4">
              Thanks for signing up. Confirm your email to get started.
            </Text>
            <Button
              href={confirmUrl}
              className="bg-blue-600 text-white px-6 py-3 rounded-md font-semibold no-underline text-center block box-border"
            >
              Confirm Email
            </Button>
            <Hr className="border-gray-200 my-6" />
            <Text className="text-sm text-gray-400 m-0">
              If you didn't sign up, ignore this email.
            </Text>
          </Container>
        </Body>
      </Tailwind>
    </Html>
  );
}

WelcomeEmail.PreviewProps = {
  name: 'John Doe',
  confirmUrl: 'https://example.com/confirm?token=abc123',
} satisfies WelcomeEmailProps;
```

## Essential Components

| Component | Renders As | Purpose |
|-----------|-----------|---------|
| `Html` | `<html>` | Root element. Always set `lang`. |
| `Head` | `<head>` | Meta info, `<Font>`, `<title>`. |
| `Body` | `<body>` | Visible content wrapper. |
| `Container` | Centered `<table>` | Centers content, max-width (default 580px). |
| `Section` | `<table>` | Groups content vertically. |
| `Row` | `<tr>` | Horizontal layout row. Contains `<Column>`. |
| `Column` | `<td>` | Column within a Row. |
| `Preview` | Hidden text | Inbox preview snippet. Place first in Body. |
| `Heading` | `<h1>`-`<h6>` | Headings. Use `as` prop for level. |
| `Text` | `<p>` | Paragraphs. Has default margins — use `m-0` to reset. |
| `Button` | `<a>` (styled) | CTA buttons with Outlook padding fix built-in. |
| `Link` | `<a>` | Hyperlinks. Default `target="_blank"`. |
| `Img` | `<img>` | Images. Always set `width`, `height`, `alt`. |
| `Hr` | `<hr>` | Dividers. |
| `Font` | `@font-face` | Custom web fonts. Place inside `<Head>`. |
| `Tailwind` | Wrapper | Enables Tailwind CSS classes. |
| `Markdown` | HTML | Renders markdown string to email-safe HTML. |
| `CodeBlock` | Syntax-highlighted code | Uses Prism.js themes. |
| `CodeInline` | `<code>` | Inline code snippets. |

For detailed component props and API, see [references/components.md](references/components.md).

## Critical Rules

### ALWAYS do these:
1. **Use `pixelBasedPreset`** with Tailwind — email clients do NOT support `rem` units
2. **Set `lang` on `<Html>`** — `<Html lang="en">`
3. **Place `<Preview>` early** in `<Body>` — it sets inbox snippet text
4. **Set `width` and `height` on all `<Img>`** — prevents layout shifts
5. **Use absolute URLs** for image `src` — host on CDN, never inline base64
6. **Use PNG or JPEG** for images — NOT SVG or WEBP
7. **Add `box-border`** to `<Button>` — prevents padding expansion
8. **Add `no-underline`** to `<Button>` — removes link underline
9. **Reset margins** on `<Text>` and `<Heading>` with `m-0` when needed
10. **Only import components you use** — avoid unused imports
11. **Always provide `fallbackFontFamily`** in `<Font>` — most clients ignore web fonts
12. **Define `PreviewProps`** on every email component for dev server previews

### NEVER do these:
1. **Never use `display: flex` or `display: grid`** — use `<Row>` / `<Column>` instead
2. **Never use `rem` units** without `pixelBasedPreset`
3. **Never use SVG or WEBP images** — poor email client support
4. **Never use CSS variables** (`var(--...)`) — not supported
5. **Never use `position: absolute/fixed`** — not supported
6. **Never use `background-image`** — very limited support
7. **Never use `dark:` Tailwind variant** — not supported in email clients
8. **Never rely on media queries alone** — Outlook desktop ignores them

## Styling with Tailwind CSS

```tsx
import { Tailwind, pixelBasedPreset } from '@react-email/components';

<Tailwind
  config={{
    presets: [pixelBasedPreset],
    theme: {
      extend: {
        colors: {
          brand: '#007bff',
          brandDark: '#0056b3',
        },
        fontFamily: {
          sans: ['Inter', 'Arial', 'sans-serif'],
        },
      },
    },
  }}
>
  {/* All email content here */}
</Tailwind>
```

Tailwind classes are converted to inline styles at render time. Responsive breakpoints (`sm:`, `md:`) work only when `<Html>` and `<Head>` are present, and only in clients that support `<style>` tags.

## Multi-Column Layouts

```tsx
<Section>
  <Row>
    <Column className="w-1/2 p-2 align-top">
      <Text className="m-0">Left column</Text>
    </Column>
    <Column className="w-1/2 p-2 align-top">
      <Text className="m-0">Right column</Text>
    </Column>
  </Row>
</Section>
```

These render as `<table>` / `<tr>` / `<td>` under the hood for cross-client compatibility.

## Rendering and Sending

```tsx
import { render, toPlainText, pretty } from '@react-email/components';

// Render to HTML (async)
const html = await render(<WelcomeEmail name="John" confirmUrl="..." />);

// Plain text version for accessibility
const text = toPlainText(html);

// Pretty-printed HTML for debugging
const prettyHtml = await pretty(await render(<WelcomeEmail name="John" confirmUrl="..." />));
```

For integration examples with Resend, Nodemailer, SendGrid, Postmark, AWS SES, MailerSend, Plunk, and Loops, see [references/integrations.md](references/integrations.md).

## Reusable Layout Pattern

```tsx
// emails/components/layout.tsx
import {
  Html, Head, Preview, Body, Container, Hr, Text, Tailwind, pixelBasedPreset,
} from '@react-email/components';

interface EmailLayoutProps {
  preview: string;
  children: React.ReactNode;
}

export function EmailLayout({ preview, children }: EmailLayoutProps) {
  return (
    <Html lang="en">
      <Head />
      <Preview>{preview}</Preview>
      <Tailwind config={{ presets: [pixelBasedPreset] }}>
        <Body className="bg-gray-100 font-sans">
          <Container className="mx-auto my-10 max-w-xl bg-white rounded-lg p-8">
            {children}
            <Hr className="border-gray-200 my-6" />
            <Text className="text-xs text-gray-400 text-center m-0">
              Company Inc. | 123 Main St
            </Text>
          </Container>
        </Body>
      </Tailwind>
    </Html>
  );
}
```

Then use in individual emails:
```tsx
import { EmailLayout } from './components/layout';
import { Heading, Text, Button } from '@react-email/components';

export default function ResetPassword({ resetUrl }: { resetUrl: string }) {
  return (
    <EmailLayout preview="Reset your password">
      <Heading className="text-2xl font-bold m-0 mb-4">Reset Password</Heading>
      <Text className="text-base text-gray-600 m-0 mb-4">
        Click the button below to reset your password.
      </Text>
      <Button
        href={resetUrl}
        className="bg-blue-600 text-white px-6 py-3 rounded-md font-semibold no-underline text-center block box-border"
      >
        Reset Password
      </Button>
    </EmailLayout>
  );
}
```

## CLI Commands

```bash
npx react-email dev --dir ./emails --port 3000   # Dev server with hot reload
npx react-email export --dir ./emails --outDir ./out --pretty  # Export to HTML
npx react-email build --dir ./emails              # Build preview app
npx react-email start                             # Start built preview app
```

## Email Client Compatibility

For detailed compatibility tables, Outlook workarounds, Gmail clipping mitigation, and dark mode strategies, see [references/compatibility.md](references/compatibility.md).

### Key Gotchas
- **Outlook desktop** uses Word's rendering engine — no flexbox, no grid, no border-radius on many elements, no box-shadow, no background-image
- **Gmail clips** emails larger than ~102KB — keep emails concise
- **`dark:` variant** does not work — apply dark colors directly if needed
- **Web fonts** only work in Apple Mail, iOS Mail, some Android clients, Thunderbird — always provide fallbacks
