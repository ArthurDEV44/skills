# React Email Components — Detailed API Reference

## Structural Components

### Html
Root wrapper element. Renders as `<html>`.

**Props:**
- `lang` (string) — Language attribute. Always set, e.g., `"en"`, `"fr"`, `"es"`.
- `dir` (string) — Text direction: `"ltr"` or `"rtl"`.
- All standard HTML `<html>` attributes.

```tsx
<Html lang="en" dir="ltr">...</Html>
```

### Head
Renders as `<head>`. Contains meta information, `<Font>`, `<title>`.

**Props:** All standard HTML `<head>` attributes.

```tsx
<Head>
  <title>Welcome Email</title>
  <Font fontFamily="Roboto" fallbackFontFamily="Verdana"
    webFont={{ url: "https://fonts.gstatic.com/s/roboto/v27/KFOmCnqEu92Fr1Mu4mxKKTU1Kg.woff2", format: "woff2" }}
    fontWeight={400} fontStyle="normal" />
</Head>
```

### Body
Renders as `<body>`. Contains all visible email content.

**Props:** `style`, `className`, all standard `<body>` attributes.

```tsx
<Body className="bg-gray-100 font-sans">...</Body>
```

### Container
Centers content horizontally with a max-width. Renders as a centered `<table>`.

**Props:** `style`, `className`, standard HTML div attributes.
**Default max-width:** 580px. Override with `style={{ maxWidth: "600px" }}` or Tailwind `className="max-w-xl"`.

```tsx
<Container className="mx-auto my-10 max-w-xl bg-white rounded-lg p-8">...</Container>
```

### Section
Layout division. Renders as a `<table>` with `<tbody>`.

**Props:** `style`, `className`, standard HTML attributes.

```tsx
<Section className="px-8 py-10">...</Section>
```

### Row
Horizontal layout row. Renders as a `<tr>`.

**Props:** `style`, `className`, standard `<tr>` attributes.
**Must contain** `<Column>` children. Used inside `<Section>`.

```tsx
<Row>
  <Column className="w-1/2">Left</Column>
  <Column className="w-1/2">Right</Column>
</Row>
```

### Column
Column within a Row. Renders as `<td>`.

**Props:**
- `align` — Horizontal alignment: `"left"`, `"center"`, `"right"`.
- `valign` — Vertical alignment: `"top"`, `"middle"`, `"bottom"`.
- `width` — Column width (px or percentage).
- `style`, `className`, standard `<td>` attributes.

**Must be a direct child of** `<Row>`.

```tsx
<Column className="w-1/3 p-2 align-top" align="center">Content</Column>
```

---

## Content Components

### Preview
Sets inbox preview text (snippet shown next to subject line in email clients).

**Props:** `children` (string) — The preview text.

**Best practices:**
- Place as early as possible inside `<Body>`.
- Keep under 140 characters.
- Make it action-oriented and meaningful.
- Text is hidden from the rendered email body.

```tsx
<Preview>Welcome to Acme - Get started with your free trial</Preview>
```

### Heading
Renders heading elements (`<h1>` through `<h6>`).

**Props:**
- `as` (string, default: `"h1"`) — Heading level: `"h1"` | `"h2"` | `"h3"` | `"h4"` | `"h5"` | `"h6"`.
- `m`, `mx`, `my`, `mt`, `mr`, `mb`, `ml` — Margin shortcuts (CSS values as strings, e.g., `"0"`, `"16px"`).
- `style`, `className`, standard heading attributes.

```tsx
<Heading as="h1" className="text-2xl font-bold text-gray-900 m-0 mb-6">Title</Heading>
<Heading as="h2" m="0" my="16px" style={{ color: "#444" }}>Subtitle</Heading>
```

### Text
Renders a `<p>` paragraph.

**Props:** `style`, `className`, standard `<p>` attributes.
**Note:** Has default browser margins. Use `m-0` to reset.

```tsx
<Text className="text-base leading-6 text-gray-600 m-0 mb-4">Body text here.</Text>
```

### Link
Renders an `<a>` hyperlink.

**Props:**
- `href` (string, required) — URL. Supports `https://`, `mailto:`, `tel:`.
- `target` (string, default: `"_blank"`).
- `style`, `className`, standard `<a>` attributes.

```tsx
<Link href="https://example.com" className="text-blue-600 underline">Visit site</Link>
<Link href="mailto:support@example.com">Email support</Link>
```

### Button
Clickable CTA button styled as an `<a>` tag. Has built-in Outlook padding workarounds.

**Props:**
- `href` (string, required) — URL destination.
- `target` (string) — Link target.
- `style`, `className`, standard `<a>` attributes.

**Critical styling rules:**
- Always add `box-border` — prevents padding from expanding total size.
- Always add `no-underline` — removes link underline styling.
- Use `block` display and `text-center` for consistent cross-client rendering.

```tsx
<Button
  href="https://example.com/action"
  className="bg-blue-600 text-white px-6 py-3 rounded-md font-semibold no-underline text-center block box-border"
>
  Call to Action
</Button>
```

### Img
Displays images. Renders as `<img>`.

**Props:**
- `src` (string, required) — Absolute URL to image. Must be hosted externally (CDN).
- `alt` (string, required) — Alt text for accessibility.
- `width` (string | number) — Image width. Always set.
- `height` (string | number) — Image height. Use `"auto"` for responsive.
- `style`, `className`, standard `<img>` attributes.

**Supported formats:** PNG, JPEG, GIF. **NOT supported:** SVG, WEBP.

```tsx
{/* Responsive image */}
<Img src="https://cdn.example.com/hero.png" alt="Hero" width="600"
  className="w-full max-w-[600px] h-auto" />

{/* Fixed-size image, centered */}
<Img src="https://cdn.example.com/logo.png" alt="Logo" width="150" height="50"
  className="block mx-auto" />
```

### Hr
Horizontal rule / divider. Renders as `<hr>`.

**Props:** `style`, `className`, standard `<hr>` attributes.

```tsx
<Hr className="border-gray-200 my-6" />
```

---

## Specialized Components

### Font
Declares custom web fonts. Must be placed inside `<Head>`.

**Props:**
- `fontFamily` (string, required) — Font name, e.g., `"Roboto"`.
- `fallbackFontFamily` (string | string[], required) — Fallback(s). Safe values: `"Arial"`, `"Helvetica"`, `"Verdana"`, `"Georgia"`, `"Times New Roman"`, `"sans-serif"`, `"serif"`, `"monospace"`.
- `webFont` (object) — `{ url: string, format: "woff" | "woff2" | "truetype" }`.
- `fontWeight` (number | string) — e.g., `400`, `700`, `"bold"`.
- `fontStyle` (string) — `"normal"` | `"italic"`.

**Reality:** Web fonts only work in Apple Mail, iOS Mail, some Android clients, Thunderbird. Gmail, Outlook, Yahoo always use the fallback.

```tsx
<Head>
  <Font
    fontFamily="Inter"
    fallbackFontFamily={["Arial", "sans-serif"]}
    webFont={{ url: "https://fonts.gstatic.com/s/inter/v13/UcC73FwrK3iLTeHuS_fvQtMwCp50KnMa2JL7W0Q5n-wU.woff2", format: "woff2" }}
    fontWeight={400}
    fontStyle="normal"
  />
</Head>
<Body style={{ fontFamily: "Inter, Arial, sans-serif" }}>...</Body>
```

### Tailwind
Wrapper component that enables Tailwind CSS utility classes in email templates.

**Props:**
- `config` (object) — Tailwind configuration: `presets`, `theme`, `theme.extend`.
- `children` (ReactNode) — Content to style.

**Critical:** Always use `pixelBasedPreset`. Email clients do not support `rem` units — this preset converts all rem-based values to pixel equivalents.

**Responsive styles:** Supported (`sm:`, `md:`, `lg:`) but only when `<Html>` and `<Head>` exist in the tree. Injected as `<style>` tags. Not all clients support them (Outlook desktop does not).

```tsx
import { Tailwind, pixelBasedPreset } from '@react-email/components';

<Tailwind config={{
  presets: [pixelBasedPreset],
  theme: {
    extend: {
      colors: { brand: '#007bff', muted: '#6c757d' },
      fontFamily: { sans: ['Inter', 'Arial', 'sans-serif'] },
    },
  },
}}>
  {/* All email content */}
</Tailwind>
```

### Markdown
Renders a Markdown string as email-safe HTML.

**Props:**
- `children` (string, required) — Markdown content.
- `markdownCustomStyles` (object) — Per-element styles: `{ h1: {...}, h2: {...}, p: {...}, a: {...}, codeInline: {...} }`.
- `markdownContainerStyles` (object) — Container wrapper styles.

```tsx
<Markdown
  markdownCustomStyles={{
    h1: { color: "#333", fontSize: "24px" },
    p: { color: "#666", lineHeight: "1.6" },
    a: { color: "#007bff" },
    codeInline: { background: "#f0f0f0", padding: "2px 4px", borderRadius: "3px" },
  }}
  markdownContainerStyles={{ padding: "16px" }}
>
  {`# Hello\n\nThis is **bold** and this is a [link](https://example.com).`}
</Markdown>
```

### CodeBlock
Renders syntax-highlighted code blocks using Prism.js.

**Props:**
- `code` (string, required) — Code to display.
- `language` (string, required) — Prism.js language: `"javascript"`, `"typescript"`, `"json"`, `"python"`, `"html"`, `"css"`, `"bash"`, `"jsx"`, `"tsx"`, etc.
- `theme` (object, required) — Prism.js theme. Import from `@react-email/components`.
- `lineNumbers` (boolean, default: `false`) — Show line numbers.
- `fontFamily` (string) — Font family for code text.

**Available themes:** `dracula`, `nightOwl`, `coldarkCold`, `coldarkDark`, `duotoneDark`, `duotoneLight`, `github`, `materialDark`, `materialLight`, `materialOceanic`, `oneDark`, `oneLight`, `solarizedDarkAtom`, `solarizedLight`, `synthwave84`, `vs`, `vsDark`.

```tsx
import { CodeBlock, dracula } from '@react-email/components';

<CodeBlock
  code={`const greeting = "Hello, World!";\nconsole.log(greeting);`}
  language="javascript"
  theme={dracula}
  lineNumbers
/>
```

### CodeInline
Renders inline code snippets.

**Props:** `style`, `className`, `children` (string), standard `<code>` attributes.

```tsx
<Text className="m-0">
  Use the <CodeInline style={{ background: "#f0f0f0", padding: "2px 4px" }}>render()</CodeInline> function.
</Text>
```
