# Email Client Compatibility Guide

## Supported Email Clients

React Email targets all major email clients: Gmail (web and mobile), Apple Mail, Outlook (desktop, web, mobile), Yahoo Mail, AOL, Samsung Email, Thunderbird, Hey, Fastmail, ProtonMail.

## CSS Feature Support

### Unsupported CSS — AVOID These

| Feature | Why | Alternative |
|---------|-----|-------------|
| `display: flex` | Not supported in Outlook, some webmail | Use `<Row>` and `<Column>` components |
| `display: grid` | Not supported in Outlook, some webmail | Use `<Row>` and `<Column>` components |
| `rem` / `em` units | Ignored by most email clients | Use `pixelBasedPreset` with Tailwind |
| SVG images | Poor support across clients | Use PNG or JPEG |
| WEBP images | Not supported in Outlook | Use PNG or JPEG |
| CSS variables (`var(--...)`) | Not supported | Use literal values |
| `position: absolute/fixed` | Not supported | Use table-based positioning via Row/Column |
| `float` | Inconsistent support | Use `<Row>` / `<Column>` |
| `background-image` | Very limited; Outlook ignores completely | Use `<Img>` component or solid background colors |
| `box-shadow` | Not supported in Outlook | Omit or use border workaround |
| `overflow: hidden` | Not supported in Outlook | Design within bounds |
| `@media` queries | Outlook desktop ignores; limited in Gmail | Use as progressive enhancement only |
| `dark:` variant | Not supported in email clients | Apply dark colors directly |
| `border-radius` on images | Not supported in Outlook desktop | Accept square corners as fallback |

### Safe CSS — Always Works

| Feature | Notes |
|---------|-------|
| `background-color` | Solid colors on any element |
| `color` | Text color |
| `font-family` | With safe fallbacks |
| `font-size` | In px units |
| `font-weight` | `normal`, `bold`, numeric values |
| `line-height` | In px or unitless |
| `text-align` | `left`, `center`, `right` |
| `text-decoration` | `underline`, `none` |
| `padding` | In px units |
| `margin` | In px units (on block elements) |
| `border` | Solid borders with px widths |
| `width` / `height` | In px or percentage |
| `max-width` | In px |
| `border-radius` | On `<td>` and `<div>` — NOT images in Outlook |

## Table-Based Layout Internals

React Email components render as email-safe table markup:

| Component | Renders As |
|-----------|-----------|
| `<Container>` | Centered `<table>` with max-width |
| `<Section>` | `<table>` with `<tbody>` |
| `<Row>` | `<tr>` |
| `<Column>` | `<td>` |

This is transparent to the developer but ensures cross-client compatibility.

---

## Outlook Desktop (2007–2021, Classic Windows)

Outlook desktop uses **Microsoft Word's rendering engine**, which is the most restrictive email renderer.

### What Breaks in Outlook Desktop
- **No flexbox or grid** — always use `<Row>` / `<Column>`
- **No border-radius on images** — images will render as rectangles
- **No background-image** — use solid `background-color` instead
- **No box-shadow** — omit shadow effects
- **No `overflow: hidden`** — design within container bounds
- **Padding on `<a>` tags** inconsistent — `<Button>` component handles this automatically
- **`max-width` ignored** on some elements — set explicit `width` as well

### Outlook Tips
- `<Button>` has built-in Outlook padding workarounds — always use it for CTAs
- Set explicit `width` on `<Container>` via `style={{ maxWidth: "600px" }}` for Outlook
- Use `border` instead of `box-shadow` for subtle visual separation
- Test with Outlook on Windows if your audience is business-heavy (~15% of business email opens)

### MSO Conditional Comments
React Email/JSX cannot natively express MSO conditional comments (`<!--[if mso]>`). If needed, post-process the rendered HTML output. This is rarely necessary — React Email's components handle most Outlook quirks automatically.

---

## Gmail Specifics

### Gmail Clipping
Gmail clips emails larger than approximately **102KB** of HTML. When clipped, only a portion displays and the user must click "View entire message".

**Mitigation:**
- Keep emails concise — avoid excessive content
- Use Tailwind sparingly (each class adds inline style bytes)
- Host images externally on CDN — never use base64-encoded images
- Don't include large code blocks or data tables inline
- Consider splitting long content across multiple emails
- Link to a web version for detailed content

### Gmail CSS Support
- Gmail web supports `<style>` tags in `<head>` (responsive breakpoints work)
- Gmail strips `class` attributes but React Email inlines styles, so this doesn't matter
- Gmail supports most basic CSS properties

---

## Apple Mail / iOS Mail

- Best CSS support of any email client
- Supports web fonts, media queries, `<style>` tags, most modern CSS
- Web fonts declared with `<Font>` work here
- Dark mode support via `@media (prefers-color-scheme: dark)` (but not via Tailwind `dark:` variant)

---

## Dark Mode Strategy

The `dark:` Tailwind variant does NOT work in email clients. Approaches:

### Option 1: Light Theme (Recommended)
Design for light backgrounds. This works everywhere and avoids dark mode issues.

### Option 2: Dark by Default
Apply dark colors directly in your styles/theme config:
```tsx
<Tailwind config={{
  presets: [pixelBasedPreset],
  theme: {
    extend: {
      colors: {
        bg: '#1a1a2e',
        text: '#e0e0e0',
        accent: '#00d4ff',
      },
    },
  },
}}>
  <Body className="bg-bg text-text">...</Body>
</Tailwind>
```

### Option 3: Neutral Design
Use medium-contrast colors that look acceptable in both light and dark mode contexts. Avoid pure white (`#ffffff`) backgrounds — use light grays (`#f5f5f5`, `#fafafa`) instead.

---

## Web Font Support

| Client | Web Fonts? |
|--------|-----------|
| Apple Mail | Yes |
| iOS Mail | Yes |
| Thunderbird | Yes |
| Some Android clients | Yes |
| Gmail (all) | No — uses fallback |
| Outlook (all) | No — uses fallback |
| Yahoo Mail | No — uses fallback |

**Always provide `fallbackFontFamily`** in the `<Font>` component.

**Safe font stacks:**
- Sans-serif: `Arial, Helvetica, sans-serif`
- Serif: `Georgia, Times New Roman, serif`
- Monospace: `Courier New, monospace`

---

## Image Best Practices for Cross-Client

1. **Always set `width` and `height`** — prevents layout shifts when images are blocked
2. **Always set `alt` text** — displayed when images are blocked (common in corporate email)
3. **Use PNG or JPEG only** — SVG and WEBP have poor support
4. **Use absolute URLs** — relative paths don't work in email
5. **Host on CDN** — fast loading, reliable availability
6. **Avoid base64 encoding** — increases email size, some clients block it, contributes to Gmail clipping
7. **Responsive pattern:** `width="600" className="w-full max-w-[600px] h-auto"`
8. **Center images:** `className="block mx-auto"`
