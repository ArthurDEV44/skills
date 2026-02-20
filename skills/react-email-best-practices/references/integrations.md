# Email Provider Integrations

All integrations follow the same pattern:
1. Build email as a React component
2. Call `await render(<Component props />)` to get an HTML string
3. Pass the HTML string to the provider's send function

## Rendering Functions

```tsx
import { render, toPlainText, pretty } from '@react-email/components';
// Or: import { render, toPlainText, pretty } from '@react-email/render';

// Render to HTML string (async)
const html = await render(<MyEmail name="John" />);

// Convert HTML to plain text (for text-only email clients)
const text = toPlainText(html);

// Pretty-print HTML for debugging
const prettyHtml = await pretty(await render(<MyEmail name="John" />));
```

**Note:** `render()` is async and returns a `Promise<string>`.

---

## Resend (Native Integration)

Resend accepts JSX directly — no need to call `render()`.

```tsx
import { Resend } from 'resend';
import { WelcomeEmail } from './emails/welcome';

const resend = new Resend(process.env.RESEND_API_KEY);

await resend.emails.send({
  from: 'Acme <onboarding@resend.dev>',
  to: ['user@example.com'],
  subject: 'Welcome to Acme',
  react: <WelcomeEmail name="John" />,
});
```

## Nodemailer

```tsx
import { render } from '@react-email/components';
import nodemailer from 'nodemailer';
import { WelcomeEmail } from './emails/welcome';

const transporter = nodemailer.createTransport({
  host: 'smtp.example.com',
  port: 465,
  secure: true,
  auth: { user: 'my_user', pass: 'my_password' },
});

const html = await render(<WelcomeEmail name="John" />);

await transporter.sendMail({
  from: 'you@example.com',
  to: 'user@example.com',
  subject: 'Welcome',
  html,
});
```

## SendGrid

```tsx
import { render } from '@react-email/components';
import sgMail from '@sendgrid/mail';
import { WelcomeEmail } from './emails/welcome';

sgMail.setApiKey(process.env.SENDGRID_API_KEY!);

const html = await render(<WelcomeEmail name="John" />);

await sgMail.send({
  to: 'user@example.com',
  from: 'noreply@example.com',
  subject: 'Welcome',
  html,
});
```

## Postmark

```tsx
import { render } from '@react-email/components';
import postmark from 'postmark';
import { WelcomeEmail } from './emails/welcome';

const client = new postmark.ServerClient(process.env.POSTMARK_API_KEY!);

const html = await render(<WelcomeEmail name="John" />);

await client.sendEmail({
  From: 'you@example.com',
  To: 'user@example.com',
  Subject: 'Welcome',
  HtmlBody: html,
});
```

## AWS SES

```tsx
import { render } from '@react-email/components';
import { SES } from '@aws-sdk/client-ses';
import { WelcomeEmail } from './emails/welcome';

const ses = new SES({ region: process.env.AWS_SES_REGION });

const html = await render(<WelcomeEmail name="John" />);

await ses.sendEmail({
  Source: 'you@example.com',
  Destination: { ToAddresses: ['user@example.com'] },
  Message: {
    Body: { Html: { Charset: 'UTF-8', Data: html } },
    Subject: { Charset: 'UTF-8', Data: 'Welcome' },
  },
});
```

## MailerSend

```tsx
import { render } from '@react-email/components';
import { MailerSend, EmailParams, Sender, Recipient } from 'mailersend';
import { WelcomeEmail } from './emails/welcome';

const mailerSend = new MailerSend({ apiKey: process.env.MAILERSEND_API_KEY! });

const html = await render(<WelcomeEmail name="John" />);

const sentFrom = new Sender('you@yourdomain.com', 'Your Name');
const recipients = [new Recipient('user@example.com', 'User Name')];

const emailParams = new EmailParams()
  .setFrom(sentFrom)
  .setTo(recipients)
  .setSubject('Welcome')
  .setHtml(html)
  .setText('Welcome to Acme!');

await mailerSend.email.send(emailParams);
```

## Plunk

```tsx
import Plunk from '@plunk/node';
import { render } from '@react-email/components';
import { WelcomeEmail } from './emails/welcome';

const plunk = new Plunk(process.env.PLUNK_API_KEY!);

const html = await render(<WelcomeEmail name="John" />);

await plunk.emails.send({
  to: 'user@example.com',
  subject: 'Welcome',
  body: html,
});
```

## Loops

```tsx
import { render } from '@react-email/components';
import { WelcomeEmail } from './emails/welcome';

const html = await render(<WelcomeEmail name="John" />);

// Via Loops REST API
await fetch('https://app.loops.so/api/v1/transactional', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${process.env.LOOPS_API_KEY}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    transactionalId: 'your-transactional-id',
    email: 'user@example.com',
    dataVariables: { name: 'John' },
  }),
});
```

---

## Next.js Integration Patterns

### Server Action (App Router)

```tsx
// app/actions/send-email.ts
'use server';

import { render } from '@react-email/components';
import { Resend } from 'resend';
import { WelcomeEmail } from '@/emails/welcome';

const resend = new Resend(process.env.RESEND_API_KEY);

export async function sendWelcomeEmail(name: string, email: string) {
  await resend.emails.send({
    from: 'Acme <noreply@acme.com>',
    to: [email],
    subject: 'Welcome to Acme',
    react: <WelcomeEmail name={name} />,
  });
}
```

### API Route (App Router)

```tsx
// app/api/send/route.ts
import { render } from '@react-email/components';
import { NextResponse } from 'next/server';
import { WelcomeEmail } from '@/emails/welcome';

export async function POST(request: Request) {
  const { name, email } = await request.json();
  const html = await render(<WelcomeEmail name={name} />);

  // Send via your provider of choice
  // ...

  return NextResponse.json({ success: true });
}
```
