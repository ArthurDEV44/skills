# Security Checklist — Detailed Vulnerability Patterns

## 1. Injection

### SQL Injection (CWE-89)
**Indicators:** String concatenation in SQL queries, f-strings/template literals with user input in queries, missing parameterized queries.

| Language | Vulnerable Pattern | Safe Pattern |
|----------|-------------------|--------------|
| Python | `f"SELECT * FROM users WHERE id={user_id}"` | `cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))` |
| JavaScript | `` `SELECT * FROM users WHERE id=${userId}` `` | `db.query("SELECT * FROM users WHERE id=$1", [userId])` |
| Rust | `format!("SELECT * FROM users WHERE id={}", id)` | `sqlx::query("SELECT * FROM users WHERE id = $1").bind(id)` |
| Go | `fmt.Sprintf("SELECT * FROM users WHERE id=%s", id)` | `db.Query("SELECT * FROM users WHERE id=$1", id)` |

### XSS — Cross-Site Scripting (CWE-79)
**Indicators:** `innerHTML`, `dangerouslySetInnerHTML`, `v-html`, `{!! !!}`, `| safe`, unescaped template variables, `document.write()`.

Check for:
- User input rendered without sanitization
- URL parameters reflected in HTML
- Markdown/rich text rendered as raw HTML
- SVG files with embedded scripts

### Command Injection (CWE-78)
**Indicators:** `exec()`, `system()`, `popen()`, `child_process.exec()`, `os.system()`, `subprocess.run(shell=True)`, backtick execution.

Check for:
- User input in shell commands
- Unescaped arguments in command strings
- Missing use of parameterized command execution (e.g., `subprocess.run([...])` instead of shell=True)

### Path Traversal (CWE-22)
**Indicators:** `../` in file paths, user input in `open()`, `readFile()`, `fs.readFileSync()`, `Path::new()` with user input.

Check for:
- Missing canonicalization before file access
- User-controlled file names without basename extraction
- Symlink following without checks

### Template Injection (CWE-1336)
**Indicators:** User input in template strings, `render_template_string()`, `Jinja2(env).from_string(user_input)`, `eval()` with template contexts.

## 2. Authentication & Authorization

### Broken Access Control (CWE-284)
Check for:
- Missing authorization checks on endpoints (handler has no auth middleware)
- Direct object reference without ownership validation (user A accessing user B's resource)
- Missing role/permission checks for privileged operations
- Horizontal privilege escalation (changing user ID in request)
- Vertical privilege escalation (accessing admin functions as regular user)

### Authentication Bypass (CWE-287)
Check for:
- Hard-coded bypass conditions (`if user == "admin"`)
- Missing authentication on sensitive routes
- JWT without signature verification
- JWT with `alg: none` accepted
- Session fixation (session ID not rotated after login)
- Missing rate limiting on login/auth endpoints

### Insecure Direct Object Reference (CWE-639)
Check for:
- Database lookups using user-provided IDs without ownership check
- File access using user-provided paths
- API endpoints exposing sequential/guessable IDs

## 3. Cryptography

### Weak Algorithms (CWE-327)
**Flag as HIGH/CRITICAL:**
- MD5 or SHA1 for password hashing or integrity
- DES, 3DES, RC4 for encryption
- RSA < 2048 bits
- ECB mode for block ciphers

**Acceptable:**
- SHA-256+ for integrity (not passwords)
- AES-GCM, ChaCha20-Poly1305 for encryption
- Argon2, bcrypt, scrypt for password hashing
- Ed25519, ECDSA P-256+ for signatures

### Hardcoded Keys (CWE-321)
Check for:
- Encryption keys as string literals
- API keys in source code
- Private keys in configuration files
- Base64-encoded secrets (decode and check)

### Insecure Random (CWE-338)
**Flag:**
- `Math.random()` for tokens, secrets, or IDs
- `rand()` (C) without seeding from secure source
- `random.random()` (Python) for security-sensitive values

**Safe:** `crypto.randomBytes()`, `secrets.token_hex()`, `OsRng`, `crypto/rand`

## 4. Secrets Exposure

### Hardcoded Credentials (CWE-798)
**Regex patterns to check:**
```
password\s*=\s*["'][^"']+["']
api[_-]?key\s*=\s*["'][^"']+["']
secret\s*=\s*["'][^"']+["']
token\s*=\s*["'][^"']+["']
(aws|gcp|azure)[_-]?(access|secret|key)
(sk|pk)[-_](live|test)[-_][a-zA-Z0-9]+
ghp_[a-zA-Z0-9]{36}
AKIA[0-9A-Z]{16}
```

Check for:
- `.env` files committed (should be in .gitignore)
- Credentials in config files, docker-compose, CI/CD configs
- Connection strings with embedded passwords
- Private keys or certificates in source

## 5. Data Handling

### Insecure Deserialization (CWE-502)
**Flag:**
- `pickle.loads()` with user input (Python)
- `JSON.parse()` → `eval()` chain
- `yaml.load()` without `Loader=SafeLoader` (Python)
- `ObjectInputStream` with untrusted data (Java)
- `serde_json::from_str` on untrusted input without size limits

### Sensitive Data Exposure (CWE-200)
Check for:
- PII logged to console/files (emails, SSNs, credit cards)
- Sensitive data in URL query parameters
- Detailed error messages in production (stack traces, internal paths)
- Missing encryption for data at rest or in transit
- Sensitive data in client-side storage (localStorage, cookies without secure flag)

### Missing Input Validation (CWE-20)
Check for:
- User input used directly without validation at system boundaries
- Missing length limits on string inputs
- Missing type coercion/validation on numeric inputs
- Email, URL, or other structured input without format validation

## 6. Configuration

### Security Misconfiguration (CWE-16)
Check for:
- `DEBUG=True` or equivalent in production config
- CORS with `Access-Control-Allow-Origin: *` on authenticated endpoints
- Missing security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)
- Default credentials in configuration
- Verbose error responses in production
- Unnecessary ports/services exposed

### CSRF (CWE-352)
Check for:
- State-changing operations on GET requests
- Missing CSRF tokens on forms
- Missing SameSite cookie attribute
- CORS configuration allowing credentials from any origin

### SSRF (CWE-918)
Check for:
- User-provided URLs fetched server-side without validation
- Missing allowlist for external requests
- Internal service URLs accessible via user input
- DNS rebinding vulnerabilities

## 7. Dependencies

Check for:
- Known CVEs in dependency versions (check against advisory databases)
- Unmaintained packages (no updates in >2 years)
- Overly permissive version ranges in manifests
- Unnecessary dependencies that increase attack surface
- Lock file changes that downgrade security-critical packages

## 8. AI-Generated Code Anti-Patterns

Patterns commonly introduced by AI code generation that are security-relevant:

| Pattern | Risk | Fix |
|---------|------|-----|
| `eval()` with any dynamic input | Code injection | Use safe alternatives (JSON.parse, AST parsing) |
| `innerHTML = userInput` | XSS | Use textContent or sanitize with DOMPurify |
| `Math.random()` for tokens/IDs | Predictable values | Use crypto.randomBytes() / crypto.getRandomValues() |
| `MD5(password)` | Weak hashing | Use bcrypt/argon2/scrypt |
| `subprocess.run(cmd, shell=True)` | Command injection | Use list form: `subprocess.run([...])` |
| `JSON.parse(untrustedInput)` without try/catch | DoS via malformed input | Wrap in try/catch with size limit |
| `fs.readFileSync(userPath)` | Path traversal | Canonicalize + validate against allowed directory |
| `.unwrap()` on user input in Rust | Panic/DoS | Use `?` or `.unwrap_or_default()` |
| Logging user input verbatim | Log injection | Sanitize before logging |
| `cors({ origin: '*', credentials: true })` | Auth bypass | Specify allowed origins explicitly |
