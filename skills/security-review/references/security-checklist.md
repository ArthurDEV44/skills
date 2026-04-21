# Security Checklist — Detailed Vulnerability Patterns

## Table of Contents

1. [Injection](#1-injection) — SQL, XSS, Command, Path Traversal, Template
2. [Authentication & Authorization](#2-authentication--authorization) — Broken Access, Auth Bypass, IDOR
3. [Cryptography](#3-cryptography) — Weak Algos, Hardcoded Keys, Insecure Random
4. [Secrets Exposure](#4-secrets-exposure) — Hardcoded Credentials, Committed Env Files
5. [Data Handling](#5-data-handling) — Deserialization, Sensitive Data, Input Validation
6. [Configuration](#6-configuration) — Misconfiguration, CSRF, SSRF
7. [Dependencies](#7-dependencies) — CVEs, Unmaintained Packages
8. [AI-Generated Code Anti-Patterns](#8-ai-generated-code-anti-patterns) — eval, innerHTML, Math.random
9. [Calibration Examples](#9-calibration-examples) — True Positive, False Positive, Borderline

---

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

## 9. Calibration Examples

These examples calibrate the boundary between true positives, false positives, and borderline findings. Use them to anchor your severity and confidence assessments.

### Example A — True Positive (CRITICAL, HIGH confidence)

**Input code (Python/Flask):**
```python
@app.route("/user/<id>")
def get_user(id):
    query = f"SELECT * FROM users WHERE id={id}"
    result = db.execute(query)
    return jsonify(result.fetchone())
```

**Expected finding:**
```
[C-1] SQL Injection in User Lookup
- File: app.py:3
- Type: CWE-89: SQL Injection
- Severity: CRITICAL | Confidence: HIGH
- Description: User-controlled route parameter `id` is interpolated directly into
  a SQL query via f-string. No parameterization or input validation.
- Reasoning: Full dataflow traced — route parameter flows directly into db.execute()
  with no sanitization. Exploitable remotely without authentication via crafted URL.
  RCE possible on some database engines.
- Remediation:
  // Before (vulnerable)
  query = f"SELECT * FROM users WHERE id={id}"
  result = db.execute(query)

  // After (fixed)
  result = db.execute("SELECT * FROM users WHERE id = :id", {"id": id})
```

### Example B — False Positive (DO NOT flag)

**Input code (Python/SQLAlchemy):**
```python
@app.route("/user/<int:user_id>")
@login_required
def get_user(user_id):
    user = db.session.query(User).filter(User.id == user_id).first()
    if user is None:
        abort(404)
    return jsonify(user.to_dict())
```

**Why this is NOT a finding:**
- SQLAlchemy ORM generates parameterized queries internally. `User.id == user_id` produces `WHERE id = $1` with bound parameter.
- Flask's `<int:user_id>` route converter rejects non-integer input at the routing layer.
- `@login_required` provides authentication.
- This is a safe, idiomatic pattern. Flagging it would be a false positive.

**Note:** This may still warrant an INFO finding for missing ownership check (user A can access user B's data) — but that is an authorization issue (CWE-284), not SQL injection.

### Example C — Borderline (MEDIUM severity, LOW confidence)

**Input code (Node.js/Express):**
```javascript
app.get("/api/file", (req, res) => {
  const filename = path.basename(req.query.name);
  const filepath = path.join(__dirname, "uploads", filename);
  if (fs.existsSync(filepath)) {
    res.sendFile(filepath);
  } else {
    res.status(404).json({ error: "Not found" });
  }
});
```

**Expected finding:**
```
[M-1] Potential Path Traversal in File Serving
- File: server.js:3
- Type: CWE-22: Path Traversal
- Severity: MEDIUM | Confidence: LOW
- Description: User-controlled query parameter `name` is used to construct a file
  path. path.basename() strips directory components, which mitigates basic ../
  traversal, but does not protect against all edge cases.
- Reasoning: path.basename() provides partial mitigation — it strips directory
  separators on the current OS. However, it does not canonicalize (resolve symlinks)
  and may behave differently on Windows vs Unix for mixed separators. The uploads
  directory may contain symlinks. Confidence is LOW because the primary traversal
  vector IS mitigated; the remaining risk is edge-case.
- Remediation:
  // Before (partial mitigation)
  const filename = path.basename(req.query.name);
  const filepath = path.join(__dirname, "uploads", filename);

  // After (full mitigation)
  const filename = path.basename(req.query.name);
  const filepath = path.resolve(path.join(__dirname, "uploads", filename));
  if (!filepath.startsWith(path.resolve(path.join(__dirname, "uploads")))) {
    return res.status(403).json({ error: "Forbidden" });
  }
```

**Why this calibration matters:** The code HAS a mitigation (path.basename). A monolithic scanner might either miss it entirely (false negative) or flag it as CRITICAL (false positive). The correct assessment acknowledges the mitigation, rates the residual risk, and assigns LOW confidence to signal "verify manually."
