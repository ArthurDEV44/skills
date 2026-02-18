# C Strings & Buffer Management

## String Fundamentals

C strings are null-terminated (`'\0'`) arrays of `char`. The length does not include the null terminator, but storage must account for it.

```c
// String literal -- stored in read-only memory
const char *greeting = "Hello";  // 6 bytes: H e l l o \0

// Mutable copy
char greeting[] = "Hello";  // stack array, 6 bytes, modifiable
greeting[0] = 'h';          // OK

// NEVER modify a string literal
char *bad = "Hello";
bad[0] = 'h';  // UB!
```

## Safe String Functions

### snprintf -- The Swiss Army Knife

Always NUL-terminates. Returns the number of characters that *would* have been written (excluding NUL), allowing truncation detection:

```c
char buf[64];
int n = snprintf(buf, sizeof buf, "User %s (id=%d)", name, id);
if (n < 0) {
    /* encoding error */
} else if ((size_t)n >= sizeof buf) {
    /* truncated -- n tells you how much space was needed */
    fprintf(stderr, "Warning: output truncated (%d chars needed)\n", n);
}
```

### Dynamic String Building

```c
// Build a string of unknown length dynamically
char *build_csv_line(const char **fields, size_t count) {
    size_t total = 0;
    for (size_t i = 0; i < count; i++)
        total += strlen(fields[i]) + 1;  // +1 for comma or NUL

    char *line = malloc(total);
    if (!line) return NULL;

    size_t offset = 0;
    for (size_t i = 0; i < count; i++) {
        size_t len = strlen(fields[i]);
        memcpy(line + offset, fields[i], len);
        offset += len;
        line[offset++] = (i < count - 1) ? ',' : '\0';
    }
    return line;
}
```

### Growable String Buffer

```c
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} StrBuf;

StrBuf strbuf_new(size_t initial_cap) {
    StrBuf sb = {0};
    sb.data = malloc(initial_cap);
    if (sb.data) {
        sb.cap = initial_cap;
        sb.data[0] = '\0';
    }
    return sb;
}

int strbuf_append(StrBuf *sb, const char *str) {
    size_t slen = strlen(str);
    size_t needed = sb->len + slen + 1;
    if (needed > sb->cap) {
        size_t new_cap = sb->cap * 2;
        if (new_cap < needed) new_cap = needed;
        char *tmp = realloc(sb->data, new_cap);
        if (!tmp) return -1;
        sb->data = tmp;
        sb->cap = new_cap;
    }
    memcpy(sb->data + sb->len, str, slen + 1);
    sb->len += slen;
    return 0;
}

int strbuf_appendf(StrBuf *sb, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if (needed < 0) return -1;

    size_t total = sb->len + (size_t)needed + 1;
    if (total > sb->cap) {
        size_t new_cap = sb->cap * 2;
        if (new_cap < total) new_cap = total;
        char *tmp = realloc(sb->data, new_cap);
        if (!tmp) return -1;
        sb->data = tmp;
        sb->cap = new_cap;
    }

    va_start(args, fmt);
    vsnprintf(sb->data + sb->len, (size_t)needed + 1, fmt, args);
    va_end(args);
    sb->len += (size_t)needed;
    return 0;
}

void strbuf_free(StrBuf *sb) {
    free(sb->data);
    *sb = (StrBuf){0};
}
```

## String Comparison and Searching

```c
// Comparison
if (strcmp(a, b) == 0) { /* equal */ }
if (strncmp(a, b, n) == 0) { /* first n chars equal */ }
if (strcasecmp(a, b) == 0) { /* case-insensitive (POSIX) */ }

// Searching
char *pos = strchr(str, 'x');     // first occurrence of char
char *pos = strrchr(str, 'x');    // last occurrence of char
char *pos = strstr(haystack, needle);  // first occurrence of substring

// Tokenizing (MODIFIES the string -- work on a copy)
char input[] = "one,two,three";
char *token = strtok(input, ",");
while (token) {
    printf("%s\n", token);
    token = strtok(NULL, ",");
}

// Thread-safe alternative
char *saveptr;
char *token = strtok_r(input, ",", &saveptr);
```

## Parsing Numbers from Strings

```c
// Prefer strtol/strtod over atoi/atof -- they report errors

long parse_long(const char *str, bool *ok) {
    if (!str || !*str) { *ok = false; return 0; }

    char *end;
    errno = 0;
    long val = strtol(str, &end, 10);

    if (errno == ERANGE) { *ok = false; return 0; }  // overflow
    if (end == str)      { *ok = false; return 0; }  // no digits
    if (*end != '\0')    { *ok = false; return 0; }  // trailing junk

    *ok = true;
    return val;
}

// Double parsing
double parse_double(const char *str, bool *ok) {
    char *end;
    errno = 0;
    double val = strtod(str, &end);
    *ok = (errno == 0 && end != str && *end == '\0');
    return val;
}
```

## Binary Buffer Operations

```c
// memcpy -- non-overlapping copy
memcpy(dst, src, n);

// memmove -- handles overlapping regions
memmove(dst, src, n);  // safe even if dst and src overlap

// memset -- fill memory
memset(buf, 0, sizeof buf);       // zero-fill
memset(buf, 0xFF, sizeof buf);    // fill with 0xFF

// memcmp -- binary comparison
if (memcmp(a, b, n) == 0) { /* first n bytes are identical */ }

// RULE: prefer memcpy for non-overlapping, memmove when in doubt
```

## String Conversion

```c
// int/long to string
char buf[32];
snprintf(buf, sizeof buf, "%d", value);
snprintf(buf, sizeof buf, "%ld", long_value);
snprintf(buf, sizeof buf, "%lld", long_long_value);

// Hex
snprintf(buf, sizeof buf, "0x%X", value);
snprintf(buf, sizeof buf, "%02x", byte);

// Float/double
snprintf(buf, sizeof buf, "%.2f", dval);    // 2 decimal places
snprintf(buf, sizeof buf, "%g", dval);      // shortest representation
snprintf(buf, sizeof buf, "%e", dval);      // scientific notation
```

## Wide Strings and Multibyte

```c
#include <wchar.h>
#include <locale.h>

// Set locale for proper multibyte handling
setlocale(LC_ALL, "");

// Wide string operations
wchar_t wstr[] = L"Hello, World!";
size_t len = wcslen(wstr);

// Multibyte <-> wide conversion
const char *mb = "Hello";
wchar_t wide[128];
mbstowcs(wide, mb, 128);

char back[128];
wcstombs(back, wide, sizeof back);
```

## Common String Pitfalls

```c
// PITFALL: strlen on unterminated buffer
char buf[4] = {'H', 'e', 'l', 'p'};  // no NUL terminator!
size_t len = strlen(buf);  // reads past buffer -- UB

// PITFALL: off-by-one in buffer sizing
char buf[10];
strncpy(buf, long_string, 10);  // no room for NUL if src >= 10 chars
// FIX: strncpy(buf, str, sizeof buf - 1); buf[sizeof buf - 1] = '\0';

// PITFALL: comparing with == instead of strcmp
if (str == "hello") { }  // compares POINTER addresses, not content!
// FIX: if (strcmp(str, "hello") == 0) { }

// PITFALL: modifying strtok input
const char *original = "keep,this,safe";
char *copy = strdup(original);
char *token = strtok(copy, ",");  // modifies copy, not original
free(copy);
```
