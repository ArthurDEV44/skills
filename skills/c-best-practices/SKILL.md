---
name: c-best-practices
description: >
  C programming best practices, idiomatic patterns, and safe coding for modern C (C11/C17/C23).
  Covers memory management (malloc/calloc/realloc/free, leak prevention), pointer safety, undefined
  behavior avoidance, struct design, opaque types, error handling (errno, return codes), string
  safety, preprocessor hygiene, type qualifiers (const, volatile, restrict), function pointers,
  callbacks, alignment, flexible array members, _Generic, C11 threads and atomics, C23 features
  (typeof, nullptr, constexpr), build tooling (GCC, Clang, sanitizers, Valgrind), and anti-patterns.
  Use when writing, reviewing, or refactoring C code: (1) Memory allocation and lifetimes,
  (2) Pointer safety, (3) Struct and opaque type design, (4) Error handling, (5) String and buffer
  safety, (6) Preprocessor macros, (7) C11/C17/C23 features, (8) Threads and atomics,
  (9) Build flags and sanitizers, (10) Avoiding undefined behavior.
---

# C Best Practices

## Memory Management

### Allocation Patterns

Always check return values. Prefer `sizeof *ptr` over `sizeof(Type)` to stay in sync with the declaration:

```c
// GOOD: sizeof *ptr adapts if type changes
int *arr = malloc(n * sizeof *arr);
if (!arr) {
    perror("malloc");
    return -1;
}

// GOOD: zero-initialized array
int *arr = calloc(n, sizeof *arr);
if (!arr) { /* handle error */ }

// BAD: type name can drift from declaration
int *arr = malloc(n * sizeof(int));
```

### Realloc Safety

Never assign `realloc` directly to the same pointer -- if it fails, you leak the original:

```c
// BAD: leaks memory on failure
buf = realloc(buf, new_size);

// GOOD: preserve original on failure
void *tmp = realloc(buf, new_size);
if (!tmp) {
    perror("realloc");
    free(buf);
    return -1;
}
buf = tmp;
```

### Ownership Discipline

Adopt clear ownership rules:

```c
// Document who owns the returned pointer
/* Caller must free() the returned string. */
char *create_greeting(const char *name) {
    char *msg = malloc(256);
    if (!msg) return NULL;
    snprintf(msg, 256, "Hello, %s!", name);
    return msg;
}

// Set pointers to NULL after freeing to prevent use-after-free
free(ptr);
ptr = NULL;
```

### Cleanup with goto

The standard C pattern for multi-resource cleanup:

```c
int process_file(const char *path) {
    int ret = -1;
    FILE *fp = NULL;
    char *buf = NULL;

    fp = fopen(path, "r");
    if (!fp) goto cleanup;

    buf = malloc(4096);
    if (!buf) goto cleanup;

    /* ... do work ... */
    ret = 0;

cleanup:
    free(buf);      // free(NULL) is safe
    if (fp) fclose(fp);
    return ret;
}
```

For detailed memory patterns (arena allocation, pool allocators, leak detection), see `references/memory-safety.md`.

## Pointer Safety

### Golden Rules

1. **Initialize all pointers** -- either to a valid address or `NULL`
2. **Check for NULL** before dereferencing
3. **Never return a pointer to a local variable**
4. **Set freed pointers to NULL**
5. **Bounds-check all array accesses**

```c
// BAD: uninitialized pointer
int *p;
*p = 42;  // undefined behavior

// GOOD: initialize
int *p = NULL;
// or
int x = 42;
int *p = &x;
```

### Pointer Arithmetic

```c
int arr[10];
int *p = arr;

// GOOD: within bounds
for (int i = 0; i < 10; i++)
    p[i] = i;

// GOOD: pointer to one-past-end is valid (but don't dereference)
int *end = arr + 10;
for (int *it = arr; it != end; it++)
    *it *= 2;

// BAD: out-of-bounds access
int val = arr[10];  // undefined behavior
```

### Restrict Qualifier

Tells the compiler pointers don't alias, enabling optimizations:

```c
// With restrict: compiler can optimize assuming no overlap
void add_arrays(int *restrict dst, const int *restrict a,
                const int *restrict b, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
}
```

## Struct Design

### Opaque Types (Information Hiding)

Header exposes only a forward declaration -- implementation hidden in `.c` file:

```c
/* connection.h */
typedef struct Connection Connection;

Connection *conn_create(const char *host, int port);
int         conn_send(Connection *conn, const void *data, size_t len);
void        conn_destroy(Connection *conn);

/* connection.c */
#include "connection.h"
struct Connection {
    int    fd;
    char   host[256];
    int    port;
    bool   connected;
};

Connection *conn_create(const char *host, int port) {
    Connection *conn = calloc(1, sizeof *conn);
    if (!conn) return NULL;
    snprintf(conn->host, sizeof conn->host, "%s", host);
    conn->port = port;
    return conn;
}

void conn_destroy(Connection *conn) {
    if (!conn) return;
    if (conn->connected) close(conn->fd);
    free(conn);
}
```

### Designated Initializers (C99+)

```c
struct Config {
    const char *host;
    int         port;
    int         max_retries;
    bool        use_tls;
};

// GOOD: clear, order-independent, zero-initializes omitted fields
struct Config cfg = {
    .host        = "localhost",
    .port        = 8080,
    .max_retries = 3,
};
// cfg.use_tls is implicitly false (zero)

// Zero-initialize everything
struct Config empty = {0};
```

### Flexible Array Members (C99+)

```c
typedef struct {
    size_t len;
    char   data[];  // flexible array member -- must be last
} Buffer;

Buffer *buf_create(size_t size) {
    Buffer *buf = malloc(sizeof *buf + size);
    if (!buf) return NULL;
    buf->len = size;
    memset(buf->data, 0, size);
    return buf;
}
```

### Struct Padding and Alignment

```c
// BAD: wastes memory due to padding
struct BadLayout {
    char  a;    // 1 byte + 7 padding
    double b;   // 8 bytes
    char  c;    // 1 byte + 7 padding
};  // Total: 24 bytes

// GOOD: order fields by decreasing size
struct GoodLayout {
    double b;   // 8 bytes
    char   a;   // 1 byte
    char   c;   // 1 byte + 6 padding
};  // Total: 16 bytes

// Verify with static_assert (C11+)
static_assert(sizeof(struct GoodLayout) == 16, "unexpected padding");
```

## Error Handling

### Return Code Conventions

```c
// Convention: 0 = success, negative = error
typedef enum {
    ERR_OK       =  0,
    ERR_NOMEM    = -1,
    ERR_IO       = -2,
    ERR_INVALID  = -3,
} ErrorCode;

const char *error_str(ErrorCode err) {
    switch (err) {
        case ERR_OK:      return "success";
        case ERR_NOMEM:   return "out of memory";
        case ERR_IO:      return "I/O error";
        case ERR_INVALID: return "invalid argument";
    }
    return "unknown error";
}

ErrorCode parse_config(const char *path, Config *out) {
    if (!path || !out) return ERR_INVALID;

    FILE *fp = fopen(path, "r");
    if (!fp) return ERR_IO;
    /* ... */
    fclose(fp);
    return ERR_OK;
}
```

### errno Usage

```c
#include <errno.h>
#include <string.h>

FILE *fp = fopen("/nonexistent", "r");
if (!fp) {
    // errno is set by fopen
    fprintf(stderr, "fopen: %s\n", strerror(errno));
    return -1;
}

// IMPORTANT: save errno if calling other functions that may change it
int saved_errno = errno;
log_message("open failed");   // this might change errno
errno = saved_errno;
```

### Output Parameters for Results

```c
// Return error code, write result via output pointer
ErrorCode parse_int(const char *str, int *out) {
    if (!str || !out) return ERR_INVALID;

    char *end;
    errno = 0;
    long val = strtol(str, &end, 10);
    if (errno == ERANGE || val > INT_MAX || val < INT_MIN)
        return ERR_INVALID;
    if (end == str || *end != '\0')
        return ERR_INVALID;

    *out = (int)val;
    return ERR_OK;
}
```

## String Safety

### Prefer Bounded Functions

```c
// BAD: buffer overflow if src is too long
strcpy(dst, src);
strcat(buf, suffix);
sprintf(buf, "Hello %s", name);

// GOOD: bounded versions
strncpy(dst, src, sizeof dst - 1);
dst[sizeof dst - 1] = '\0';  // strncpy doesn't guarantee NUL termination!

// BEST: snprintf -- always NUL-terminates, returns needed length
int n = snprintf(buf, sizeof buf, "Hello %s", name);
if (n < 0 || (size_t)n >= sizeof buf) {
    /* truncation occurred */
}
```

### strlcpy / strlcat (POSIX / BSD / C23)

```c
// C23 or provide your own -- always NUL-terminates, returns src length
size_t n = strlcpy(dst, src, sizeof dst);
if (n >= sizeof dst) { /* truncated */ }
```

### String Literals and const

```c
// GOOD: string literals are read-only, use const
const char *greeting = "Hello";

// BAD: modifying a string literal is undefined behavior
char *s = "Hello";
s[0] = 'h';  // UB!

// GOOD: use an array if you need mutation
char greeting[] = "Hello";
greeting[0] = 'h';  // OK
```

For detailed string patterns and buffer management, see `references/strings-buffers.md`.

## Preprocessor Hygiene

### Include Guards

```c
// GOOD: traditional include guard
#ifndef PROJECT_MODULE_H
#define PROJECT_MODULE_H

/* declarations */

#endif /* PROJECT_MODULE_H */

// ALSO GOOD: #pragma once (non-standard but universally supported)
#pragma once
```

### Macro Best Practices

```c
// ALWAYS parenthesize macro arguments and the whole expression
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// BAD: no parens -- breaks with expressions
#define SQUARE(x) x * x
// SQUARE(1+2) expands to 1+2 * 1+2 = 5, not 9!

// GOOD: use do-while(0) for multi-statement macros
#define LOG_ERROR(msg)          \
    do {                        \
        fprintf(stderr, "[ERROR] %s:%d: %s\n", \
                __FILE__, __LINE__, (msg));     \
    } while (0)

// Prefer inline functions over macros when possible (type-safe, debuggable)
static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

// Use _Generic (C11) for type-generic selection
#define abs_val(x) _Generic((x),    \
    int:    abs,                     \
    long:   labs,                    \
    float:  fabsf,                   \
    double: fabs                     \
)(x)
```

## Type Qualifiers

### const Correctness

```c
// Pointer to const data -- promise not to modify through this pointer
void print_data(const int *data, size_t n) {
    for (size_t i = 0; i < n; i++)
        printf("%d ", data[i]);
}

// const pointer to mutable data
int *const ptr = &x;  // can't change ptr, can change *ptr

// const pointer to const data
const int *const ptr = &x;  // can't change either

// Use const liberally -- documents intent, enables compiler optimizations
```

### volatile

```c
// For hardware registers, signal handlers, memory-mapped I/O
volatile int *status_reg = (volatile int *)0xFFFF0000;

// In signal handlers
volatile sig_atomic_t got_signal = 0;

void handler(int sig) {
    got_signal = 1;  // volatile ensures compiler doesn't optimize out
}
```

## Function Pointers & Callbacks

```c
// Callback pattern with user context
typedef void (*EventCallback)(int event_type, void *user_data);

typedef struct {
    EventCallback on_event;
    void         *user_data;
} EventHandler;

void register_handler(EventHandler *handler,
                       EventCallback cb, void *data) {
    handler->on_event  = cb;
    handler->user_data = data;
}

void dispatch(EventHandler *handler, int event) {
    if (handler->on_event)
        handler->on_event(event, handler->user_data);
}

// Virtual table pattern (poor man's OOP)
typedef struct {
    int  (*open)(void *self, const char *path);
    int  (*read)(void *self, void *buf, size_t n);
    void (*close)(void *self);
} StreamVTable;

typedef struct {
    const StreamVTable *vtable;
    /* implementation-specific fields follow */
} Stream;
```

## Common Anti-Patterns

### Undefined Behavior Traps

```c
// Signed integer overflow is UB
int x = INT_MAX;
x += 1;  // UB! Use unsigned or check before

// Null pointer dereference
int *p = NULL;
*p = 42;  // UB!

// Use-after-free
free(ptr);
*ptr = 0;  // UB! Set ptr = NULL after free

// Double free
free(ptr);
free(ptr);  // UB!

// Buffer overflow
char buf[10];
strcpy(buf, "this is way too long");  // UB!

// Unsequenced modifications
int i = 0;
i = i++ + ++i;  // UB! Don't do this

// Missing return value
int foo(int x) {
    if (x > 0) return x;
    // UB: falls off end of non-void function
}
```

### Magic Numbers

```c
// BAD
if (status == 3) { /* what is 3? */ }

// GOOD
enum { STATUS_READY = 3 };
if (status == STATUS_READY) { /* clear intent */ }

// Or use an enum
typedef enum { STATUS_INIT, STATUS_RUNNING, STATUS_READY } Status;
```

### Not Using const

```c
// BAD: caller doesn't know if their data will be mutated
void process(char *data, int len);

// GOOD: promise not to modify
void process(const char *data, int len);
```

For C11/C17/C23 features (threads, atomics, typeof, nullptr, constexpr), see `references/modern-c.md`.

For build tooling (compiler flags, sanitizers, Valgrind, static analysis), see `references/build-tooling.md`.

For concurrency patterns (C11 threads, mutexes, atomics, memory orders), see `references/concurrency.md`.
