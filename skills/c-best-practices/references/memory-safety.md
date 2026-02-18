# C Memory Safety

## Allocation Strategies

### Arena (Bump) Allocator

Allocate from a contiguous region, free everything at once. Ideal for request-scoped or frame-scoped lifetimes:

```c
typedef struct {
    char  *base;
    size_t size;
    size_t offset;
} Arena;

Arena arena_create(size_t size) {
    Arena a = {0};
    a.base = malloc(size);
    a.size = a.base ? size : 0;
    a.offset = 0;
    return a;
}

void *arena_alloc(Arena *a, size_t bytes, size_t align) {
    size_t aligned = (a->offset + align - 1) & ~(align - 1);
    if (aligned + bytes > a->size) return NULL;
    void *ptr = a->base + aligned;
    a->offset = aligned + bytes;
    return ptr;
}

void arena_reset(Arena *a) {
    a->offset = 0;  // "free" everything instantly
}

void arena_destroy(Arena *a) {
    free(a->base);
    *a = (Arena){0};
}

// Usage
Arena arena = arena_create(1024 * 1024);  // 1MB arena
int *data = arena_alloc(&arena, 100 * sizeof(int), _Alignof(int));
char *str = arena_alloc(&arena, 256, 1);
arena_reset(&arena);  // free all at once
arena_destroy(&arena);
```

### Pool Allocator

Fixed-size block allocator with O(1) alloc/free via a free list:

```c
typedef struct PoolBlock {
    struct PoolBlock *next;
} PoolBlock;

typedef struct {
    void      *memory;
    PoolBlock *free_list;
    size_t     block_size;
    size_t     capacity;
} Pool;

Pool pool_create(size_t block_size, size_t count) {
    // Ensure block_size fits a pointer (for free list)
    if (block_size < sizeof(PoolBlock))
        block_size = sizeof(PoolBlock);

    Pool pool = {0};
    pool.memory = malloc(block_size * count);
    if (!pool.memory) return pool;
    pool.block_size = block_size;
    pool.capacity = count;

    // Build free list
    pool.free_list = NULL;
    for (size_t i = count; i > 0; i--) {
        PoolBlock *blk = (PoolBlock *)((char *)pool.memory + (i - 1) * block_size);
        blk->next = pool.free_list;
        pool.free_list = blk;
    }
    return pool;
}

void *pool_alloc(Pool *pool) {
    if (!pool->free_list) return NULL;
    PoolBlock *blk = pool->free_list;
    pool->free_list = blk->next;
    return blk;
}

void pool_free(Pool *pool, void *ptr) {
    if (!ptr) return;
    PoolBlock *blk = ptr;
    blk->next = pool->free_list;
    pool->free_list = blk;
}

void pool_destroy(Pool *pool) {
    free(pool->memory);
    *pool = (Pool){0};
}
```

### Temporary Buffer Pattern

```c
// Stack-first, heap fallback for variable-size buffers
#define STACK_LIMIT 4096

void process_data(const char *input, size_t len) {
    char stack_buf[STACK_LIMIT];
    char *buf = stack_buf;
    bool heap = false;

    if (len + 1 > STACK_LIMIT) {
        buf = malloc(len + 1);
        if (!buf) { /* handle error */ return; }
        heap = true;
    }

    memcpy(buf, input, len);
    buf[len] = '\0';

    /* ... use buf ... */

    if (heap) free(buf);
}
```

## Common Memory Bugs

### Use-After-Free

```c
// BAD
char *name = strdup("Alice");
add_to_list(list, name);
free(name);
// list now holds a dangling pointer!

// GOOD: transfer ownership clearly
char *name = strdup("Alice");
add_to_list(list, name);  // list now owns name
name = NULL;               // relinquish our reference
// list_destroy() will free all entries
```

### Double Free

```c
// BAD
free(ptr);
/* ... some code ... */
free(ptr);  // UB!

// GOOD: NULL after free prevents double-free
free(ptr);
ptr = NULL;
free(ptr);  // free(NULL) is a no-op -- safe
```

### Memory Leak Patterns

```c
// LEAK: early return without free
char *buf = malloc(1024);
if (error_condition) return -1;  // leaked!
/* ... */
free(buf);

// FIX: goto cleanup
char *buf = malloc(1024);
if (error_condition) goto cleanup;
/* ... */
cleanup:
    free(buf);
    return ret;

// LEAK: lost pointer
void func(void) {
    malloc(100);  // return value not stored -- leaked!
}

// LEAK: realloc overwrite
buf = realloc(buf, new_size);  // if NULL, old buf leaked
```

### Buffer Overflow

```c
// BAD: no bounds check
void copy_name(char *dst, const char *src) {
    strcpy(dst, src);  // overflow if src > dst size
}

// GOOD: bounded copy
void copy_name(char *dst, size_t dst_size, const char *src) {
    if (dst_size == 0) return;
    size_t len = strlen(src);
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}
```

### Integer Overflow in Allocation

```c
// BAD: n * sizeof(int) can overflow
int *arr = malloc(n * sizeof(int));

// GOOD: check for overflow before multiplying
if (n > SIZE_MAX / sizeof(int)) {
    /* overflow -- reject */
    return NULL;
}
int *arr = malloc(n * sizeof(int));

// ALTERNATIVE: use calloc which checks internally
int *arr = calloc(n, sizeof(int));
```

## Debugging Memory Issues

### Valgrind

```bash
# Detect leaks, use-after-free, uninit reads
valgrind --leak-check=full --show-leak-kinds=all ./my_program

# Track origins of uninitialized values
valgrind --track-origins=yes ./my_program
```

### Address Sanitizer (ASan)

```bash
# Compile with ASan
gcc -fsanitize=address -fno-omit-frame-pointer -g -O1 -o prog prog.c

# Run -- ASan reports errors at runtime
./prog
```

ASan detects: heap/stack/global buffer overflow, use-after-free, double-free, memory leaks.

### Memory Sanitizer (MSan)

```bash
# Detects use of uninitialized memory (Clang only)
clang -fsanitize=memory -fno-omit-frame-pointer -g -O1 -o prog prog.c
```

### Custom Debug Allocator

```c
#ifdef DEBUG
#include <stdio.h>

static size_t total_allocs = 0;
static size_t total_frees  = 0;

void *debug_malloc(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    fprintf(stderr, "[ALLOC] %p (%zu bytes) at %s:%d\n", ptr, size, file, line);
    total_allocs++;
    return ptr;
}

void debug_free(void *ptr, const char *file, int line) {
    fprintf(stderr, "[FREE]  %p at %s:%d\n", ptr, file, line);
    total_frees++;
    free(ptr);
}

void debug_report(void) {
    fprintf(stderr, "Allocations: %zu, Frees: %zu, Leaked: %zu\n",
            total_allocs, total_frees, total_allocs - total_frees);
}

#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr)    debug_free(ptr, __FILE__, __LINE__)
#endif
```

## Ownership Conventions

Document ownership in function comments:

```c
/**
 * Creates a new widget.
 * @return Newly allocated Widget. Caller takes ownership; free with widget_destroy().
 */
Widget *widget_create(void);

/**
 * Processes data.
 * @param data Borrowed reference -- caller retains ownership.
 */
void process(const char *data);

/**
 * Sets the widget name.
 * @param name Ownership transferred to widget. Caller must not free.
 */
void widget_set_name(Widget *w, char *name);

/**
 * Gets the widget name.
 * @return Borrowed reference -- valid until widget is modified or destroyed.
 */
const char *widget_get_name(const Widget *w);
```
