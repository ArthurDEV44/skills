# Modern C Features (C11 / C17 / C23)

## C11 Features

### static_assert (Compile-Time Assertions)

```c
#include <assert.h>

// Verify assumptions at compile time -- zero runtime cost
static_assert(sizeof(int) == 4, "int must be 32 bits");
static_assert(sizeof(void *) == 8, "64-bit pointers required");

// Verify struct layout
struct Packet {
    uint8_t  type;
    uint8_t  flags;
    uint16_t length;
    uint32_t payload;
};
static_assert(sizeof(struct Packet) == 8, "Packet must be 8 bytes");
static_assert(offsetof(struct Packet, payload) == 4, "payload at offset 4");
```

### _Generic (Type-Generic Selection)

Compile-time dispatch based on expression type:

```c
#include <math.h>
#include <stdio.h>

// Type-generic abs
#define abs_val(x) _Generic((x),    \
    int:        abs,                 \
    long:       labs,                \
    long long:  llabs,               \
    float:      fabsf,              \
    double:     fabs,               \
    long double:fabsl               \
)(x)

// Type-generic print
#define print_val(x) _Generic((x),       \
    int:         printf("%d\n", (x)),     \
    double:      printf("%f\n", (x)),     \
    const char*: printf("%s\n", (x)),     \
    char*:       printf("%s\n", (x))      \
)

// Type-generic max
#define max(a, b) _Generic((a) + (b),    \
    int:    max_int,                      \
    float:  max_float,                    \
    double: max_double                    \
)((a), (b))

static inline int    max_int(int a, int b)       { return a > b ? a : b; }
static inline float  max_float(float a, float b) { return a > b ? a : b; }
static inline double max_double(double a, double b) { return a > b ? a : b; }
```

### _Alignas and _Alignof

```c
#include <stdalign.h>

// Force alignment
_Alignas(16) float vector[4];  // 16-byte aligned for SIMD
alignas(64) char cache_line_data[64];  // cache-line aligned

// Query alignment
size_t align = alignof(double);     // typically 8
size_t align = _Alignof(max_align_t);  // max fundamental alignment

// Aligned struct member
struct AlignedBuffer {
    alignas(4096) char page[4096];  // page-aligned
};
```

### _Noreturn

```c
#include <stdnoreturn.h>

noreturn void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    abort();
}

// C23: use [[noreturn]] attribute instead
[[noreturn]] void die(const char *msg);
```

### Anonymous Structs and Unions

```c
struct Vector3 {
    union {
        struct { float x, y, z; };     // anonymous struct
        float components[3];           // array access
    };                                 // anonymous union
};

struct Vector3 v = { .x = 1.0f, .y = 2.0f, .z = 3.0f };
printf("%f\n", v.components[0]);  // same as v.x
```

### _Thread_local

```c
#include <threads.h>

// Each thread gets its own copy
_Thread_local int thread_errno = 0;
// C23: thread_local is a keyword (no underscore prefix needed)
```

## C17 Features

C17 (ISO/IEC 9899:2018) is primarily a bug-fix release. Key clarifications:

- `__STDC_VERSION__` defined as `201710L`
- No new language features; corrects defects in C11
- Clarified behavior of `register` keyword, `gets` removal, etc.
- Implementations that conform to C17 automatically conform to C11 intent

## C23 Features

### typeof and typeof_unqual

Deduce type of an expression at compile time:

```c
// typeof preserves qualifiers
const int x = 42;
typeof(x) y = 10;            // y is const int

// typeof_unqual strips qualifiers
typeof_unqual(x) z = 10;     // z is int (mutable)

// Practical use: type-safe macros
#define SWAP(a, b) do {           \
    typeof(a) _tmp = (a);         \
    (a) = (b);                    \
    (b) = _tmp;                   \
} while (0)

// Works with any type
int a = 1, b = 2;
SWAP(a, b);
double x = 1.0, y = 2.0;
SWAP(x, y);
```

### nullptr and nullptr_t

Proper null pointer constant (no more `(void *)0` ambiguity):

```c
#include <stddef.h>

int *p = nullptr;          // clear intent
if (p == nullptr) { }

// nullptr has type nullptr_t
nullptr_t null_val = nullptr;

// Advantage over NULL/0: won't accidentally match int overloads
// in _Generic or be confused with integer 0
```

### constexpr

Compile-time evaluated constants:

```c
constexpr int BUFFER_SIZE = 4096;
constexpr double PI = 3.14159265358979323846;

// Can be used in array bounds and static_assert
char buffer[BUFFER_SIZE];
static_assert(BUFFER_SIZE > 0, "buffer size must be positive");

// Replaces many #define constants -- type-safe and scoped
```

### auto Type Inference

```c
// auto deduces type from initializer (C23)
auto x = 42;       // int
auto pi = 3.14;    // double
auto msg = "hello"; // const char *

// Useful in macros
#define DECLARE_AND_INIT(name, value) auto name = (value)
```

### Attributes

```c
// [[nodiscard]] -- warn if return value is ignored
[[nodiscard]] int compute_result(void);
compute_result();  // compiler warning!

// [[maybe_unused]] -- suppress unused warnings
void callback([[maybe_unused]] void *ctx) {
    // ctx not used yet, but suppress warning
}

// [[deprecated]] -- mark as deprecated
[[deprecated("use new_function instead")]]
void old_function(void);

// [[fallthrough]] -- intentional switch fallthrough
switch (x) {
    case 1:
        do_one();
        [[fallthrough]];
    case 2:
        do_two();
        break;
}

// [[noreturn]] -- replaces _Noreturn
[[noreturn]] void panic(const char *msg);

// [[reproducible]] and [[unsequenced]] (C23)
// Hints for compiler optimization of function calls
[[unsequenced]] int pure_func(int x);      // no side effects, no state
[[reproducible]] int stable_func(int x);   // same args -> same result
```

### Improved Enums

```c
// Fixed underlying type (C23)
enum Color : unsigned char {
    RED   = 0,
    GREEN = 1,
    BLUE  = 2,
};
static_assert(sizeof(enum Color) == 1, "");

// Enhanced enumerator value control
enum BigValues : long long {
    BIG = 1LL << 40,
};
```

### #embed Directive

Include binary data at compile time:

```c
// Embed file contents as array initializer
static const unsigned char icon[] = {
    #embed "icon.png"
};
static_assert(sizeof icon > 0, "icon data loaded");

// With limit
static const char license[1024] = {
    #embed "LICENSE" limit(1024)
};
```

### Digit Separators

```c
int million   = 1'000'000;
long big      = 0xFF'FF'FF'FF;
double pi     = 3.14159'26535'89793;
int binary    = 0b1010'1100'0011'1001;
```

### Binary Integer Literals

```c
int flags = 0b00001111;
int mask  = 0b1010'0101;
unsigned byte = 0b11001100;
```

### Improved Compound Literals

```c
// Storage class specifiers on compound literals (C23)
static const int *ptr = (static const int[]){1, 2, 3};

// constexpr compound literals
constexpr struct Point origin = (struct Point){.x = 0, .y = 0};
```

### free_sized and free_aligned_sized (C23)

```c
// Deallocation with size hint (enables allocator optimizations)
void *p = malloc(1024);
free_sized(p, 1024);

// Aligned deallocation
void *q = aligned_alloc(64, 1024);
free_aligned_sized(q, 64, 1024);
```

### Other C23 Additions

- **`unreachable()`**: `#include <stddef.h>` -- marks unreachable code (UB if reached, enables optimization)
- **`char8_t`**: UTF-8 character type
- **`#warning`**: Standard preprocessor warning directive
- **`_BitInt(N)`**: Fixed-width integers of arbitrary bit width
- **`ckd_add`, `ckd_sub`, `ckd_mul`**: Checked integer arithmetic (`#include <stdckdint.h>`)
- **`memset_explicit`**: Guaranteed not to be optimized away (for clearing secrets)
- **`strdup`, `strndup`**: Finally standardized (were POSIX-only)
- **Variadic `va_start`**: No longer requires a second argument

```c
#include <stdckdint.h>

// Checked arithmetic -- returns true on overflow
int result;
if (ckd_add(&result, a, b)) {
    // overflow occurred
}

#include <string.h>

// memset_explicit -- never optimized away
char secret[128];
/* use secret */
memset_explicit(secret, 0, sizeof secret);  // guaranteed to zero
```
