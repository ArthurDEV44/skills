# C Build Tooling & Compilation

## Compiler Flags

### GCC / Clang -- Recommended Warning Flags

```bash
# Strict warnings -- catch bugs early
gcc -Wall -Wextra -Wpedantic -Werror -std=c17 -o prog prog.c

# Full paranoid mode
gcc -Wall -Wextra -Wpedantic -Werror \
    -Wshadow -Wconversion -Wsign-conversion \
    -Wstrict-prototypes -Wold-style-definition \
    -Wmissing-prototypes -Wmissing-declarations \
    -Wnull-dereference -Wdouble-promotion \
    -Wformat=2 -Wformat-overflow=2 -Wformat-truncation=2 \
    -Wundef -Wunused -Wwrite-strings \
    -Wcast-align -Wcast-qual -Wpointer-arith \
    -Wswitch-enum -Wunreachable-code \
    -fstack-protector-strong \
    -std=c17 -o prog prog.c

# C23 mode
gcc -std=c2x -o prog prog.c     # GCC
clang -std=c23 -o prog prog.c   # Clang
```

### Optimization Levels

```bash
# Debug build (no optimization, full debug info)
gcc -O0 -g3 -o debug_prog prog.c

# Release build (optimized, some debug info)
gcc -O2 -g -DNDEBUG -o prog prog.c

# Maximum optimization (may change behavior of UB-dependent code)
gcc -O3 -march=native -DNDEBUG -o fast_prog prog.c

# Optimize for size
gcc -Os -o small_prog prog.c

# Link-time optimization
gcc -O2 -flto -o prog prog.c
```

### Key Flag Explanations

| Flag | Purpose |
|------|---------|
| `-Wall` | Enable most common warnings |
| `-Wextra` | Additional useful warnings |
| `-Wpedantic` | Strict ISO C compliance |
| `-Werror` | Treat warnings as errors |
| `-Wshadow` | Warn when variable shadows another |
| `-Wconversion` | Warn on implicit type conversions |
| `-Wformat=2` | Thorough printf/scanf format checks |
| `-Wnull-dereference` | Warn on possible NULL dereference paths |
| `-fstack-protector-strong` | Stack buffer overflow detection |
| `-D_FORTIFY_SOURCE=2` | Runtime buffer overflow checks (requires `-O1+`) |
| `-DNDEBUG` | Disable `assert()` in release builds |

## Sanitizers

### Address Sanitizer (ASan)

Detects: heap/stack/global buffer overflow, use-after-free, double-free, leaks.

```bash
gcc -fsanitize=address -fno-omit-frame-pointer -g -O1 -o prog prog.c
./prog

# With leak detection
ASAN_OPTIONS=detect_leaks=1 ./prog
```

### Undefined Behavior Sanitizer (UBSan)

Detects: signed overflow, null dereference, misaligned access, shift overflow, etc.

```bash
gcc -fsanitize=undefined -fno-omit-frame-pointer -g -O1 -o prog prog.c
./prog

# Specific checks
gcc -fsanitize=signed-integer-overflow,null,alignment,shift -o prog prog.c
```

### Thread Sanitizer (TSan)

Detects: data races, lock-order inversions.

```bash
gcc -fsanitize=thread -g -O1 -o prog prog.c
./prog
```

### Memory Sanitizer (MSan) -- Clang Only

Detects: use of uninitialized memory.

```bash
clang -fsanitize=memory -fno-omit-frame-pointer -g -O1 -o prog prog.c
./prog
```

### Combining Sanitizers

```bash
# ASan + UBSan (compatible)
gcc -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1 -o prog prog.c

# NOTE: ASan and TSan cannot be combined
# NOTE: ASan and MSan cannot be combined
```

## Valgrind

```bash
# Memory errors (leaks, use-after-free, uninit reads)
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./prog

# Cache profiling
valgrind --tool=cachegrind ./prog
cg_annotate cachegrind.out.12345

# Call graph profiling
valgrind --tool=callgrind ./prog
callgrind_annotate callgrind.out.12345

# Thread error detection
valgrind --tool=helgrind ./prog
valgrind --tool=drd ./prog
```

## Static Analysis

### Clang Static Analyzer

```bash
# Single file
scan-build gcc -c prog.c

# Entire project
scan-build make

# Generate HTML report
scan-build -o reports/ make
```

### Cppcheck

```bash
# Basic analysis
cppcheck --enable=all --std=c17 prog.c

# Suppress specific warnings
cppcheck --enable=all --suppress=unusedFunction prog.c

# Check entire project
cppcheck --enable=all --std=c17 src/
```

### GCC Static Analysis

```bash
# GCC 10+ built-in static analyzer
gcc -fanalyzer -Wall -o prog prog.c
```

## Build Systems

### Makefile (Minimal)

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -std=c17
LDFLAGS =
LDLIBS  =

# Debug/Release toggle
ifdef DEBUG
CFLAGS += -O0 -g3 -fsanitize=address,undefined
LDFLAGS += -fsanitize=address,undefined
else
CFLAGS += -O2 -DNDEBUG
endif

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
BIN = myprogram

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

-include $(OBJ:.o=.d)

clean:
	rm -f $(OBJ) $(OBJ:.o=.d) $(BIN)

.PHONY: all clean
```

### CMake (Minimal)

```cmake
cmake_minimum_required(VERSION 3.20)
project(myproject C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizers in debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

add_executable(myprogram src/main.c src/module.c)
target_include_directories(myprogram PRIVATE include)
```

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release build
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

## Debugging with GDB

```bash
# Compile with debug info
gcc -g3 -O0 -o prog prog.c

# Run under GDB
gdb ./prog

# Essential GDB commands
# break main          -- set breakpoint at main
# break file.c:42     -- breakpoint at line 42
# run                 -- start program
# next / step         -- step over / step into
# print var           -- inspect variable
# bt                  -- backtrace
# watch var           -- break when var changes
# info locals         -- show local variables
# continue            -- resume execution

# Run with arguments
gdb --args ./prog arg1 arg2

# Examine core dump
gcc -g3 -o prog prog.c
ulimit -c unlimited
./prog  # crashes, produces core file
gdb ./prog core
```

## Formatting

### clang-format

```yaml
# .clang-format
BasedOnStyle: LLVM
IndentWidth: 4
UseTab: Never
BreakBeforeBraces: Linux
AllowShortFunctionsOnASingleLine: None
ColumnLimit: 100
SortIncludes: true
```

```bash
# Format in place
clang-format -i src/*.c include/*.h

# Check formatting (CI)
clang-format --dry-run --Werror src/*.c
```

## Project Structure

```
project/
├── CMakeLists.txt (or Makefile)
├── include/
│   └── project/
│       ├── module_a.h
│       └── module_b.h
├── src/
│   ├── main.c
│   ├── module_a.c
│   └── module_b.c
├── tests/
│   ├── test_module_a.c
│   └── test_module_b.c
├── .clang-format
└── .gitignore
```

### Header Organization

```c
/* module.h */
#ifndef PROJECT_MODULE_H
#define PROJECT_MODULE_H

#include <stddef.h>   /* system headers first */
#include <stdbool.h>

/* Public types */
typedef struct Module Module;

/* Public API */
Module *module_create(void);
void    module_destroy(Module *m);
int     module_process(Module *m, const void *data, size_t len);

#endif /* PROJECT_MODULE_H */
```

```c
/* module.c */
#include "project/module.h"  /* own header first */

#include <stdlib.h>          /* system headers */
#include <string.h>

/* Private types and functions */
struct Module {
    char *buffer;
    size_t size;
};

static int internal_helper(Module *m) {
    /* ... */
}

/* Public API implementation */
Module *module_create(void) {
    Module *m = calloc(1, sizeof *m);
    return m;
}
```
