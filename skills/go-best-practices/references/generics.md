# Go Generics (Go 1.18+)

## Type Parameters

```go
func Min[T cmp.Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}

// Usage -- type inferred
result := Min(3, 5)       // int
result := Min(3.0, 5.0)   // float64
```

## Constraints

### Built-in Constraints (Go 1.21+ `cmp` package)

```go
import "cmp"

// cmp.Ordered: any type that supports < <= >= > == !=
func Max[T cmp.Ordered](a, b T) T { /* ... */ }
```

### Common Constraint Interfaces

```go
import "golang.org/x/exp/constraints"

// constraints.Integer -- all integer types
// constraints.Float -- all float types
// constraints.Signed -- signed integers
// constraints.Unsigned -- unsigned integers
// constraints.Ordered -- ordered types (integers, floats, strings)
// constraints.Complex -- complex types
```

### Custom Constraints

```go
// Type set: union of types
type Number interface {
    ~int | ~int32 | ~int64 | ~float32 | ~float64
}

// ~ means "underlying type" -- allows named types
type Celsius float64  // satisfies ~float64

// Constraint with methods
type Stringer interface {
    comparable
    String() string
}
```

### The `comparable` Constraint

```go
// comparable: types that support == and != (usable as map keys)
func Contains[T comparable](slice []T, target T) bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}
```

### The `any` Constraint

```go
// any == interface{} -- no constraint at all
func PrintAll[T any](items []T) {
    for _, item := range items {
        fmt.Println(item)
    }
}
```

## Generic Types

```go
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// Usage
s := &Stack[int]{}
s.Push(42)
```

## Generic Interfaces

```go
type Container[T any] interface {
    Len() int
    Get(index int) T
    Set(index int, val T)
}
```

## slices and maps Packages (Go 1.21+)

Standard library generic functions:

```go
import "slices"

slices.Sort(s)                    // sort in place
slices.SortFunc(s, cmpFunc)       // sort with custom comparator
slices.Contains(s, val)           // membership check
slices.Index(s, val)              // find index
slices.Compact(s)                 // deduplicate consecutive
slices.Reverse(s)                 // reverse in place
slices.Clone(s)                   // shallow copy
slices.Equal(a, b)                // element-wise equality
slices.BinarySearch(s, val)       // binary search (sorted slice)
slices.Grow(s, n)                 // pre-grow capacity
slices.Delete(s, i, j)            // delete s[i:j]
slices.Insert(s, i, vals...)      // insert at index
slices.Replace(s, i, j, vals...)  // replace s[i:j]

import "maps"

maps.Keys(m)                      // all keys as slice
maps.Values(m)                    // all values as slice
maps.Clone(m)                     // shallow copy
maps.Equal(a, b)                  // element-wise equality
maps.DeleteFunc(m, predicate)     // conditional delete
maps.Copy(dst, src)               // merge src into dst
```

## When to Use Generics

**Good uses:**
- Functions that operate on slices, maps, channels of any element type
- General-purpose data structures (stacks, queues, trees, sets)
- Functions where the logic is identical across types (Min, Max, Contains, Filter, Map)

**Avoid generics when:**
- A simple interface works fine (`io.Reader`, `fmt.Stringer`)
- The implementation differs significantly per type
- It only saves one or two duplicate functions
- `any` or `interface{}` would work just as well (generics add no type safety if unconstrained)

**Rule of thumb:** If you're writing the same function body for 3+ types, consider generics. If you're writing different logic per type, use interfaces or type switches.

## Type Inference

Go infers type parameters from arguments when possible:

```go
// Explicit (rarely needed)
result := Min[int](3, 5)

// Inferred (preferred)
result := Min(3, 5)
```

Type inference doesn't work for:
- Generic types with no function arguments: `s := Stack[int]{}`
- Constraints that can't be inferred from arguments
