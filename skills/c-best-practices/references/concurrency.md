# C Concurrency (C11 Threads & Atomics)

## C11 Threads (`<threads.h>`)

### Thread Creation and Joining

```c
#include <threads.h>
#include <stdio.h>

typedef struct {
    int  id;
    int  result;
} ThreadArg;

int worker(void *arg) {
    ThreadArg *ta = arg;
    printf("Thread %d running\n", ta->id);
    ta->result = ta->id * 42;
    return thrd_success;
}

int main(void) {
    enum { NUM_THREADS = 4 };
    thrd_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].id = i;
        if (thrd_create(&threads[i], worker, &args[i]) != thrd_success) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            return 1;
        }
    }

    // Join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        int res;
        thrd_join(threads[i], &res);
        printf("Thread %d result: %d\n", i, args[i].result);
    }
    return 0;
}
```

### Mutex

```c
#include <threads.h>

mtx_t mutex;
int shared_counter = 0;

int increment_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++) {
        mtx_lock(&mutex);
        shared_counter++;
        mtx_unlock(&mutex);
    }
    return thrd_success;
}

int main(void) {
    mtx_init(&mutex, mtx_plain);

    thrd_t t1, t2;
    thrd_create(&t1, increment_worker, NULL);
    thrd_create(&t2, increment_worker, NULL);

    thrd_join(t1, NULL);
    thrd_join(t2, NULL);

    printf("Counter: %d\n", shared_counter);  // 200000

    mtx_destroy(&mutex);
    return 0;
}
```

### Mutex Types

```c
// Plain mutex (no recursion, no timeout)
mtx_init(&mtx, mtx_plain);

// Recursive mutex (same thread can lock multiple times)
mtx_init(&mtx, mtx_plain | mtx_recursive);

// Timed mutex
mtx_init(&mtx, mtx_timed);
struct timespec ts;
timespec_get(&ts, TIME_UTC);
ts.tv_sec += 5;  // 5-second timeout
if (mtx_timedlock(&mtx, &ts) == thrd_timedout) {
    /* lock acquisition timed out */
}
```

### Condition Variables

```c
#include <threads.h>
#include <stdbool.h>

mtx_t mtx;
cnd_t cnd;
bool data_ready = false;
int  shared_data = 0;

int producer(void *arg) {
    (void)arg;
    mtx_lock(&mtx);
    shared_data = 42;
    data_ready = true;
    cnd_signal(&cnd);   // wake one waiter
    mtx_unlock(&mtx);
    return thrd_success;
}

int consumer(void *arg) {
    (void)arg;
    mtx_lock(&mtx);
    while (!data_ready) {            // always use a while loop (spurious wakeups)
        cnd_wait(&cnd, &mtx);
    }
    printf("Got data: %d\n", shared_data);
    mtx_unlock(&mtx);
    return thrd_success;
}

int main(void) {
    mtx_init(&mtx, mtx_plain);
    cnd_init(&cnd);

    thrd_t prod, cons;
    thrd_create(&cons, consumer, NULL);
    thrd_create(&prod, producer, NULL);

    thrd_join(prod, NULL);
    thrd_join(cons, NULL);

    cnd_destroy(&cnd);
    mtx_destroy(&mtx);
    return 0;
}
```

### Thread-Local Storage

```c
// C11 keyword
_Thread_local int tls_counter = 0;

// C23 keyword (no underscore)
thread_local int tls_counter = 0;

// Runtime TLS (dynamic)
tss_t key;
tss_create(&key, free);          // free is the destructor

char *data = malloc(256);
snprintf(data, 256, "thread-specific");
tss_set(key, data);

char *val = tss_get(key);        // per-thread value

tss_delete(key);
```

### call_once

```c
once_flag init_flag = ONCE_FLAG_INIT;

void initialize(void) {
    printf("Initializing (once)\n");
    /* expensive one-time setup */
}

int worker(void *arg) {
    (void)arg;
    call_once(&init_flag, initialize);
    /* ... work ... */
    return thrd_success;
}
```

## Atomic Operations (`<stdatomic.h>`)

### Basic Atomics

```c
#include <stdatomic.h>

atomic_int counter = 0;

int worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++) {
        atomic_fetch_add(&counter, 1);  // lock-free increment
    }
    return thrd_success;
}

// After all threads join:
int val = atomic_load(&counter);
```

### Atomic Types

```c
// Predefined atomic types
atomic_bool     flag;
atomic_int      count;
atomic_long     big_count;
atomic_size_t   size;
atomic_intptr_t ptr_val;

// Generic atomic wrapper
_Atomic int x = 0;
_Atomic(struct Point) point;  // any type (may not be lock-free)

// Check if lock-free
if (atomic_is_lock_free(&x)) {
    /* hardware-supported atomic */
}
```

### Atomic Operations

```c
atomic_int val = 0;

// Store and load
atomic_store(&val, 42);
int v = atomic_load(&val);

// Read-modify-write
atomic_fetch_add(&val, 10);     // add, return old value
atomic_fetch_sub(&val, 5);      // subtract
atomic_fetch_or(&val, 0xFF);    // bitwise OR
atomic_fetch_and(&val, 0x0F);   // bitwise AND
atomic_fetch_xor(&val, 0xAA);   // bitwise XOR

// Compare-and-swap (CAS) -- fundamental lock-free primitive
int expected = 42;
bool success = atomic_compare_exchange_strong(&val, &expected, 100);
// If val == 42: sets val = 100, returns true
// If val != 42: sets expected = current val, returns false

// Weak CAS (can spuriously fail -- use in loops)
while (!atomic_compare_exchange_weak(&val, &expected, new_val)) {
    expected = /* re-read or recalculate */;
}

// Exchange
int old = atomic_exchange(&val, 99);  // set to 99, return old value
```

### Memory Ordering

```c
// Relaxed: no ordering guarantees, only atomicity
atomic_store_explicit(&flag, 1, memory_order_relaxed);
int v = atomic_load_explicit(&counter, memory_order_relaxed);

// Acquire/Release: synchronize producer-consumer
// Producer
atomic_store_explicit(&data, 42, memory_order_relaxed);
atomic_store_explicit(&ready, true, memory_order_release);  // fence: all prior writes visible

// Consumer
while (!atomic_load_explicit(&ready, memory_order_acquire));  // fence: all subsequent reads see writes
int d = atomic_load_explicit(&data, memory_order_relaxed);    // guaranteed to see 42

// Sequential consistency (default) -- strongest, safest, slowest
atomic_store(&val, 42);  // equivalent to memory_order_seq_cst
int v = atomic_load(&val);
```

### Memory Order Summary

| Order | Guarantee | Use Case |
|-------|-----------|----------|
| `relaxed` | Atomicity only | Counters, statistics |
| `acquire` | Reads after this see prior writes | Consumer side |
| `release` | Writes before this visible to acquirers | Producer side |
| `acq_rel` | Both acquire and release | Read-modify-write |
| `seq_cst` | Total global order (default) | When in doubt |

### Atomic Fences

```c
// Standalone fence (not tied to a specific atomic variable)
atomic_thread_fence(memory_order_acquire);
atomic_thread_fence(memory_order_release);
atomic_thread_fence(memory_order_seq_cst);

// Signal fence (for signal handler communication)
atomic_signal_fence(memory_order_release);
```

## Lock-Free Patterns

### Lock-Free Stack (LIFO)

```c
typedef struct Node {
    int data;
    struct Node *next;
} Node;

_Atomic(Node *) stack_top = NULL;

void push(int value) {
    Node *new_node = malloc(sizeof *new_node);
    new_node->data = value;
    new_node->next = atomic_load(&stack_top);
    while (!atomic_compare_exchange_weak(&stack_top, &new_node->next, new_node))
        ;  // retry on CAS failure
}

int pop(int *value) {
    Node *top = atomic_load(&stack_top);
    while (top) {
        if (atomic_compare_exchange_weak(&stack_top, &top, top->next)) {
            *value = top->data;
            free(top);  // caution: ABA problem in real code
            return 1;
        }
    }
    return 0;  // empty
}
```

### Spinlock

```c
typedef atomic_flag Spinlock;

void spin_lock(Spinlock *lock) {
    while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire))
        ;  // busy-wait
}

void spin_unlock(Spinlock *lock) {
    atomic_flag_clear_explicit(lock, memory_order_release);
}

// Usage
Spinlock lock = ATOMIC_FLAG_INIT;
spin_lock(&lock);
/* critical section */
spin_unlock(&lock);
```

## POSIX Threads (pthreads)

When `<threads.h>` isn't available or you need more features:

```c
#include <pthread.h>

pthread_t thread;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond  = PTHREAD_COND_INITIALIZER;

void *worker(void *arg) {
    /* ... */
    return NULL;
}

// Create
pthread_create(&thread, NULL, worker, arg);

// Join
pthread_join(thread, &result);

// Mutex
pthread_mutex_lock(&mutex);
/* critical section */
pthread_mutex_unlock(&mutex);

// Read-write lock
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
pthread_rwlock_rdlock(&rwlock);  // multiple readers
pthread_rwlock_wrlock(&rwlock);  // exclusive writer
pthread_rwlock_unlock(&rwlock);
```

## Common Concurrency Pitfalls

```c
// DEADLOCK: lock ordering
mtx_lock(&mtx_a);
mtx_lock(&mtx_b);  // if another thread locks b then a: deadlock
// FIX: always lock in the same global order

// DATA RACE: unsynchronized access
int shared = 0;
// Thread 1: shared++;
// Thread 2: shared++;
// FIX: use mutex or atomic

// RACE CONDITION: check-then-act
mtx_lock(&mtx);
if (queue_size > 0) {
    mtx_unlock(&mtx);
    // Another thread may empty the queue here!
    item = dequeue();  // may fail
}
// FIX: keep lock held during entire operation

// SPURIOUS WAKEUP: missing while loop
cnd_wait(&cnd, &mtx);       // BAD: may wake without signal
while (!ready) cnd_wait(&cnd, &mtx);  // GOOD: re-check predicate
```
