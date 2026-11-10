#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

/*
 * Count malloc/calloc/realloc calls at or above ALLOC_COUNT_MIN bytes.
 * Loaded with LD_PRELOAD. Counting stays off until alloc_counter_enable(1).
 */

static void *(*real_malloc)(size_t) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static atomic_ullong count = 0;
static atomic_int enabled = 0;
static size_t threshold = 0;
static int initializing = 0;
static char bootstrap[1 << 20];
static size_t bootstrap_used = 0;

static void init_real(void) {
    initializing = 1;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    const char *env = getenv("ALLOC_COUNT_MIN");
    if (env && *env) threshold = (size_t)strtoull(env, NULL, 10);
    initializing = 0;
}

static void bump(size_t n) {
    if (atomic_load(&enabled) && n >= threshold) atomic_fetch_add(&count, 1);
}

void *malloc(size_t n) {
    if (!real_malloc) {
        if (initializing) {
            size_t aligned = (n + 15u) & ~15u;
            if (bootstrap_used + aligned > sizeof(bootstrap)) return NULL;
            void *p = bootstrap + bootstrap_used;
            bootstrap_used += aligned;
            return p;
        }
        init_real();
    }
    bump(n);
    return real_malloc(n);
}

void *calloc(size_t nmemb, size_t size) {
    if (!real_calloc) {
        if (!real_malloc) {
            if (initializing) {
                size_t n = nmemb * size;
                void *p = malloc(n);
                if (p) memset(p, 0, n);
                return p;
            }
        }
        init_real();
    }
    bump(nmemb * size);
    return real_calloc(nmemb, size);
}

void *realloc(void *ptr, size_t size) {
    if (!real_realloc) init_real();
    bump(size);
    return real_realloc(ptr, size);
}

void alloc_counter_reset(void) { atomic_store(&count, 0); }
void alloc_counter_enable(int v) { atomic_store(&enabled, v); }
unsigned long long alloc_counter_get(void) { return atomic_load(&count); }
