/*
 * gpubridge_pool.c — Milestone 2.5 (spec2.5.md section 7): a pooled
 * device-memory allocator for tensor allocations.
 *
 * Two singly linked lists, both keyed by exact byte size (no size-class
 * bucketing/rounding — spec2.5.md section 5 non-goal #2):
 *
 *   g_free_blocks  every currently-pooled (released, not yet reused) device
 *                  pointer, tagged with its byte size.
 *   g_size_stats   one entry per distinct byte size ever seen, tracking how
 *                  many times a block of that size has been served from the
 *                  pool — this is the real reuse_count signal
 *                  gpuBridgeTensorAlloc feeds into gpuBridgeChooseMemoryStrategy().
 *
 * Both lists are drained together by gpuBridgePoolDrain(): a block held in
 * the pool is only ever actually freed (via the backend's free_device())
 * when the whole pool is drained, not on an individual release — that is
 * the entire point of pooling.
 *
 * Single-threaded only (spec2.5.md section 5 non-goal #9): no locking.
 */
#include "gpubridge_internal.h"

#include <stdlib.h>

typedef struct FreeBlock {
    void* ptr;
    size_t bytes;
    struct FreeBlock* next;
} FreeBlock;

typedef struct SizeStat {
    size_t bytes;
    int served_count;
    struct SizeStat* next;
} SizeStat;

static FreeBlock* g_free_blocks = NULL;
static SizeStat* g_size_stats = NULL;

static SizeStat* find_or_create_size_stat(size_t bytes)
{
    for (SizeStat* s = g_size_stats; s != NULL; s = s->next) {
        if (s->bytes == bytes) {
            return s;
        }
    }

    SizeStat* s = malloc(sizeof(SizeStat));
    s->bytes = bytes;
    s->served_count = 0;
    s->next = g_size_stats;
    g_size_stats = s;
    return s;
}

void* gpuBridgePoolAcquire(size_t bytes, int* out_reuse_count)
{
    SizeStat* stat = find_or_create_size_stat(bytes);

    /* Search for the first free block of exactly this size (spec2.5.md
     * section 5 non-goal #2 — exact-size match only, no splitting/merging). */
    FreeBlock** link = &g_free_blocks;
    while (*link != NULL && (*link)->bytes != bytes) {
        link = &(*link)->next;
    }

    if (*link == NULL) {
        /* Miss: nothing pooled at this size yet. */
        *out_reuse_count = stat->served_count;
        return NULL;
    }

    FreeBlock* node = *link;
    *link = node->next;
    void* ptr = node->ptr;
    free(node);

    stat->served_count++;
    *out_reuse_count = stat->served_count;
    return ptr;
}

void gpuBridgePoolRelease(void* ptr, size_t bytes)
{
    FreeBlock* node = malloc(sizeof(FreeBlock));
    node->ptr = ptr;
    node->bytes = bytes;
    node->next = g_free_blocks;
    g_free_blocks = node;
}

void gpuBridgePoolDrain(const GpuBridgeBackend* backend)
{
    FreeBlock* block = g_free_blocks;
    while (block != NULL) {
        FreeBlock* next = block->next;
        if (backend != NULL) {
            backend->free_device(block->ptr);
        }
        free(block);
        block = next;
    }
    g_free_blocks = NULL;

    SizeStat* stat = g_size_stats;
    while (stat != NULL) {
        SizeStat* next = stat->next;
        free(stat);
        stat = next;
    }
    g_size_stats = NULL;
}
