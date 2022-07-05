#pragma once
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include <cstdint>
#include <cstdlib>
#include <malloc.h>

namespace EdgeMapVertexMap {
class BitArray {
public:
  uint32_t *array;

private:
  uint64_t len;
  bool to_free;

public:
  static inline bool bit_array_get(uint32_t const *const array, uint64_t i) {
    return (array[i / 32] >> i % 32) & 1U;
  }
  static inline void bit_array_prefetch(uint32_t const *const array,
                                        uint64_t i) {
    __builtin_prefetch(&array[i / 32]);
  }
  static inline void bit_array_set(uint32_t *const array, uint64_t i) {
    array[i / 32] |= (1U << i % 32);
  }
  static inline void bit_array_set_atomic(uint32_t *const array, uint64_t i) {
    __atomic_fetch_or(&array[i / 32], 1UL << (i % 32), __ATOMIC_RELAXED);
  }
  static inline void bit_array_flip(uint32_t *const array, uint64_t i) {
    array[i / 32] ^= (1U << i % 32);
  }

  static inline uint64_t bit_array_size(uint64_t size) {
    if (size == 0) {
      return 0;
    }
    if (size < 32) {
      size = 32;
    }
    uint64_t n = size / 32;
    if (n * 32 < size) {
      n += 1;
    }
    return n * 4;
  }
  static uint64_t bit_array_count(uint32_t *array, uint64_t len) {

    uint64_t count = 0;
    for (uint64_t i = 0; i < len / 32; i++) {
      count += __builtin_popcount(array[i]);
    }
    return count;
  }
  static bool bit_array_non_empty(uint32_t const *const array, uint64_t len) {
    for (uint64_t i = 0; i < len / 32; i++) {
      if (array[i] > 0) {
        return true;
      }
    }
    return false;
  }

  void resize_and_clear(uint64_t size) {
    free(array);
    to_free = false;
    if (size > 0) {
      to_free = true;
      uint64_t n = bit_array_size(size);
      array = (uint32_t *)memalign(32, n);
      len = n * 8;
      for (size_t i = 0; i < len / 32; i++) {
        array[i] = 0;
      }
    }
  }

  BitArray(uint32_t *arr, uint64_t size)
      : array(arr), len(size), to_free(false) {}
  BitArray() : array(nullptr), len(0), to_free(false) {}
  explicit BitArray(uint64_t size) {
    uint64_t n = bit_array_size(size);
    array = (uint32_t *)memalign(32, n);
    len = n * 8;
    ParallelTools::parallel_for(0, len / 32, [&](size_t i) { array[i] = 0; });
    to_free = true;
  }
  BitArray(const BitArray &other) {
    len = other.len;
    array = (uint32_t *)memalign(32, len / 8);
    to_free = true;
    ParallelTools::parallel_for(0, len / 32,
                                [&](size_t i) { array[i] = other.array[i]; });
  }
  void clear() const {
    ParallelTools::parallel_for(0, len / 32, [&](size_t i) { array[i] = 0; });
  }

  ~BitArray() {
    if (to_free) {
      free(array);
    }
  }
  [[nodiscard]] bool get(uint64_t i) const { return bit_array_get(array, i); }
  void prefetch(uint64_t i) const { return bit_array_prefetch(array, i); }
  void set(uint64_t i) const { bit_array_set(array, i); }
  void set_atomic(uint64_t i) const { bit_array_set_atomic(array, i); }
  void flip(uint64_t i) const { bit_array_flip(array, i); }
  [[nodiscard]] uint64_t count() const { return bit_array_count(array, len); }
  [[nodiscard]] bool non_empty() const {
    return bit_array_non_empty(array, len);
  }
  template <class F> void map(F &f, bool parallel = true) const {
    if (parallel) {
      ParallelTools::parallel_for(
          0, len,
          [&](size_t i) {
            if (get(i)) {
              f(i);
            }
          },
          256);
    } else {
      for (uint64_t i = 0; i < len; i++) {
        if (get(i)) {
          f(i);
        }
      }
    }
  }

  uint64_t intersection_count(const BitArray &b, uint64_t limit) const {
    ParallelTools::Reducer_sum<uint64_t> count;
    ParallelTools::parallel_for(
        0, limit / 32,
        [&](size_t i) { count.add(__builtin_popcount(array[i] & b.array[i])); },
        256);
    for (uint64_t i = (limit / 32) * 32; i < limit; i++) {
      if (get(i) && b.get(i)) {
        count.inc();
      }
    }
    return count.get();
  }
  [[nodiscard]] uint64_t length() const { return len; }
};
} // namespace EdgeMapVertexMap
