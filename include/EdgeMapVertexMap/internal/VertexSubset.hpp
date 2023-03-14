#pragma once
#include "BitArray.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include <cstdint>
#include <cstdio>
namespace EdgeMapVertexMap {
template <class node_t> class VertexSubset {
  bool all = false;
  bool is_sparse = false;
  uint64_t max_el = 0;
  BitArray *ba = nullptr;
  ParallelTools::Reducer_Vector<node_t> *queue = nullptr;

public:
  [[nodiscard]] bool sparse() const { return is_sparse; }
  [[nodiscard]] bool complete() const { return all; }
  [[nodiscard]] bool has(node_t i) const {
    if (all) {
      return true;
    }
    if (is_sparse) {
      printf("shouldn't be calling has, is currently sparse\n");
      __builtin_unreachable();
      return false;
    } else {
      return ba->get(i);
    }
  }
  [[nodiscard]] bool has_dense_no_all(node_t i) const { return ba->get(i); }
  void has_dense_no_all_prefetch(node_t i) const { return ba->prefetch(i); }

  [[nodiscard]] size_t get_n() const {
    if (all) {
      return max_el;
    } else if (is_sparse) {
      return queue->size();
    } else {
      return ba->count();
    }
  }
  [[nodiscard]] bool non_empty() const {
    if (all) {
      return true;
    } else if (is_sparse) {
      return !queue->empty();
    } else {
      return ba->non_empty();
    }
  }
  void print() const {
    printf("all = %d, is_sparse = %d, max_el = %lu, size = %lu\n", all,
           is_sparse, max_el, get_n());
    printf("queue = %p, ba = %p\n", queue, ba);
    if (get_n() > 100) {
      return;
    }
    if (all) {
      return;
    }
    if (is_sparse) {
      printf("{");
      queue->serial_for_each([](node_t e) { printf("%d, ", e); });
      printf("}\n");
    } else {
      printf("{");
      for (node_t i = 0; i < max_el; i++) {
        if (ba->get(i)) {
          printf("%d, ", i);
        }
      }
      printf("}\n");
    }
  }
  void insert(node_t i) const {
    if (is_sparse) {
      queue->push_back(i);
      return;
    }
    return ba->set(i);
  }
  void insert_dense(node_t i) const { return ba->set(i); }
  void insert_sparse(node_t i) const { queue->push_back(i); }

  template <class F> void map_sparse(F &f) const {
    // printf("queue in map = %p\n", queue);
    queue->for_each([&](node_t item) { f(item); });
  }
  template <class F> void map(F f) {
    if (all) {
      ParallelTools::parallel_for(0, max_el, [&](node_t i) { f(i); });
      return;
    }
    if (is_sparse) {
      return map_sparse(f);
    } else {
      return ba->map(f);
    }
  }

  // used to return empty vertexsubsets when we have no output
  VertexSubset() {}

  VertexSubset(node_t e, uint64_t max_el_, bool all_ = false)
      : all(all_), is_sparse(true), max_el(max_el_) {
    if (all) {
      is_sparse = false;
      return;
    }
    queue = new ParallelTools::Reducer_Vector<node_t>();
    queue->push_back(e);
  }
  VertexSubset(bool const *const els, node_t len)
      : all(false), is_sparse(false), max_el(len) {
    ba = new BitArray(max_el);
    ParallelTools::parallel_for(
        0, max_el,
        [&](node_t i) {
          if (els[i]) {
            ba->set(i);
          }
        },
        256);
  }

  VertexSubset(const VertexSubset &other)
      : all(other.all), is_sparse(other.is_sparse), max_el(other.max_el),
        ba(other.ba), queue(other.queue) {
    // printf("queue = %p\n", queue);
  }
  VertexSubset &operator=(const VertexSubset &other) {
    all = other.all;
    is_sparse = other.is_sparse;
    max_el = other.max_el;
    ba = other.ba;
    queue = other.queue;
    return *this;
  }

  VertexSubset empty_version_for_insert() {
    VertexSubset vs;
    vs.all = false;
    vs.is_sparse = is_sparse;
    vs.max_el = max_el;
    if (is_sparse) {
      vs.queue = new ParallelTools::Reducer_Vector<node_t>();
    } else {
      vs.ba = new BitArray(max_el);
    }
    return vs;
  }

  void del() {
    if (ba != nullptr) {
      delete ba;
      ba = nullptr;
    }
    if (queue != nullptr) {
      delete queue;
      // printf("deleteing queue %p\n", queue);
      queue = nullptr;
    }
  }

  [[nodiscard]] VertexSubset convert_to_dense() const {
    VertexSubset vs;
    if (all || !is_sparse) {
      return vs;
    }
    vs.all = all;
    vs.max_el = max_el;
    vs.queue = nullptr;
    vs.is_sparse = false;
    vs.ba = new BitArray(max_el);
    queue->for_each([&](node_t item) { vs.ba->set_atomic(item); });
    return vs;
  }

  [[nodiscard]] VertexSubset convert_to_sparse() const {
    VertexSubset vs;
    if (all || is_sparse) {
      return vs;
    }
    vs.all = all;
    vs.max_el = max_el;
    vs.is_sparse = true;
    vs.queue = new ParallelTools::Reducer_Vector<node_t>();
    vs.ba = nullptr;
    ParallelTools::parallel_for(0, max_el, [&](node_t i) {
      if (ba->get(i)) {
        vs.queue->push_back(i);
      }
    });
    return vs;
  }

  uint64_t get_out_degree(const auto &G, const auto &d) {
    ParallelTools::Reducer_sum<uint64_t> out_degree;
    map([&](auto el) { out_degree += G.get_degree(el, d); });
    return out_degree;
  }
};
} // namespace EdgeMapVertexMap
