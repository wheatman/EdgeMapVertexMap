#pragma once
#include "BitArray.hpp"
#include "ParallelTools/parallel.h"
#include "gap/sliding_queue.h"
#include <cstdint>
#include <cstdio>

template <class node_t> class VertexSubset {
  bool all = false;
  bool is_sparse = false;
  uint64_t max_el = 0;
  BitArray *ba = nullptr;
  SlidingQueue<node_t> *queue = nullptr;
  QueueBuffer<node_t> *queue_array = nullptr;

public:
  [[nodiscard]] bool sparse() const {
    return is_sparse;
  }[[nodiscard]] bool complete() const {
    return all;
  }
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
  }[[nodiscard]] bool has_dense_no_all(node_t i) const {
    return ba->get(i);
  }
  void has_dense_no_all_prefetch(node_t i) const { return ba->prefetch(i); }

  [[nodiscard]] node_t get_n() const {
    if (all) {
      return max_el;
    } else if (is_sparse) {
      return queue->size();
    } else {
      return ba->count();
    }
  }[[nodiscard]] bool non_empty() const {
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
    printf("queue = %p, queue_array = %p, ba = %p\n", queue, queue_array, ba);
    if (get_n() > 100) {
      return;
    }
    if (all) {
      return;
    }
    if (is_sparse) {
      const size_t start = queue->shared_out_start;
      const size_t end = queue->shared_out_end;
      printf("{");
      for (size_t i = start; i < end; i++) {
        printf("%d, ", queue->shared[i]);
      }
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
      queue_array[4 * getWorkerNum()].push_back(i);
      return;
    }
    return ba->set(i);
  }
  void insert_dense(node_t i) const { return ba->set(i); }
  void insert_sparse(node_t i) const {
    queue_array[4 * getWorkerNum()].push_back(i);
  }
  template <class F> void map_sparse(F &f) const {
    // printf("queue in map = %p\n", queue);
    const size_t start = queue->shared_out_start;
    const size_t end = queue->shared_out_end;
    parallel_for(size_t i = start; i < end; i++) { f.update(queue->shared[i]); }
  }
  template <class F> void map(F &f) {
    if (all) {
      parallel_for(node_t i = 0; i < max_el; i++) { f.update(i); }
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
    queue = new SlidingQueue<node_t>(max_el);
    queue->push_back(e);
    queue->slide_window();
  }
  VertexSubset(bool const *const els, node_t len)
      : all(false), is_sparse(false), max_el(len) {
    ba = new BitArray(max_el);
    parallel_for_256(node_t i = 0; i < max_el; i++) {
      if (els[i]) {
        ba->set(i);
      }
    }
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
    queue_array = nullptr;
    return *this;
  }

  // can't add anything to these once they have been copied,
  // just for keeping state like pushing past frontiers into a vector
  VertexSubset(const VertexSubset &other, bool copy_data)
      : all(other.all), is_sparse(other.is_sparse), max_el(other.max_el) {
    if (copy_data) {
      if (all) {
        return;
      }
      ba = nullptr;
      if (is_sparse) {
        if (other.queue) {
          queue = new SlidingQueue<node_t>(*other.queue, max_el);
        }
        if (other.queue_array) {
          queue_array = (QueueBuffer<node_t> *)malloc(
              4 * sizeof(QueueBuffer<node_t>) * getWorkers());
          for (int i = 0; i < getWorkers(); i++) {
            new (&queue_array[i * 4]) QueueBuffer<node_t>(
                *queue, other.queue_array[i * 4].local_size);
            queue_array[i * 4].in = other.queue_array[i * 4].in;
            memcpy(queue_array[i * 4].local_queue,
                   other.queue_array[i * 4].local_queue,
                   queue_array[i * 4].in * sizeof(node_t));
          }
        }
      } else {
        if (other.ba) {
          ba = new BitArray(*other.ba);
        }
      }
    } else { // just create something similar where we will push the next set of
      // data into
      // sparse and dense stay they way they are, will be changed by something
      // else all turns to dense, if we knew it was going to stay as all we
      // would have no output and not use a new vertexsubset anyway
      all = false;
      if (is_sparse) {
        queue = new SlidingQueue<node_t>(max_el);
        queue_array = (QueueBuffer<node_t> *)malloc(
            4 * sizeof(QueueBuffer<node_t>) * getWorkers());
        for (int i = 0; i < getWorkers(); i++) {
          new (&queue_array[i * 4]) QueueBuffer<node_t>(*queue);
        }
      } else {
        ba = new BitArray(max_el);
      }
    }
  }
  void finalize() {
    if (is_sparse) {
      parallel_for_1(int i = 0; i < getWorkers(); i++) {
        queue_array[i * 4].flush();
        queue_array[i * 4].~QueueBuffer();
      }
      queue->slide_window();
      free(queue_array);
      queue_array = nullptr;
    }
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
    vs.queue_array = nullptr;
    vs.is_sparse = false;
    vs.ba = new BitArray(max_el);
    parallel_for(size_t i = queue->shared_out_start; i < queue->shared_out_end;
                 i++) {
      vs.ba->set_atomic(queue->shared[i]);
    }
    return vs;
  }[[nodiscard]] VertexSubset convert_to_sparse() const {
    VertexSubset vs;
    if (all || is_sparse) {
      return vs;
    }
    vs.all = all;
    vs.max_el = max_el;
    vs.is_sparse = true;
    vs.queue = new SlidingQueue<node_t>(max_el);
    vs.queue_array = (QueueBuffer<node_t> *)malloc(
        4 * sizeof(QueueBuffer<node_t>) * getWorkers());
    vs.ba = nullptr;
    for (int i = 0; i < getWorkers(); i++) {
      new (&vs.queue_array[i * 4]) QueueBuffer<node_t>(*vs.queue);
    }
    parallel_for(node_t i = 0; i < max_el; i++) {
      if (ba->get(i)) {
        vs.queue_array[4U * getWorkerNum()].push_back(i);
      }
    }
    parallel_for(int i = 0; i < getWorkers(); i++) {
      vs.queue_array[i * 4].flush();
    }
    for (int i = 0; i < getWorkers(); i++) {
      vs.queue_array[i * 4].~QueueBuffer();
    }
    vs.queue->slide_window();
    free(vs.queue_array);
    vs.queue_array = nullptr;
    return vs;
  }
};
