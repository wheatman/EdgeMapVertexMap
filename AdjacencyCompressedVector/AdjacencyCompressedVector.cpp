/*
 * adjacency matrix
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iterator>
#include <limits>
#include <vector>

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

using namespace EdgeMapVertexMap;

template <class T> class compressedvector {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T can only be 4 or 8 bytes");
  static constexpr size_t max_element_size = sizeof(T) + (sizeof(T) / 4);

  template <typename U> static inline T unaligned_load(const void *loc) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8,
                  "Size of U must be either 2, 4, or 8");
    U data;
    std::memcpy(&data, loc, sizeof(U));
    return data;
  }
  template <typename U> static inline void unaligned_store(void *loc, U value) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8,
                  "Size of U must be either 2, 4, or 8");
    std::memcpy(loc, &value, sizeof(U));
  }
  class FindResult {
  public:
    T difference;
    int64_t loc;
    int64_t size;
  };
  class EncodeResult {
  public:
    static constexpr int storage_size = std::max(max_element_size + 1, 8UL);
    uint8_t data[storage_size] = {0};
    int64_t size;

    static int64_t write_encoded(T difference, uint8_t *loc) {
      loc[0] = difference & 0x7FU;
      int64_t num_bytes = 1;
      difference >>= 7;
      while (difference) {
        loc[num_bytes - 1] |= 0x80U;
        loc[num_bytes] = difference & 0x7FU;
        num_bytes += 1;
        difference >>= 7;
      }
      return num_bytes;
    }
    EncodeResult(T difference) {
      assert(difference != 0);
      size = write_encoded(difference, data);
    }
  };
  class DecodeResult {
  public:
    T difference = 0;
    int64_t old_size = 0;
    void print() {
      std::cout << "DecodeResult { difference=" << difference
                << ", old_size=" << old_size << " }" << std::endl;
    }
    static constexpr std::array<uint64_t, 8> extract_masks = {
        0x000000000000007FUL, 0x0000000000007F7FUL, 0x00000000007F7F7FUL,
        0x000000007F7F7F7FUL, 0x0000007F7F7F7F7FUL, 0x00007F7F7F7F7F7FUL,
        0x007F7F7F7F7F7F7FUL, 0x7F7F7F7F7F7F7F7FUL};

    static constexpr std::array<uint64_t, 16> masks_for_4 = {
        0x7FUL, 0x7F7FUL,     0x7FUL, 0x7F7F7FUL,    0x7FUL, 0x7F7FUL,
        0x7FUL, 0x7F7F7F7FUL, 0x7FUL, 0x7F7FUL,      0x7FUL, 0x7F7F7FUL,
        0x7FUL, 0x7F7FUL,     0x7FUL, 0x7F7F7F7F7FUL};
    static constexpr std::array<int8_t, 16> index_for_4 = {
        1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5};

    DecodeResult() = default;

    DecodeResult(T d, int64_t s) : difference(d), old_size(s) {}

    DecodeResult(const uint8_t *loc) {

      if (*loc == 0) {
        difference = 0;
        old_size = 0;
        return;
      }
#if __BMI2__ == 1
      uint64_t chunks = unaligned_load<uint64_t>(loc);
      if ((chunks & 0x80UL) == 0) {
        difference = *loc;
        old_size = 1;
        return;
      }

      uint64_t mask = _pext_u64(chunks, 0x8080808080808080UL);
      if (sizeof(T) == 4 ||
          (chunks & 0x8080808080808080UL) != 0x8080808080808080UL) {
        int32_t index = _mm_tzcnt_64(~mask);
        difference = _pext_u64(chunks, extract_masks[index]);
        old_size = index + 1;

        return;
      }

#endif
      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }
    DecodeResult(const uint8_t *loc, int64_t max_size) {
      if (*loc == 0) {
        difference = 0;
        old_size = 0;
        return;
      }
      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          if (old_size >= max_size) {
            break;
          }
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }
  };
  T head = std::numeric_limits<T>::max();
  std::vector<uint8_t> data;

  // returns the starting byte location, the length of the specified element,
  // and the difference from the previous element length is 0 if the element
  // is not found starting byte location of 0 means this is the head
  // if we have just changed the head we pass in the old head val so we can
  // correctly interpret the bytes after that
  FindResult find(T x) const {
    T curr_elem = head;
    T prev_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr;
    while (curr_elem < x) {
      dr = DecodeResult(data.data() + curr_loc);
      prev_elem = curr_elem;
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
    }
    if (x == curr_elem) {
      return {0, curr_loc - dr.old_size, dr.old_size};
    }
    // std::cout << "x = " << x << ", curr_elem = " << curr_elem << std::endl;
    return {x - prev_elem, curr_loc - dr.old_size, 0};
  }

public:
  // inserts an element
  // first return value indicates if something was inserted
  // if something was inserted the second value tells you the current size
  std::pair<T, int64_t> push_back(T x, T last, int64_t hint) {
    if (head == std::numeric_limits<T>::max()) {
      head = x;
      return {x, 0};
    }
    if (data.size() == 0) {
      // make the thing 8 longer than it needs to be so we can just read by
      // block and we can tell the end by the zeros
      data.resize(8);
    }

    if (x == last) {
      return {last, hint};
    }
    EncodeResult er(x - last);
    // we are inserting a new last element and don't need to slide
    data.resize(data.size() + er.size);
    memcpy(data.data() + hint, er.data, er.size);
    return {x, hint + er.size};
  }

  template <class F> void map(F f) const {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    if (data.size() == 0) {
      if (head != std::numeric_limits<T>::max()) {
        f(head);
      }
      return;
    }
    do {
      curr_elem += dr.difference;

      curr_loc += dr.old_size;

      f(curr_elem);

      dr = DecodeResult(data.data() + curr_loc);
    } while (dr.difference != 0);
  }
};

template <class node_t> class AdjacencyCompressedVector {
  // data members
  std::vector<compressedvector<node_t>> nodes;

public:
  // function headings

  AdjacencyCompressedVector(node_t n,
                            std::vector<std::tuple<node_t, node_t>> &edges_list)
      : nodes(n) {
    std::vector<node_t> shuffle_map(n);
    ParallelTools::parallel_for(0, n, [&](size_t i) { shuffle_map[i] = i; });

    std::random_device rd;
    std::shuffle(shuffle_map.begin(), shuffle_map.end(), rd);
    ParallelTools::sort(
        edges_list.begin(), edges_list.end(), [&shuffle_map](auto &a, auto &b) {
          return std::tie(shuffle_map[std::get<0>(a)], std::get<1>(a)) <
                 std::tie(shuffle_map[std::get<0>(b)], std::get<1>(b));
          ;
        });
    auto new_end = std::unique(edges_list.begin(), edges_list.end());
    edges_list.resize(std::distance(edges_list.begin(), new_end));
    uint64_t n_workers = ParallelTools::getWorkers();
    uint64_t p = std::min(std::max(1UL, edges_list.size() / 100), n_workers);
    std::vector<uint64_t> indxs(p + 1);
    indxs[0] = 0;
    indxs[p] = edges_list.size();
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * edges_list.size()) / p;
      node_t start_val = std::get<0>(edges_list[start]);
      while (std::get<0>(edges_list[start]) == start_val) {
        start += 1;
        if (start == edges_list.size()) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](size_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];
      int64_t hint = 0;
      node_t last = 0;

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        node_t x = std::get<0>(edges_list[idx]);
        node_t y = std::get<1>(edges_list[idx]);
        if (x == y) {
          continue;
        }
        std::tie(last, hint) = nodes[x].push_back(y, last, hint);
      }
    });
  }

  size_t num_nodes() const { return nodes.size(); }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     [[maybe_unused]] bool parallel) const {
    nodes[node].map([&](uint64_t dest) { f(node, dest); });
  }
};

int main(int32_t argc, char *argv[]) {
  if (argc < 3) {
    printf("call with graph filename, and which algorithm to run, and "
           "optionally the start node\n");
    return 0;
  }
  std::string graph_filename = std::string(argv[1]);

  uint64_t edge_count;
  uint32_t node_count;
  auto edges =
      get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
  AdjacencyCompressedVector<uint32_t> g =
      AdjacencyCompressedVector<uint32_t>(node_count, edges);
  std::string algorithm_to_run = std::string(argv[2]);
  if (algorithm_to_run == "bfs") {
    uint64_t source_node = std::strtol(argv[3], nullptr, 10);
    int32_t *bfs_out = BFS(g, source_node);
    std::vector<uint32_t> depths(node_count,
                                 std::numeric_limits<uint32_t>::max());
    ParallelTools::parallel_for(0, node_count, [&](uint32_t j) {
      uint32_t current_depth = 0;
      int32_t current_parent = j;
      if (bfs_out[j] < 0) {
        return;
      }
      while (current_parent != bfs_out[current_parent]) {
        current_depth += 1;
        current_parent = bfs_out[current_parent];
      }
      depths[j] = current_depth;
    });
    std::ofstream myfile;
    myfile.open("bfs.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << depths[i] << std::endl;
    }
    myfile.close();
    free(bfs_out);
  }
  if (algorithm_to_run == "bc") {
    uint64_t source_node = std::strtol(argv[3], nullptr, 10);
    double *bc_out = BC(g, source_node);
    std::ofstream myfile;
    myfile.open("bc.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << bc_out[i] << std::endl;
    }
    myfile.close();
    free(bc_out);
  }
  if (algorithm_to_run == "pr") {
    uint64_t iters = std::strtol(argv[3], nullptr, 10);
    double *pr_out = PR_S<double>(g, iters);
    std::ofstream myfile;
    myfile.open("pr.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << pr_out[i] << std::endl;
    }
    myfile.close();
    free(pr_out);
  }
  if (algorithm_to_run == "cc") {
    uint32_t *cc_out = CC(g);
    std::ofstream myfile;
    myfile.open("cc.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << cc_out[i] << std::endl;
    }
    myfile.close();
    free(cc_out);
  }

  if (algorithm_to_run == "all") {
    run_static_algorithms(g, std::strtol(argv[3], nullptr, 10));
  }
}
