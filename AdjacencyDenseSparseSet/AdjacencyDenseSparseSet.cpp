/*
 * adjacency matrix
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <set>
#include <vector>

#include "../io_util.hpp"

#include "../algorithms/BC.h"
#include "../algorithms/BFS.h"
#include "../algorithms/Components.h"
#include "../algorithms/PageRank.h"

#include "../BitArray.hpp"
#include "../GraphHelpers.hpp"

using namespace EdgeMapVertexMap;

template <class T> class DenseToSparseSetWithFlip {
  std::vector<T> sparse_set;
  BitArray dense_set;
  uint64_t max_elements;
  uint64_t num_elements;
  static constexpr double ratio_to_flip = .001;
  uint64_t count_to_flip;

  void convert_to_dense() {
    dense_set.resize_and_clear(max_elements);
    for (const auto &el : sparse_set) {
      dense_set.set(el);
    }
    sparse_set.clear();
  }

  void compact() {
    if (num_elements >= count_to_flip) {
      dense_set.map([&](size_t i) { sparse_set.push_back(i); });
      dense_set.resize_and_clear(0);
    }
  }

public:
  DenseToSparseSetWithFlip() = default;
  DenseToSparseSetWithFlip(T max_elements)
      : max_elements(max_elements), num_elements(0),
        count_to_flip(
            std::max(4UL, ((uint64_t)(max_elements * ratio_to_flip)))) {}

  std::pair<T, bool> insert(T element) {
    if (num_elements < count_to_flip) {
      size_t i = 0;
      for (; i < sparse_set.size(); i++) {
        if (sparse_set[i] == element) {
          return {i, false};
        }
        if (sparse_set[i] > element) {
          break;
        }
      }
      num_elements += 1;
      if (i < sparse_set.size()) {
        sparse_set.insert(sparse_set.begin() + i, element);
      } else {
        sparse_set.push_back(element);
      }
      if (num_elements == count_to_flip) {
        convert_to_dense();
        return {element, true};
      }
      return {i, true};
    } else {
      if (!dense_set.get(element)) {
        num_elements += 1;
        dense_set.set(element);
        return {element, true};
      } else {
        return {element, false};
      }
    }
  }

  void erase(size_t index_to_remove) {
    if (num_elements >= count_to_flip) {
      dense_set.flip(index_to_remove);
      num_elements -= 1;
      if (num_elements < count_to_flip) {
        compact();
      }
    } else {
      sparse_set.erase(sparse_set.begin() + index_to_remove);
      num_elements -= 1;
    }
  }

  template <class F> void map(F f, bool parallel) const {
    if (num_elements < count_to_flip) {
      for (const auto &el : sparse_set) {
        f(el);
      }
    } else {
      dense_set.map(f, parallel);
    }
  }
  size_t size() { return sparse_set.size(); }
};

template <class node_t> class AdjacencyDenseSparseSet {

  std::vector<DenseToSparseSetWithFlip<node_t>> nodes;

public:
  AdjacencyDenseSparseSet(node_t n) : nodes(n, n) {}

  size_t num_nodes() const { return nodes.size(); }

  void add_edge(node_t source, node_t dest) { nodes[source].insert(dest); }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     [[maybe_unused]] bool parallel) const {
    nodes[node].map([&](uint64_t dest) { f(node, dest); }, parallel);
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
      get_edges_from_file_adj_sym(graph_filename, &edge_count, &node_count);
  AdjacencyDenseSparseSet<uint32_t> g =
      AdjacencyDenseSparseSet<uint32_t>(node_count);
  for (auto edge : edges) {
    g.add_edge(edge.first, edge.second);
  }
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
