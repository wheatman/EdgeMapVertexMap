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

#include "EdgeMapVertexMap/internal/BitArray.hpp"
#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

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

public:
  DenseToSparseSetWithFlip() = default;
  DenseToSparseSetWithFlip(T max_elements)
      : max_elements(max_elements), num_elements(0),
        count_to_flip(
            std::max(4UL, ((uint64_t)(max_elements * ratio_to_flip)))) {}

  void insert(T element) {
    if (num_elements < count_to_flip) {
      size_t i = 0;
      for (; i < sparse_set.size(); i++) {
        if (sparse_set[i] == element) {
          return;
        }
        if (sparse_set[i] > element) {
          break;
        }
      }
      num_elements += 1;
      sparse_set.insert(sparse_set.begin() + i, element);

      if (num_elements == count_to_flip) {
        convert_to_dense();
      }
      return;
    } else {
      if (!dense_set.get(element)) {
        num_elements += 1;
        dense_set.set(element);
      }
      return;
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

  static uint64_t intersection_count(const DenseToSparseSetWithFlip &A,
                                     const DenseToSparseSetWithFlip &B, T a,
                                     T b) {
    if (A.num_elements < A.count_to_flip && B.num_elements < B.count_to_flip) {
      auto it_A = A.sparse_set.begin();
      auto it_B = B.sparse_set.begin();
      auto end_A = A.sparse_set.end();
      auto end_B = B.sparse_set.end();
      uint64_t ans = 0;
      while (it_A != end_A && it_B != end_B && *it_A < a &&
             *it_B < b) { // count "directed" triangles
        if (*it_A == *it_B) {
          ++it_A, ++it_B, ans++;
        } else if (*it_A < *it_B) {
          ++it_A;
        } else {
          ++it_B;
        }
      }
      return ans;
    }
    T limit = std::min(a, b);
    if (A.num_elements < A.count_to_flip) {
      uint64_t ans = 0;
      for (const auto &el : A.sparse_set) {
        if (el >= limit) {
          break;
        }
        ans += B.dense_set.get(el);
      }
      return ans;
    }
    if (B.num_elements < B.count_to_flip) {
      uint64_t ans = 0;
      for (const auto &el : B.sparse_set) {
        if (el >= limit) {
          break;
        }
        ans += A.dense_set.get(el);
      }
      return ans;
    }
    return A.dense_set.intersection_count(B.dense_set, limit);
  }
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

  uint64_t common_neighbors(node_t a, node_t b) const {
    return DenseToSparseSetWithFlip<node_t>::intersection_count(nodes[a],
                                                                nodes[b], a, b);
  }
};

int main(int32_t argc, char *argv[]) {
  cxxopts::Options options("Graph tester",
                           "Runs different algorithms on a Graph");
  add_options_to_parser(options);
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t iters = result["iters"].as<uint64_t>();
  uint64_t pr_rounds = result["pr_rounds"].as<uint64_t>();
  uint64_t nClusters = result["nClusters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  bool dump_output = result["dump_output"].as<bool>();
  std::string y_location = result["y_location"].as<std::string>();

  uint64_t edge_count;
  uint32_t node_count;
  std::vector<std::tuple<uint32_t, uint32_t>> edges;
  if (graph_filename == "skew") {
    node_count = 10000000;
    edges = very_skewed_graph<uint32_t>(node_count, 100, node_count / 2);
  } else {
    edges =
        get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
  }
  AdjacencyDenseSparseSet<uint32_t> g =
      AdjacencyDenseSparseSet<uint32_t>(node_count);
  uint64_t start = get_usecs();
  parallel_batch_insert(g, edges);
  uint64_t end = get_usecs();
  printf("loading the graph took %lu\n", end - start);
  run_unweighted_algorithms<true>(g, algorithm_to_run, src, iters, pr_rounds,
                                  nClusters, y_location, dump_output);
}
