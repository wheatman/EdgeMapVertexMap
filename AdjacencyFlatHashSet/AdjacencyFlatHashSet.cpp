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
#include <vector>

#include "cxxopts.hpp"

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

using namespace EdgeMapVertexMap;

template <class node_t, class weight_t = bool> class AdjacencyFlatHashSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<
      typename std::conditional<binary, absl::flat_hash_set<node_t>,
                                absl::flat_hash_map<node_t, weight_t>>::type>
      nodes;

public:
  AdjacencyFlatHashSet(node_t n) : nodes(n) {}

  size_t num_nodes() const { return nodes.size(); }

  void add_edge(node_t source, node_t dest) {
    static_assert(binary);
    nodes[source].insert(dest);
  }

  void add_edge(node_t source, node_t dest, weight_t weight) {
    static_assert(!binary);
    nodes[source][dest] = weight;
  }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     [[maybe_unused]] bool parallel) const {
    // can't parallel iterate a hash_set
    if constexpr (binary) {
      for (const auto &dest : nodes[node]) {
        f(node, dest);
      }
    } else {
      for (const auto &[dest, weight] : nodes[node]) {
        f(node, dest, weight);
      }
    }
  }

  uint64_t common_neighbors(node_t a, node_t b) const {
    uint64_t ans = 0;
    node_t smaller = (nodes[a].size() <= nodes[b].size()) ? a : b;
    node_t larger = (nodes[a].size() <= nodes[b].size()) ? b : a;
    if constexpr (binary) {
      for (const auto &element : nodes[smaller]) {
        if (element < smaller && nodes[larger].contains(element) &&
            element < larger) {
          ans += 1;
        }
      }
    } else {
      for (const auto &[element, weight] : nodes[smaller]) {
        if (element < smaller && nodes[larger].contains(element) &&
            element < larger) {
          ans += 1;
        }
      }
    }
    return ans;
  }
};

template <class node_t, class weight_t = bool>
class DirectedAdjacencyFlatHashSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<
      typename std::conditional<binary, absl::flat_hash_set<node_t>,
                                absl::flat_hash_map<node_t, weight_t>>::type>
      out_nodes;
  std::vector<
      typename std::conditional<binary, absl::flat_hash_set<node_t>,
                                absl::flat_hash_map<node_t, weight_t>>::type>
      in_nodes;

public:
  DirectedAdjacencyFlatHashSet(node_t n) : out_nodes(n), in_nodes(n) {}

  size_t num_nodes() const {
    assert(in_nodes.size() == out_nodes.size());
    return out_nodes.size();
  }

  void add_edge(node_t source, node_t dest) {
    static_assert(binary);
    out_nodes[source].insert(dest);
    in_nodes[dest].insert(source);
  }

  void add_edge(node_t source, node_t dest, weight_t weight) {
    static_assert(!binary);
    out_nodes[source][dest] = weight;
    in_nodes[dest][source] = weight;
  }

  template <class F>
  void map_out_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                         [[maybe_unused]] bool parallel) const {
    // can't parallel iterate a btree_set
    if constexpr (binary) {
      for (const auto &dest : out_nodes[node]) {
        f(node, dest);
      }
    } else {
      for (const auto &[dest, weight] : out_nodes[node]) {
        f(node, dest, weight);
      }
    }
  }
  template <class F>
  void map_in_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                        [[maybe_unused]] bool parallel) const {
    // can't parallel iterate a btree_set
    if constexpr (binary) {
      for (const auto &src : in_nodes[node]) {
        f(src, node);
      }
    } else {
      for (const auto &[src, weight] : in_nodes[node]) {
        f(src, node, weight);
      }
    }
  }

  void write_adj_file(const std::string &filename) {
    std::ofstream myfile;
    myfile.open(filename);
    if constexpr (binary) {
      myfile << "AdjacencyGraph\n";
    } else {
      myfile << "WeightedAdjacencyGraph\n";
    }
    assert(in_nodes.size() == out_nodes.size());
    myfile << out_nodes.size() << "\n";
    ParallelTools::Reducer_sum<uint64_t> edge_count_reducer;
    ParallelTools::parallel_for(0, out_nodes.size(), [&](size_t i) {
      edge_count_reducer += out_nodes[i].size();
    });
    myfile << edge_count_reducer.get() << "\n";
    uint64_t running_edge_total = 0;
    for (uint64_t i = 0; i < out_nodes.size(); i++) {
      myfile << running_edge_total << "\n";
      running_edge_total += out_nodes[i].size();
    }
    for (uint64_t i = 0; i < out_nodes.size(); i++) {
      if constexpr (binary) {
        for (const auto &dest : out_nodes[i]) {
          myfile << dest << "\n";
        }
      } else {
        for (const auto &[dest, weight] : out_nodes[i]) {
          myfile << dest << "\n";
        }
      }
    }
    if constexpr (!binary) {
      for (uint64_t i = 0; i < out_nodes.size(); i++) {
        for (const auto &[dest, weight] : out_nodes[i]) {
          myfile << weight << "\n";
        }
      }
    }
    myfile.close();
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
  std::string y_location = result["y_location"].as<std::string>();
  bool use_weights = result["weights"].as<bool>();
  bool symetric = result["symetric"].as<bool>();
  bool laplacian = result["laplacian"].as<bool>();
  bool dump_output = result["dump_output"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;
  if (symetric) {
    if (!use_weights) {
      auto edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                           &node_count, symetric);

      auto g = AdjacencyFlatHashSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<true>(g, algorithm_to_run, src, iters,
                                      pr_rounds, nClusters, y_location,
                                      dump_output);
    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, symetric);
      auto g = AdjacencyFlatHashSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src, iters, nClusters,
                              y_location, laplacian, dump_output);
    }
  } else {
    if (!use_weights) {
      auto edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                           &node_count, symetric);

      auto g = DirectedAdjacencyFlatHashSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest] : edges) {
        g.add_edge(src, dest);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<false>(g, algorithm_to_run, src, iters,
                                       pr_rounds, nClusters, y_location,
                                       dump_output);
    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, symetric);
      auto g = DirectedAdjacencyFlatHashSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest, val] : edges) {
        g.add_edge(src, dest, val);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src, iters, nClusters,
                              y_location, laplacian, dump_output);
    }
  }
}
