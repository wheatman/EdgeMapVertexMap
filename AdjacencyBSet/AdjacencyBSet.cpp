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
#include <type_traits>
#include <vector>

#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

template <class node_t, class weight_t = bool> class AdjacencyBSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<typename std::conditional<
      binary, absl::btree_set<node_t>, absl::btree_map<node_t, weight_t>>::type>
      nodes;

public:
  AdjacencyBSet(node_t n) : nodes(n) {}

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
    // can't parallel iterate a btree_set
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
    auto it_A = nodes[a].begin();
    auto it_B = nodes[b].begin();
    auto end_A = nodes[a].end();
    auto end_B = nodes[b].end();
    uint64_t ans = 0;
    if constexpr (binary) {
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
    } else {
      while (it_A != end_A && it_B != end_B && it_A->first < a &&
             it_B->second < b) { // count "directed" triangles
        if (it_A->first == it_B->first) {
          ++it_A, ++it_B, ans++;
        } else if (it_A->first < it_B->first) {
          ++it_A;
        } else {
          ++it_B;
        }
      }
    }
    return ans;
  }

  void write_adj_file(const std::string &filename) {
    std::ofstream myfile;
    myfile.open(filename);
    if constexpr (binary) {
      myfile << "AdjacencyGraph\n";
    } else {
      myfile << "WeightedAdjacencyGraph\n";
    }
    myfile << nodes.size() << "\n";
    ParallelTools::Reducer_sum<uint64_t> edge_count_reducer;
    ParallelTools::parallel_for(0, nodes.size(), [&](size_t i) {
      edge_count_reducer += nodes[i].size();
    });
    myfile << edge_count_reducer.get() << "\n";
    uint64_t running_edge_total = 0;
    for (uint64_t i = 0; i < nodes.size(); i++) {
      myfile << running_edge_total << "\n";
      running_edge_total += nodes[i].size();
    }
    for (uint64_t i = 0; i < nodes.size(); i++) {
      if constexpr (binary) {
        for (const auto &dest : nodes[i]) {
          myfile << dest << "\n";
        }
      } else {
        for (const auto &[dest, weight] : nodes[i]) {
          myfile << dest << "\n";
        }
      }
    }
    if constexpr (!binary) {
      for (uint64_t i = 0; i < nodes.size(); i++) {
        for (const auto &[dest, weight] : nodes[i]) {
          myfile << weight << "\n";
        }
      }
    }
    myfile.close();
  }
};

template <class node_t, class weight_t = bool> class DirectedAdjacencyBSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<typename std::conditional<
      binary, absl::btree_set<node_t>, absl::btree_map<node_t, weight_t>>::type>
      out_nodes;
  std::vector<typename std::conditional<
      binary, absl::btree_set<node_t>, absl::btree_map<node_t, weight_t>>::type>
      in_nodes;

public:
  DirectedAdjacencyBSet(node_t n) : out_nodes(n), in_nodes(n) {}

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
  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("src","what node to start from",cxxopts::value<uint64_t>()->default_value("0"))
    ("priters","how many iters for pr",cxxopts::value<uint64_t>()->default_value("10"))
    ("g,graph", "graph file path", cxxopts::value<std::string>())
    ("algorithm", "which algorithm to run", cxxopts::value<std::string>())
    ("w,weights", "run with a weighted graph", cxxopts::value<bool>()->default_value("false")) 
    ("s,symetric", "symeterizes the graph while reading it in and then run on an undirected graph", cxxopts::value<bool>()->default_value("true")) 
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t pr_iters = result["priters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  bool use_weights = result["weights"].as<bool>();
  bool symetric = result["symetric"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;
  if (symetric) {
    if (!use_weights) {

      std::vector<std::tuple<uint32_t, uint32_t>> edges;
      if (graph_filename == "skew") {
        node_count = 10000000;
        edges = very_skewed_graph<uint32_t>(node_count, 100, node_count / 2);
      } else {
        edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                        &node_count, symetric);
      }
      AdjacencyBSet<uint64_t> g = AdjacencyBSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<true>(g, algorithm_to_run, src, pr_iters);

    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, true);
      auto g = AdjacencyBSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src);
    }
  } else {
    if (!use_weights) {
      auto edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                           &node_count, symetric);

      auto g = DirectedAdjacencyBSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest] : edges) {
        g.add_edge(src, dest);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<false>(g, algorithm_to_run, src, pr_iters);

    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, symetric);
      auto g = DirectedAdjacencyBSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest, val] : edges) {
        g.add_edge(src, dest, val);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src);
    }
  }
}
