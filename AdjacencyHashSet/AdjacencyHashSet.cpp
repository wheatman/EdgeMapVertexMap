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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

template <class node_t, class weight_t = bool> class AdjacencyHashSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<
      typename std::conditional<binary, std::unordered_set<node_t>,
                                std::unordered_map<node_t, weight_t>>::type>
      nodes;

public:
  // function headings

  AdjacencyHashSet(node_t n) : nodes(n) {}

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
};

template <class node_t, class weight_t = bool> class DirectedAdjacencyHashSet {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

  std::vector<
      typename std::conditional<binary, std::unordered_set<node_t>,
                                std::unordered_map<node_t, weight_t>>::type>
      out_nodes;

  std::vector<
      typename std::conditional<binary, std::unordered_set<node_t>,
                                std::unordered_map<node_t, weight_t>>::type>
      in_nodes;

public:
  // function headings

  DirectedAdjacencyHashSet(node_t n) : out_nodes(n), in_nodes(n) {}

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
    ("nClusters", "number of clusters for algorithms that need it, currently only gee", cxxopts::value<uint64_t>()->default_value("0")) 
    ("y_location", "path to the y vector of GEE", cxxopts::value<std::string>()->default_value("")) 
    ("laplacian", "use the laplacian in weighted GEE", cxxopts::value<bool>()->default_value("false")) 
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t pr_iters = result["priters"].as<uint64_t>();
  uint64_t nClusters = result["nClusters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  std::string y_location = result["y_location"].as<std::string>();
  bool use_weights = result["weights"].as<bool>();
  bool symetric = result["symetric"].as<bool>();
  bool laplacian = result["laplacian"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;
  if (symetric) {
    if (!use_weights) {
      auto edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                           &node_count, symetric);
      auto g = AdjacencyHashSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<false>(g, algorithm_to_run, src, pr_iters,
                                       nClusters, y_location);

    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, symetric);
      auto g = AdjacencyHashSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      parallel_batch_insert(g, edges);
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src, nClusters, y_location,
                              laplacian);
    }
  } else {
    if (!use_weights) {
      auto edges = get_edges_from_file_adj(graph_filename, &edge_count,
                                           &node_count, symetric);
      auto g = DirectedAdjacencyHashSet<uint64_t>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest] : edges) {
        g.add_edge(src, dest);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_unweighted_algorithms<false>(g, algorithm_to_run, src, pr_iters,
                                       nClusters, y_location);
    } else {
      using weight_type = uint32_t;
      auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
          graph_filename, &edge_count, &node_count, symetric);
      auto g = DirectedAdjacencyHashSet<uint64_t, weight_type>(node_count);
      uint64_t start = get_usecs();
      for (const auto &[src, dest, val] : edges) {
        g.add_edge(src, dest, val);
      }
      uint64_t end = get_usecs();
      printf("loading the graph took %lu\n", end - start);
      run_weighted_algorithms(g, algorithm_to_run, src, nClusters, y_location,
                              laplacian);
    }
  }
}
