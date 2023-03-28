/*
 * CSR
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

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

template <class node_t, class edge_t, class weight_t = bool> class CSR {
  // data members
  uint64_t n; // num vertices
  uint64_t m; // num edges
  // nodes is pointer into edges, for where those edges for that node starts
  edge_t *nodes;
  // edges is which node that edge points at
  node_t *edges;
  weight_t *weights;
  static constexpr bool binary = std::is_same_v<weight_t, bool>;

public:
  // function headings

  CSR(node_t n, std::vector<std::tuple<node_t, node_t>> &edges_list) : n(n) {
    static_assert(binary);
    ParallelTools::sort(edges_list.begin(), edges_list.end());
    auto new_end = std::unique(edges_list.begin(), edges_list.end());
    edges_list.resize(std::distance(edges_list.begin(), new_end));
    m = edges_list.size();
    nodes = (edge_t *)malloc((n + 1) * sizeof(edge_t));
    edges = (node_t *)malloc((m) * sizeof(node_t));
    ParallelTools::parallel_for(
        0, m, [&](edge_t i) { edges[i] = std::get<1>(edges_list[i]); });
    nodes[0] = 0;
    node_t current_node = 0;
    edge_t current_position = 0;
    while (current_node < n && current_position < m) {
      auto edge = edges_list[current_position];
      if (std::get<0>(edge) > current_node) {
        for (node_t i = current_node + 1; i <= std::get<0>(edge); i++) {
          nodes[i] = current_position;
        }
        current_node = std::get<0>(edge);
      }
      current_position++;
    }
    for (node_t i = current_node + 1; i <= n; i++) {
      nodes[i] = m;
    }
  }
  CSR(node_t n, std::vector<std::tuple<node_t, node_t, weight_t>> &edges_list)
      : n(n) {
    static_assert(!binary);
    // just take the first weight of each edge if their are duplicates
    ParallelTools::sort(edges_list.begin(), edges_list.end());
    auto new_end =
        std::unique(edges_list.begin(), edges_list.end(),
                    [](const auto &tup1, const auto &tup2) {
                      return std::get<0>(tup1) == std::get<0>(tup2) &&
                             std::get<1>(tup1) == std::get<1>(tup2);
                    });
    edges_list.resize(std::distance(edges_list.begin(), new_end));
    m = edges_list.size();
    nodes = (edge_t *)malloc((n + 1) * sizeof(edge_t));
    edges = (node_t *)malloc((m) * sizeof(node_t));
    weights = (weight_t *)malloc((m) * sizeof(node_t));
    ParallelTools::parallel_for(0, m, [&](edge_t i) {
      edges[i] = std::get<1>(edges_list[i]);
      weights[i] = std::get<2>(edges_list[i]);
      assert(weights[i] != 0);
    });
    nodes[0] = 0;
    node_t current_node = 0;
    edge_t current_position = 0;
    while (current_node < n && current_position < m) {
      auto edge = edges_list[current_position];
      if (std::get<0>(edge) > current_node) {
        for (node_t i = current_node + 1; i <= std::get<0>(edge); i++) {
          nodes[i] = current_position;
        }
        current_node = std::get<0>(edge);
      }
      current_position++;
    }
    for (node_t i = current_node + 1; i <= n; i++) {
      nodes[i] = m;
    }
  }

  ~CSR() {
    free(nodes);
    free(edges);
    if constexpr (!binary) {
      free(weights);
    }
  }

  size_t num_nodes() const { return n; }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    edge_t start = nodes[node];
    edge_t end = nodes[node + 1];
    if constexpr (binary) {
      if (parallel) {
        ParallelTools::parallel_for(start, end,
                                    [&](edge_t i) { f(node, edges[i]); });
      } else {
        for (edge_t i = start; i < end; i++) {
          f(node, edges[i]);
        }
      }
    } else {
      if (parallel) {
        ParallelTools::parallel_for(
            start, end, [&](edge_t i) { f(node, edges[i], weights[i]); });
      } else {
        for (edge_t i = start; i < end; i++) {
          f(node, edges[i], weights[i]);
        }
      }
    }
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
  bool laplacian = result["laplacian"].as<bool>();
  bool dump_output = result["dump_output"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;

  if (!use_weights) {
    auto edges =
        get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
    CSR<uint32_t, uint32_t> g(node_count, edges);
    run_unweighted_algorithms<false>(g, algorithm_to_run, src, iters, pr_rounds,
                                     nClusters, y_location, dump_output);
  } else {
    using weight_type = uint32_t;
    auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
        graph_filename, &edge_count, &node_count, true);
    CSR<uint32_t, uint32_t, weight_type> g(node_count, edges);
    run_weighted_algorithms(g, algorithm_to_run, src, iters, nClusters,
                            y_location, laplacian, dump_output);
  }
}
