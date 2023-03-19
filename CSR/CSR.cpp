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

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

template <class node_t, class edge_t> class CSR {
  // data members
  uint64_t n; // num vertices
  uint64_t m; // num edges
  // nodes is pointer into edges, for where those edges for that node starts
  edge_t *nodes;
  // edges is which node that edge points at
  node_t *edges;

public:
  // function headings

  CSR(node_t n, std::vector<std::tuple<node_t, node_t>> &edges_list) : n(n) {
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
  void print() {
    printf("AdjacencyGraph\n%lu\n%lu\n", n, m);
    for (uint64_t i = 0; i < n; i++) {
      printf("%u\n", nodes[i]);
    }
    for (uint64_t i = 0; i < m; i++) {
      printf("%u\n", edges[i]);
    }
  }

  ~CSR() {
    free(nodes);
    free(edges);
  }

  size_t num_nodes() const { return n; }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    edge_t start = nodes[node];
    edge_t end = nodes[node + 1];

    if (parallel) {
      ParallelTools::parallel_for(start, end,
                                  [&](edge_t i) { f(node, edges[i]); });
    } else {
      for (edge_t i = start; i < end; i++) {
        f(node, edges[i]);
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
    ("nClusters", "number of clusters for algorithms that need it, currently only gee", cxxopts::value<uint64_t>()->default_value("0")) 
    ("y_location", "path to the y vector of GEE", cxxopts::value<std::string>()->default_value("")) 
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t pr_iters = result["priters"].as<uint64_t>();
  uint64_t nClusters = result["nClusters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  std::string y_location = result["y_location"].as<std::string>();
  uint64_t edge_count;
  uint32_t node_count;

  auto edges =
      get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);

  CSR<uint32_t, uint32_t> g = CSR<uint32_t, uint32_t>(node_count, edges);
  run_unweighted_algorithms<false>(g, algorithm_to_run, src, pr_iters,
                                   nClusters, y_location);
}
