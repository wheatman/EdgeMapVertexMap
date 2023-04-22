/*
 * adjacency matrix
 */

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "EdgeMapVertexMap/internal/BitArray.hpp"
#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

class BinaryAdjacencyMatrix {
public:
  using node_t = uint64_t;
  using weight_t = bool;
  using extra_data_t = void *;

private:
  // data members
  uint64_t n; // num vertices
  BitArray array;

public:
  // function headings
  BinaryAdjacencyMatrix(uint64_t init_n) : n(init_n), array(n * n) {}

  bool has_edge(node_t src, node_t dest) const {
    return array.get(src * n + dest);
  }
  void add_edge(node_t src, node_t dest) const { array.set(src * n + dest); }
  size_t num_nodes() const { return n; }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    if (parallel) {
      ParallelTools::parallel_for(0, n, [&](node_t i) {
        if (array.get(node * n + i)) {
          f(node, i);
        }
      });
    } else {
      for (node_t i = 0; i < n; i++) {
        if (array.get(node * n + i)) {
          f(node, i);
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
  bool dump_output = result["dump_output"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;

  auto edges =
      get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
  BinaryAdjacencyMatrix g(node_count);
  for (const auto &[src, dest] : edges) {
    g.add_edge(src, dest);
  }
  run_unweighted_algorithms<false>(g, algorithm_to_run, src, iters, pr_rounds,
                                   nClusters, y_location, dump_output);
}
