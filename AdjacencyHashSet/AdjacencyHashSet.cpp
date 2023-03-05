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
#include <unordered_set>
#include <vector>

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

using namespace EdgeMapVertexMap;

template <class node_t> class AdjacencyHashMap {

  std::vector<std::unordered_set<node_t>> nodes;

public:
  // function headings

  AdjacencyHashMap(node_t n) : nodes(n) {}

  size_t num_nodes() const { return nodes.size(); }

  void add_edge(node_t source, node_t dest) { nodes[source].insert(dest); }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     [[maybe_unused]] bool parallel) const {
    // can't parallel iterate a hash_set

    for (const auto &dest : nodes[node]) {
      f(node, dest);
    }
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
  AdjacencyHashMap<uint32_t> g = AdjacencyHashMap<uint32_t>(node_count);
  parallel_batch_insert(g, edges);
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
