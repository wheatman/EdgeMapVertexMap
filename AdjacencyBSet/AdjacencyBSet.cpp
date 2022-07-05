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

#include "../io_util.hpp"

#include "../algorithms/BC.h"
#include "../algorithms/BFS.h"
#include "../algorithms/Components.h"
#include "../algorithms/PageRank.h"
#include "../algorithms/TC.h"

#include "../GraphHelpers.hpp"

#include "absl/container/btree_set.h"

using namespace EdgeMapVertexMap;

template <class node_t> class AdjacencyBSet {

  std::vector<absl::btree_set<node_t>> nodes;

public:
  AdjacencyBSet(node_t n) : nodes(n) {}

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
  uint64_t common_neighbors(node_t a, node_t b) const {
    auto it_A = nodes[a].begin();
    auto it_B = nodes[b].begin();
    auto end_A = nodes[a].end();
    auto end_B = nodes[b].end();
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
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  if (graph_filename == "skew") {
    node_count = 10000000;
    edges = very_skewed_graph<uint32_t>(node_count, 100, node_count / 2);
  } else {
    edges =
        get_edges_from_file_adj_sym(graph_filename, &edge_count, &node_count);
  }
  AdjacencyBSet<uint64_t> g = AdjacencyBSet<uint64_t>(node_count);
  uint64_t start = get_usecs();
  parallel_batch_insert(g, edges);
  uint64_t end = get_usecs();
  printf("loading the graph took %lu\n", end - start);
  // for (auto edge : edges) {
  //   g.add_edge(edge.first, edge.second);
  // }
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

  if (algorithm_to_run == "tc") {
    uint64_t start = get_usecs();
    uint64_t tris = TC(g);
    uint64_t end = get_usecs();
    printf("triangle count = %ld, took %lu\n", tris, end - start);
  }

  if (algorithm_to_run == "all") {
    run_static_algorithms(g, std::strtol(argv[3], nullptr, 10));
  }
}
