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

  CSR(node_t n, edge_t m, std::vector<std::pair<node_t, node_t>> edges_list)
      : n(n), m(m) {
    std::sort(edges_list.begin(), edges_list.end());
    auto new_end = std::unique(edges_list.begin(), edges_list.end());
    edges_list.resize(std::distance(edges_list.begin(), new_end));
    nodes = (edge_t *)malloc((n + 1) * sizeof(edge_t));
    edges = (node_t *)malloc((m) * sizeof(node_t));
    ParallelTools::parallel_for(
        0, m, [&](edge_t i) { edges[i] = edges_list[i].second; });
    nodes[0] = 0;
    node_t current_node = 0;
    edge_t current_position = 0;
    while (current_node < n && current_position < m) {
      auto edge = edges_list[current_position];
      if (edge.first > current_node) {
        for (node_t i = current_node + 1; i <= edge.first; i++) {
          nodes[i] = current_position;
        }
        current_node = edge.first;
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

  void *getExtraData() const { return nullptr; }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    edge_t start = nodes[node];
    edge_t end = nodes[node + 1];

    if (parallel) {
      ParallelTools::parallel_for(start, end,
                                  [&](edge_t i) { f.update(node, edges[i]); });
    } else {
      for (edge_t i = start; i < end; i++) {
        f.update(node, edges[i]);
      }
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
      get_edges_from_file_adj_sym(graph_filename, &edge_count, &node_count);
  CSR<uint32_t, uint32_t> g =
      CSR<uint32_t, uint32_t>(node_count, edge_count, edges);
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
}
