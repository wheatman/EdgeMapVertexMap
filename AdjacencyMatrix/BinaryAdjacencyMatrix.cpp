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

#include "../BitArray.hpp"
#include "../io_util.hpp"

#include "../algorithms/BC.h"
#include "../algorithms/BFS.h"

class BinaryAdjacencyMatrix {
  // data members
  uint64_t n; // num vertices
  BitArray array;

public:
  // function headings
  BinaryAdjacencyMatrix(uint64_t init_n) : n(init_n), array(n * n) {}

  bool has_edge(uint32_t src, uint32_t dest) const {
    return array.get(src * n + dest);
  }
  void add_edge(uint32_t src, uint32_t dest) const {
    array.set(src * n + dest);
  }
  size_t num_nodes() const { return n; }

  void *getExtraData() const { return nullptr; }

  template <class F, class node_t>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    if (parallel) {
      parallel_for(node_t i = 0; i < n; i++) {
        if (array.get(node * n + i)) {
          f.update(node, i);
        }
      }
    } else {
      for (node_t i = 0; i < n; i++) {
        if (array.get(node * n + i)) {
          f.update(node, i);
        }
      }
    }
  }

  template <class F, class node_t>
  void map_range(F f, node_t node_start, node_t node_end,
                 [[maybe_unused]] void *d) const {
    for (node_t i = node_start; i < node_end; i++) {
      map_neighbors(i, f, d, false);
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
  BinaryAdjacencyMatrix g = BinaryAdjacencyMatrix(node_count);
  for (const auto &edge : edges) {
    g.add_edge(edge.first, edge.second);
  }
  std::string algorithm_to_run = std::string(argv[2]);
  if (algorithm_to_run == "bfs") {
    uint64_t source_node = std::strtol(argv[3], nullptr, 10);
    int32_t *bfs_out = BFS(g, source_node);
    std::vector<uint32_t> depths(node_count,
                                 std::numeric_limits<uint32_t>::max());
    parallel_for(uint32_t j = 0; j < node_count; j++) {
      uint32_t current_depth = 0;
      int32_t current_parent = j;
      if (bfs_out[j] < 0) {
        continue;
      }
      while (current_parent != bfs_out[current_parent]) {
        current_depth += 1;
        current_parent = bfs_out[current_parent];
      }
      depths[j] = current_depth;
    }
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
}
