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

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

using namespace EdgeMapVertexMap;

template <class node_t> class AdjacencyVector {
  // data members
  std::vector<std::vector<node_t>> nodes;

public:
  // function headings

  AdjacencyVector(node_t n, std::vector<std::tuple<node_t, node_t>> &edges_list)
      : nodes(n) {
    std::vector<node_t> shuffle_map(n);
    ParallelTools::parallel_for(0, n, [&](size_t i) { shuffle_map[i] = i; });

    std::random_device rd;
    std::shuffle(shuffle_map.begin(), shuffle_map.end(), rd);
    ParallelTools::sort(
        edges_list.begin(), edges_list.end(), [&shuffle_map](auto &a, auto &b) {
          return std::tie(shuffle_map[std::get<0>(a)], std::get<1>(a)) <
                 std::tie(shuffle_map[std::get<0>(b)], std::get<1>(b));
          ;
        });
    auto new_end = std::unique(edges_list.begin(), edges_list.end());
    edges_list.resize(std::distance(edges_list.begin(), new_end));

    uint64_t n_workers = ParallelTools::getWorkers();
    uint64_t p = std::min(std::max(1UL, edges_list.size() / 100), n_workers);
    std::vector<uint64_t> indxs(p + 1);
    indxs[0] = 0;
    indxs[p] = edges_list.size();
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * edges_list.size()) / p;
      node_t start_val = std::get<0>(edges_list[start]);
      while (std::get<0>(edges_list[start]) == start_val) {
        start += 1;
        if (start == edges_list.size()) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](size_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        node_t x = std::get<0>(edges_list[idx]);
        node_t y = std::get<1>(edges_list[idx]);
        if (x == y) {
          continue;
        }
        nodes[x].push_back(y);
      }
    });
  }

  size_t num_nodes() const { return nodes.size(); }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {

    if (parallel) {
      ParallelTools::parallel_for(0, nodes[node].size(),
                                  [&](size_t i) { f(node, nodes[node][i]); });
    } else {
      for (size_t i = 0; i < nodes[node].size(); i++) {
        f(node, nodes[node][i]);
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
  std::vector<std::tuple<uint32_t, uint32_t>> edges;
  if (graph_filename != "random") {
    edges =
        get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
  } else {
    node_count = 100000000;
    edges = uniform_random_sym_edges<uint32_t>(node_count, 10);
  }

  AdjacencyVector<uint32_t> g = AdjacencyVector<uint32_t>(node_count, edges);
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
