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

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/BellmanFord.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"
#include "EdgeMapVertexMap/algorithms/TC.h"

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
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t pr_iters = result["priters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  bool use_weights = result["weights"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;
  if (!use_weights) {

    std::vector<std::tuple<uint32_t, uint32_t>> edges;
    if (graph_filename == "skew") {
      node_count = 10000000;
      edges = very_skewed_graph<uint32_t>(node_count, 100, node_count / 2);
    } else {
      edges = get_edges_from_file_adj(graph_filename, &edge_count, &node_count,
                                      true);
    }
    AdjacencyBSet<uint64_t> g = AdjacencyBSet<uint64_t>(node_count);
    uint64_t start = get_usecs();
    parallel_batch_insert(g, edges);
    uint64_t end = get_usecs();
    printf("loading the graph took %lu\n", end - start);
    // for (auto edge : edges) {
    //   g.add_edge(edge.first, edge.second);
    // }
    if (algorithm_to_run == "bfs") {
      int32_t *bfs_out = BFS(g, src);
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
      double *bc_out = BC(g, src);
      std::ofstream myfile;
      myfile.open("bc.out");
      for (unsigned int i = 0; i < node_count; i++) {
        myfile << bc_out[i] << std::endl;
      }
      myfile.close();
      free(bc_out);
    }
    if (algorithm_to_run == "pr") {
      double *pr_out = PR_S<double>(g, pr_iters);
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
      run_static_algorithms(g, src);
    }
  } else {
    using weight_type = uint32_t;
    auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
        graph_filename, &edge_count, &node_count, true);
    auto g = AdjacencyBSet<uint64_t, weight_type>(node_count);
    uint64_t start = get_usecs();
    parallel_batch_insert(g, edges);
    uint64_t end = get_usecs();
    printf("loading the graph took %lu\n", end - start);
    if (algorithm_to_run == "bf") {
      int32_t *bf_out = BF(g, src);
      std::ofstream myfile;
      myfile.open("bf.out");
      for (unsigned int i = 0; i < node_count; i++) {
        myfile << bf_out[i] << std::endl;
      }
      myfile.close();
      free(bf_out);
    }
    std::string fname = "test_graph.adj";
  }
}
