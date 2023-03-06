#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "ParallelTools/sort.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/BellmanFord.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"
#include "EdgeMapVertexMap/algorithms/TC.h"

namespace EdgeMapVertexMap {

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

template <typename T>
std::vector<std::tuple<T, T>>
uniform_random_sym_edges(T nodes, T expected_edges_per_node) {
  uint64_t expected_edges = nodes * expected_edges_per_node;
  std::vector<std::random_device> rds(ParallelTools::getWorkers());

  std::vector<std::tuple<T, T>> edges(expected_edges);
  ParallelTools::parallel_for(0, nodes, [&](uint64_t j) {
    uint64_t start = j * expected_edges_per_node;
    uint64_t end = (j + 1) * expected_edges_per_node;
    std::mt19937_64 eng(
        rds[ParallelTools::getWorkerNum()]()); // a source of random data

    std::uniform_int_distribution<T> dist(0, nodes - 1);
    for (size_t i = start; i < end; i += 2) {
      T other = dist(eng);
      edges[i] = {j, other};
      edges[i + 1] = {other, j};
    }
  });
  return edges;
}

template <typename T>
std::vector<std::tuple<T, T>>
very_skewed_graph(T nodes, T edges_with_high_degree, T edges_per_high_degree) {
  uint64_t expected_edges = edges_with_high_degree * edges_per_high_degree * 2;
  std::vector<std::random_device> rds(ParallelTools::getWorkers());

  std::vector<std::tuple<T, T>> edges(expected_edges);
  ParallelTools::parallel_for(0, edges_with_high_degree, [&](uint64_t j) {
    uint64_t start = j * edges_per_high_degree * 2;
    uint64_t end = (j + 1) * edges_per_high_degree * 2;
    std::mt19937_64 eng(
        rds[ParallelTools::getWorkerNum()]()); // a source of random data

    std::uniform_int_distribution<T> dist(0, nodes - 1);
    for (size_t i = start; i < end; i += 2) {
      T other = dist(eng);
      edges[i] = {j, other};
      edges[i + 1] = {other, j};
    }
  });
  return edges;
}

template <typename G, typename edge_t>
void parallel_batch_insert(G &g, const std::vector<edge_t> &edges) {
  static constexpr bool binary = std::tuple_size_v<edge_t> == 2;
  uint64_t n_workers = ParallelTools::getWorkers() * 10;
  uint64_t p = std::min(std::max(1UL, edges.size() / 100), n_workers);
  std::vector<uint64_t> indxs(p + 1);
  indxs[0] = 0;
  indxs[p] = edges.size();
  for (uint64_t i = 1; i < p; i++) {
    uint64_t start = std::max((i * edges.size()) / p, indxs[i - 1]);
    if (start >= edges.size()) {
      indxs[i] = edges.size();
      continue;
    }
    auto start_val = std::get<0>(edges[start]);
    while (start < edges.size() && std::get<0>(edges[start]) == start_val) {
      start += 1;
    }
    indxs[i] = start;
  }
  ParallelTools::parallel_for(0, p, [&](size_t i) {
    uint64_t idx = indxs[i];
    uint64_t end = indxs[i + 1];

    for (; idx < end; idx++) {

      // Not including self loops to compare to aspen
      auto x = std::get<0>(edges[idx]);
      auto y = std::get<1>(edges[idx]);
      if (x == y) {
        continue;
      }
      if constexpr (binary) {
        g.add_edge(x, y);
      }
      if constexpr (!binary) {
        auto w = std::get<2>(edges[idx]);
        g.add_edge(x, y, w);
      }
    }
  });
}

template <class G, class T>
uint64_t sum_all_edges_with_order(const G &g, const std::vector<T> &order) {
  ParallelTools::Reducer_sum<uint64_t> sum;
  ParallelTools::parallel_for(0, order.size(), [&](uint64_t i) {
    uint64_t local_sum = 0;
    g.map_neighbors(
        order[i],
        [&local_sum]([[maybe_unused]] auto src, auto dest) {
          local_sum += dest;
        },
        nullptr, false);
    sum.add(local_sum);
  });
  return sum.get();
}

template <bool run_tc, class G>
void run_unweighted_algorithms(const G &g, const std::string &algorithm_to_run,
                               uint64_t src, uint64_t pr_iters) {
  uint64_t node_count = g.num_nodes();

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
  if constexpr (run_tc) {
    if (algorithm_to_run == "tc") {
      uint64_t start = get_usecs();
      uint64_t tris = TC(g);
      uint64_t end = get_usecs();
      printf("triangle count = %ld, took %lu\n", tris, end - start);
    }
  }
}

template <class G>
void run_weighted_algorithms(const G &g, const std::string &algorithm_to_run,
                             uint64_t src) {
  uint64_t node_count = g.num_nodes();
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
}
} // namespace EdgeMapVertexMap
