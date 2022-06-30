#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include "ParallelTools/ParallelTools/parallel.h"
#include "ParallelTools/parallel.h"
#include "ParallelTools/sort.hpp"

#include "algorithms/BC.h"
#include "algorithms/BFS.h"
#include "algorithms/Components.h"
#include "algorithms/PageRank.h"

namespace EdgeMapVertexMap {

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

template <typename T>
std::vector<std::pair<T, T>>
uniform_random_sym_edges(T nodes, T expected_edges_per_node) {
  uint64_t expected_edges = nodes * expected_edges_per_node;
  std::vector<std::random_device> rds(ParallelTools::getWorkers());

  std::vector<std::pair<T, T>> edges(expected_edges);
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

template <class G, typename T>
void parallel_batch_insert(G &g, std::vector<std::pair<T, T>> &edges) {
  ParallelTools::sort(edges.begin(), edges.end());
  uint64_t n_workers = ParallelTools::getWorkers();
  uint64_t p = std::min(std::max(1UL, edges.size() / 100), n_workers);
  std::vector<uint64_t> indxs(p + 1);
  indxs[0] = 0;
  indxs[p] = edges.size();
  for (uint64_t i = 1; i < p; i++) {
    uint64_t start = (i * edges.size()) / p;
    T start_val = std::get<0>(edges[start]);
    while (std::get<0>(edges[start]) == start_val) {
      start += 1;
      if (start == edges.size()) {
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
      T x = std::get<0>(edges[idx]);
      T y = std::get<1>(edges[idx]);
      if (x == y) {
        continue;
      }
      g.add_edge(x, y);
    }
  });
}

template <class G>
void run_static_algorithms(const G &g, uint64_t source_node) {
  uint64_t node_count = g.num_nodes();
  {
    uint64_t start = get_usecs();
    int32_t *bfs_out = BFS(g, source_node);
    uint64_t end = get_usecs();
    std::cout << "bfs took " << double(end - start) / 1000000.0 << " seconds\n";
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
  {
    uint64_t start = get_usecs();
    double *bc_out = BC(g, source_node);
    uint64_t end = get_usecs();
    std::cout << "bc took " << double(end - start) / 1000000.0 << " seconds\n";
    std::ofstream myfile;
    myfile.open("bc.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << bc_out[i] << std::endl;
    }
    myfile.close();
    free(bc_out);
  }
  {
    uint64_t iters = 10;
    uint64_t start = get_usecs();
    double *pr_out = PR_S<double>(g, iters);
    uint64_t end = get_usecs();
    std::cout << "pr took " << double(end - start) / 1000000.0 << " seconds\n";
    std::ofstream myfile;
    myfile.open("pr.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << pr_out[i] << std::endl;
    }
    myfile.close();
    free(pr_out);
  }
  {
    uint64_t start = get_usecs();
    uint32_t *cc_out = CC(g);
    uint64_t end = get_usecs();
    std::cout << "cc took " << double(end - start) / 1000000.0 << " seconds\n";
    std::ofstream myfile;
    myfile.open("cc.out");
    for (unsigned int i = 0; i < node_count; i++) {
      myfile << cc_out[i] << std::endl;
    }
    myfile.close();
    free(cc_out);
  }
}

} // namespace EdgeMapVertexMap