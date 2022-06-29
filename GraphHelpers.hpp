#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <vector>

#include "ParallelTools/parallel.h"

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

template <class G>
void run_static_algorithms(const G &g, uint64_t source_node) {
  uint64_t node_count = g.num_nodes();
  {
    uint64_t start = get_usecs();
    int32_t *bfs_out = BFS(g, source_node);
    uint64_t end = get_usecs();
    std::cout << "bfs took " << end - start << " microseconds\n";
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
    std::cout << "bc took " << end - start << " microseconds\n";
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
    std::cout << "pr took " << end - start << " microseconds\n";
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
    std::cout << "cc took " << end - start << " microseconds\n";
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