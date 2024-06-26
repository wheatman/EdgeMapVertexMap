// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <cstdint>

#include "EdgeMapVertexMap/internal/EdgeMap.hpp"
#include "EdgeMapVertexMap/internal/VertexMap.hpp"
#include "EdgeMapVertexMap/internal/VertexSubset.hpp"
#include "ParallelTools/parallel.h"

namespace EdgeMapVertexMap {
using intE = int32_t;

template <typename node_t> struct BF_F {

  template <class T> inline bool writeMin(T *a, T b) {
    T c;
    bool r = false;
    do {
      c = *a;
    } while (c > b && !(r = __sync_bool_compare_and_swap(a, c, b)));
    return r;
  }

  template <class ET> inline bool CAS(ET *ptr, ET oldv, ET newv) {
    if constexpr (sizeof(ET) == 1) {
      return __sync_bool_compare_and_swap((bool *)ptr, *((bool *)&oldv),
                                          *((bool *)&newv));
    } else if constexpr (sizeof(ET) == 4) {
      return __sync_bool_compare_and_swap((int *)ptr, *((int *)&oldv),
                                          *((int *)&newv));
    } else if constexpr (sizeof(ET) == 8) {
      return __sync_bool_compare_and_swap((long *)ptr, *((long *)&oldv),
                                          *((long *)&newv));
    } else {
      std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
      abort();
    }
  }
  static constexpr bool cond_true = true;
  intE *ShortestPathLen;
  int *Visited;
  BF_F(intE *ShortestPathLen_, int *Visited_)
      : ShortestPathLen(ShortestPathLen_), Visited(Visited_) {}
  // Update ShortestPathLen if found a shorter path
  inline bool update(node_t s, node_t d, intE edgeLen) {
    intE newDist = ShortestPathLen[s] + edgeLen;
    if (ShortestPathLen[d] > newDist) {
      ShortestPathLen[d] = newDist;
      if (Visited[d] == 0) {
        Visited[d] = 1;
        return 1;
      }
    }
    return 0;
  }
  inline bool updateAtomic(node_t s, node_t d, intE edgeLen) { // atomic Update
    intE newDist = ShortestPathLen[s] + edgeLen;
    return (writeMin(&ShortestPathLen[d], newDist) && CAS(&Visited[d], 0, 1));
  }
  inline bool cond([[maybe_unused]] node_t d) { return true; }
};

// reset visited vertices
template <typename node_t> struct BF_Vertex_F {
  int *Visited;
  BF_Vertex_F(int *Visited_) : Visited(Visited_) {}
  inline bool operator()(node_t i) {
    Visited[i] = 0;
    return 1;
  }
};

template <typename Graph> intE *BF(const Graph &G, uint32_t start) {
  using node_t = typename Graph::node_t;
  uint64_t n = G.num_nodes();
  assert(start < n);

  const auto data = EdgeMapVertexMap::getExtraData(G);
  // initialize ShortestPathLen to "infinity"
  intE *ShortestPathLen = (intE *)malloc(n * sizeof(intE));

  ParallelTools::parallel_for(0, n, [&](uint64_t i) {
    ShortestPathLen[i] = std::numeric_limits<int>::max() / 2;
  });

  ShortestPathLen[start] = 0;

  int *Visited = (int *)malloc(n * sizeof(int));
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Visited[i] = 0; });

  VertexSubset<node_t> frontier(start, n); // creates initial frontier

  uint64_t round = 0;
  while (frontier.non_empty()) {
    // printf("round %lu, num vertices %lu\n", round, frontier.get_n());
    if (round == n) {
      // negative weight cycle

      ParallelTools::parallel_for(0, n, [&](uint64_t i) {
        ShortestPathLen[i] = -(std::numeric_limits<intE>::max() / 2);
      });
      return ShortestPathLen;
    }
    VertexSubset output =
        edgeMap(G, frontier, BF_F<node_t>(ShortestPathLen, Visited), data);
    vertexMap(output, BF_Vertex_F<node_t>(Visited), false);
    frontier.del();
    frontier = std::move(output);
    round++;
  }
  frontier.del();
  free(Visited);
  return ShortestPathLen;
}
} // namespace EdgeMapVertexMap
