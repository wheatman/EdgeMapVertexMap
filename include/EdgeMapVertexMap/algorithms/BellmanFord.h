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
using uintE = uint32_t;
using intE = int32_t;

struct BF_F {
  static constexpr bool cond_true = true;
  intE *ShortestPathLen;
  int *Visited;
  BF_F(intE *_ShortestPathLen, int *_Visited)
      : ShortestPathLen(_ShortestPathLen), Visited(_Visited) {}
  // Update ShortestPathLen if found a shorter path
  inline bool update(uintE s, uintE d, intE edgeLen) {
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
  inline bool updateAtomic(uintE s, uintE d, intE edgeLen) { // atomic Update
    intE newDist = ShortestPathLen[s] + edgeLen;
    return (writeMin(&ShortestPathLen[d], newDist) && CAS(&Visited[d], 0, 1));
  }
  inline bool cond([[maybe_unused]] uintE d) { return true; }
};

// reset visited vertices
struct BF_Vertex_F {
  int *Visited;
  BF_Vertex_F(int *_Visited) : Visited(_Visited) {}
  inline bool operator()(uintE i) {
    Visited[i] = 0;
    return 1;
  }
};

template <typename Graph> intE *BF(const Graph &G, uint32_t start) {
  uint64_t n = G.get_rows();

  const auto data = EdgeMapVertexMap::getExtraData(G);
  // initialize ShortestPathLen to "infinity"
  intE *ShortestPathLen = (intE *)malloc(n * sizeof(intE));

  ParallelTools::parallel_for(0, n, [&](uint64_t i) {
    ShortestPathLen[i] = std::numeric_limits<int>::max() / 2;
  });

  ShortestPathLen[start] = 0;

  int *Visited = (int *)malloc(n * sizeof(int));
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Visited[i] = 0; });

  VertexSubset frontier = VertexSubset(start, n); // creates initial frontier

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
        edgeMap(G, frontier, BF_F(ShortestPathLen, Visited), data);
    vertexMap(output, BF_Vertex_F(Visited), false);
    frontier.del();
    frontier = output;
    round++;
  }
  frontier.del();
  free(Visited);
  return ShortestPathLen;
}
} // namespace EdgeMapVertexMap
