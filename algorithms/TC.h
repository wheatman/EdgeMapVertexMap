#pragma once
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

// Triangle counting code (assumes a symmetric graph, so pass the "-s"
// flag). This is not optimized (no ordering heuristic is used)--for
// optimized code, see "Multicore Triangle Computations Without
// Tuning", ICDE 2015. Currently only works with uncompressed graphs,
// and not with compressed graphs.
#include "ParallelTools/reducer.h"

// assumes sorted neighbor lists
// int64_t countCommon(const SparseMatrixV<true, bool> &G, uint32_t a,
//                     uint32_t b) {
//   TinySetV_small<>::iterator it_A = G.neighbor_begin(a);
//   TinySetV_small<>::iterator it_B = G.neighbor_begin(b);
//   TinySetV_small<>::iterator end_A = G.neighbor_end(a);
//   TinySetV_small<>::iterator end_B = G.neighbor_end(b);
//   int64_t ans = 0;
//   while (it_A != end_A && it_B != end_B && (*it_A).first < a &&
//          (*it_B).first < b) { // count "directed" triangles
//     if ((*it_A).first == (*it_B).first) {
//       ++it_A, ++it_B, ans++;
//     } else if ((*it_A).first < (*it_B).first) {
//       ++it_A;
//     } else {
//       ++it_B;
//     }
//   }
//   return ans;
// }

namespace EdgeMapVertexMap {

template <class Graph> struct countF { // for edgeMap
  static constexpr bool cond_true = true;
  const Graph &G;
  ParallelTools::Reducer_sum<uint64_t> &counts;
  countF(const Graph &G_, ParallelTools::Reducer_sum<uint64_t> &_counts)
      : G(G_), counts(_counts) {}
  inline bool update(uint32_t s, uint32_t d) {
    if (s > d) { // only count "directed" triangles
      counts.add(G.common_neighbors(s, d, true));
    }
    return true;
  }
  inline bool updateAtomic(uint32_t s, uint32_t d) {
    if (s > d) { // only count "directed" triangles
      counts.add(G.common_neighbors(s, d, true));
    }
    return true;
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; } // does nothing
};

template <class Graph> void TC(const Graph &G) {
  uint32_t n = G.get_rows();
  ParallelTools::Reducer_sum<uint64_t> counts;
  VertexSubset Frontier(0, n, true); // frontier contains all vertices
  const auto data = EdgeMapVertexMap::getExtraData(G, true);

  edgeMap(G, Frontier, countF(G, counts), data, false);
  uint64_t count = counts.get();
  printf("triangle count = %ld\n", count);
}
} // namespace EdgeMapVertexMap