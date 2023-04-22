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
#include <cstdint>

#include "EdgeMapVertexMap/internal/EdgeMap.hpp"
#include "EdgeMapVertexMap/internal/VertexMap.hpp"
#include "EdgeMapVertexMap/internal/VertexSubset.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"

namespace EdgeMapVertexMap {

template <class Graph, class node_t> struct countF { // for edgeMap
  static constexpr bool cond_true = true;
  const Graph &G;
  ParallelTools::Reducer_sum<uint64_t> &counts;
  countF(const Graph &G_, ParallelTools::Reducer_sum<uint64_t> &_counts)
      : G(G_), counts(_counts) {}
  inline bool update(node_t s, node_t d) {
    if (s > d) { // only count "directed" triangles
      counts.add(G.common_neighbors(s, d));
    }
    return true;
  }
  inline bool updateAtomic(node_t s, node_t d) {
    if (s > d) { // only count "directed" triangles
      counts.add(G.common_neighbors(s, d));
    }
    return true;
  }
  inline bool cond([[maybe_unused]] node_t d) { return true; } // does nothing
};

template <class Graph> uint64_t TC(const Graph &G) {
  auto n = G.num_nodes();
  ParallelTools::Reducer_sum<uint64_t> counts;
  VertexSubset<typename Graph::node_t> Frontier(
      0, n, true); // frontier contains all vertices
  const auto data = EdgeMapVertexMap::getExtraData(G, true);

  edgeMap(G, Frontier, countF<Graph, typename Graph::node_t>(G, counts), data,
          false);
  uint64_t count = counts.get();
  return count;
}
} // namespace EdgeMapVertexMap
