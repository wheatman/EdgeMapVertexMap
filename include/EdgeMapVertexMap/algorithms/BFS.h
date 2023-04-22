#pragma once

#include <cstdint>

#include "EdgeMapVertexMap/internal/EdgeMap.hpp"
#include "EdgeMapVertexMap/internal/VertexMap.hpp"
#include "EdgeMapVertexMap/internal/VertexSubset.hpp"
#include "ParallelTools/parallel.h"

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
namespace EdgeMapVertexMap {
template <typename node_t> struct BFS_F {
  static constexpr bool cond_true = false;
  int32_t *Parents;
  explicit BFS_F(int32_t *Parents_) : Parents(Parents_) {}
  inline bool update(node_t s, node_t d) { // Update
    // printf("update %u, %u\n", s, d);
    if (Parents[d] == -1) {
      Parents[d] = s;
      return true;
    }
    return false;
  }
  inline bool updateAtomic(node_t s, node_t d) { // atomic version of Update
    // printf("updateAtomic %u, %u\n", s, d);
    return __sync_bool_compare_and_swap(&Parents[d], -1, s);
  }
  // cond function checks if vertex has been visited yet
  inline bool cond(node_t d) { return (Parents[d] == -1); }
};

template <class Graph>
int32_t *BFS(const Graph &G, typename Graph::node_t src) {
  using node_t = typename Graph::node_t;
  node_t start = src;
  int64_t n = G.num_nodes();
  if (n == 0) {
    return nullptr;
  }
  const auto data = EdgeMapVertexMap::getExtraData(G);

  // creates Parents array, initialized to all -1, except for start
  int32_t *Parents = (int32_t *)malloc(n * sizeof(int32_t));
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Parents[i] = -1; });
  if (n == 0) {
    return Parents;
  }
  Parents[start] = start;
  VertexSubset<node_t> frontier =
      VertexSubset<node_t>(start, n); // creates initial frontier
  while (frontier.non_empty()) {      // loop until frontier is empty
    auto next_frontier =
        edgeMap(G, frontier, BFS_F<node_t>(Parents), data, true, 20);
    frontier.del();
    frontier = next_frontier;
  }
  frontier.del();
  return Parents;
}
} // namespace EdgeMapVertexMap
