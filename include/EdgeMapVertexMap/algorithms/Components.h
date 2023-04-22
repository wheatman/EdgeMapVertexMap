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

#include "EdgeMapVertexMap/internal/EdgeMap.hpp"
#include "EdgeMapVertexMap/internal/VertexMap.hpp"
#include "EdgeMapVertexMap/internal/VertexSubset.hpp"
#include "ParallelTools/parallel.h"
namespace EdgeMapVertexMap {

template <typename node_t> struct CC_Shortcut {
  node_t *IDs, *prevIDs;
  CC_Shortcut(node_t *IDs_, node_t *prevIDs_) : IDs(IDs_), prevIDs(prevIDs_) {}
  inline bool operator()(node_t i) {
    node_t l = IDs[IDs[i]];
    if (IDs[i] != l) {
      IDs[i] = l;
    }
    if (prevIDs[i] != IDs[i]) {
      prevIDs[i] = IDs[i];
      return true;
    }
    return false;
  }
};
template <typename node_t> struct CC_Vertex_F {
  node_t *IDs, *prevIDs;
  CC_Vertex_F(node_t *IDs_, node_t *prevIDs_) : IDs(IDs_), prevIDs(prevIDs_) {}
  inline bool operator()(node_t i) {
    prevIDs[i] = IDs[i];
    return true;
  }
};

template <typename node_t> struct CC_F {

  template <class T> inline bool writeMin(T *a, T b) {
    T c;
    bool r = false;
    do {
      c = *a;
    } while (c > b && !(r = __sync_bool_compare_and_swap(a, c, b)));
    return r;
  }

  static constexpr bool cond_true = true;
  node_t *IDs, *prevIDs;
  CC_F(node_t *IDs_, node_t *prevIDs_) : IDs(IDs_), prevIDs(prevIDs_) {}
  inline bool update(node_t s, node_t d) { // Update function writes min ID
    node_t origID = IDs[d];
    if (IDs[s] < origID) {
      IDs[d] = IDs[s];
      if (origID == prevIDs[d]) {
        return true;
      }
    }
    return false;
  }
  inline bool updateAtomic(node_t s, node_t d) { // atomic Update
    node_t origID = IDs[d];
    return (writeMin(&IDs[d], IDs[s]) && origID == prevIDs[d]);
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; } // does nothing
};

template <typename Graph> typename Graph::node_t *CC(const Graph &G) {
  using node_t = typename Graph::node_t;
  int64_t n = G.num_nodes();
  node_t *IDs = (node_t *)malloc(n * sizeof(node_t));
  node_t *prevIDs = (node_t *)malloc(n * sizeof(node_t));
  // initialize unique IDs
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { IDs[i] = i; });

  const auto data = EdgeMapVertexMap::getExtraData(G);

  VertexSubset<node_t> Active(0, n,
                              true); // initial frontier contains all vertices

  while (Active.non_empty()) { // iterate until IDS converge
    vertexMap(Active, CC_Vertex_F<node_t>(IDs, prevIDs), false);
    VertexSubset<node_t> next =
        edgeMap(G, Active, CC_F<node_t>(IDs, prevIDs), data);
    Active.del();
    Active = next;
  }
  Active.del();
  free(prevIDs);
  return IDs;
}
} // namespace EdgeMapVertexMap
