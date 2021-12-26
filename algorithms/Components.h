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

#include "../EdgeMap.hpp"
#include "../VertexMap.hpp"
#include "../VertexSubset.hpp"
#include "ParallelTools/parallel.h"

using uintE = uint32_t;

struct CC_Shortcut {
  uint32_t *IDs, *prevIDs;
  CC_Shortcut(uint32_t *_IDs, uint32_t *_prevIDs)
      : IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool operator()(uint32_t i) {
    uint32_t l = IDs[IDs[i]];
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
struct CC_Vertex_F {
  uintE *IDs, *prevIDs;
  CC_Vertex_F(uintE *_IDs, uintE *_prevIDs) : IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool operator()(uintE i) {
    prevIDs[i] = IDs[i];
    return true;
  }
};

struct CC_F {

  template <class T> inline bool writeMin(T *a, T b) {
    T c;
    bool r = false;
    do {
      c = *a;
    } while (c > b && !(r = __sync_bool_compare_and_swap(a, c, b)));
    return r;
  }

  static constexpr bool cond_true = true;
  uint32_t *IDs, *prevIDs;
  CC_F(uint32_t *_IDs, uint32_t *_prevIDs) : IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool update(uint32_t s, uint32_t d) { // Update function writes min ID
    uint32_t origID = IDs[d];
    if (IDs[s] < origID) {
      IDs[d] = IDs[s];
      if (origID == prevIDs[d]) {
        return true;
      }
    }
    return false;
  }
  inline bool updateAtomic(uint32_t s, uint32_t d) { // atomic Update
    uint32_t origID = IDs[d];
    return (writeMin(&IDs[d], IDs[s]) && origID == prevIDs[d]);
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; } // does nothing
};

template <typename SM> uint32_t *CC(const SM &G) {
  int64_t n = G.num_nodes();
  uint32_t *IDs = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *prevIDs = (uint32_t *)malloc(n * sizeof(uint32_t));
  // initialize unique IDs
  parallel_for(int64_t i = 0; i < n; i++) { IDs[i] = i; }

  const auto data = G.getExtraData();

  VertexSubset<uint32_t> Active = VertexSubset<uint32_t>(
      0, n, true); // initial frontier contains all vertices

  while (Active.non_empty()) { // iterate until IDS converge
    vertexMap(Active, CC_Vertex_F(IDs, prevIDs), false);
    VertexSubset<uint32_t> next = edgeMap(G, Active, CC_F(IDs, prevIDs), data);
    Active.del();
    Active = next;
  }
  Active.del();
  free(prevIDs);
  return IDs;
}
