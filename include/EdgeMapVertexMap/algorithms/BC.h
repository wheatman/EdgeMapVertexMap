#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/types.h>
#include <vector>

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
using fType = double;

template <typename uintE> struct BC_F {
  static constexpr bool cond_true = false;
  fType *NumPaths;
  bool *Visited;
  BC_F(fType *NumPaths_, bool *Visited_)
      : NumPaths(NumPaths_), Visited(Visited_) {}
  inline bool update(uintE s, uintE d) { // Update function for forward phase
    // printf("update called with %u, %u\n", s, d);
    fType oldV = NumPaths[d];
    NumPaths[d] += NumPaths[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic(uintE s, uintE d) { // atomic Update, basically an
    // printf("updateAtomic called with %u, %u\n", s, d);
    // add
    volatile fType oldV, newV;
    do {
      oldV = NumPaths[d];
      newV = oldV + NumPaths[s];
    } while (!__sync_bool_compare_and_swap((long *)&NumPaths[d],
                                           *((long *)&oldV), *((long *)&newV)));
    return oldV == 0.0;
  }
  inline bool cond(uintE d) { return Visited[d] == 0; } // check if visited
};

template <typename uintE> struct BC_Back_F {
  static constexpr bool cond_true = false;
  fType *Dependencies;
  bool *Visited;
  BC_Back_F(fType *Dependencies_, bool *Visited_)
      : Dependencies(Dependencies_), Visited(Visited_) {}
  inline bool update(uintE s, uintE d) { // Update function for backwards phase
    fType oldV = Dependencies[d];
    Dependencies[d] += Dependencies[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic(uintE s, uintE d) { // atomic Update
    volatile fType oldV, newV;
    do {
      oldV = Dependencies[d];
      newV = oldV + Dependencies[s];
    } while (!__sync_bool_compare_and_swap((long *)&Dependencies[d],
                                           *((long *)&oldV), *((long *)&newV)));
    return oldV == 0.0;
  }
  inline bool cond(uintE d) { return Visited[d] == 0; } // check if visited
};

// vertex map function to mark visited vertexSubset
template <typename uintE> struct BC_Vertex_F {
  bool *Visited;
  explicit BC_Vertex_F(bool *Visited_) : Visited(Visited_) {}
  inline bool operator()(uintE i) {
    Visited[i] = true;
    return true;
  }
};

// vertex map function (used on backwards phase) to mark visited vertexSubset
// and add to Dependencies score
template <typename uintE> struct BC_Back_Vertex_F {
  bool *Visited;
  fType *Dependencies, *inverseNumPaths;
  BC_Back_Vertex_F(bool *Visited_, fType *Dependencies_,
                   fType *inverseNumPaths_)
      : Visited(Visited_), Dependencies(Dependencies_),
        inverseNumPaths(inverseNumPaths_) {}
  inline bool operator()(uintE i) {
    Visited[i] = true;
    Dependencies[i] += inverseNumPaths[i];
    return true;
  }
};

template <class Graph>
fType *BC(const Graph &G, const typename Graph::node_t &start,
          [[maybe_unused]] bool use_dense_forward = false) {
  using node_t = typename Graph::node_t;
  const node_t n = G.num_nodes();
  if (n == 0) {
    return nullptr;
  }
  fType *NumPaths = (fType *)malloc(n * sizeof(fType));

  ParallelTools::parallel_for(0, n, [&](uint64_t i) { NumPaths[i] = 0.0; });

  bool *Visited = (bool *)malloc(n * sizeof(bool));

  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Visited[i] = false; });

  const auto data = EdgeMapVertexMap::getExtraData(G);

  Visited[start] = true;
  NumPaths[start] = 1.0;

  VertexSubset<node_t> Frontier(start,
                                n); // creates initial frontier

  std::vector<VertexSubset<node_t>> Levels;
  Levels.push_back(Frontier);
  int64_t round = 0;
  while (Frontier.non_empty()) {
    round++;
    VertexSubset<node_t> output =
        edgeMap(G, Frontier, BC_F<node_t>(NumPaths, Visited), data, true, 20);
    Levels.push_back(output);
    Frontier = output;
    vertexMap(Frontier, BC_Vertex_F<node_t>(Visited), false); // mark visited
  }

  fType *Dependencies = (fType *)malloc(n * sizeof(fType));
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Dependencies[i] = 0.0; });

  ParallelTools::parallel_for(
      0, n, [&](uint64_t i) { NumPaths[i] = 1 / NumPaths[i]; });
  Levels[round].del();

  ParallelTools::parallel_for(0, n, [&](uint64_t i) { Visited[i] = false; });

  vertexMap(Levels[round - 1],
            BC_Back_Vertex_F<node_t>(Visited, Dependencies, NumPaths), false);
  for (int64_t r = round - 2; r >= 0; r--) {
    edgeMap(G, Levels[r + 1], BC_Back_F<node_t>(Dependencies, Visited), data,
            false, 20);
    Levels[r + 1].del();
    vertexMap(Levels[r],
              BC_Back_Vertex_F<node_t>(Visited, Dependencies, NumPaths), false);
  }
  ParallelTools::parallel_for(0, n, [&](uint64_t i) {
    Dependencies[i] = (Dependencies[i] - NumPaths[i]) / NumPaths[i];
  });
  Levels[0].del();
  free(NumPaths);
  free(Visited);

  return Dependencies;
}
} // namespace EdgeMapVertexMap
