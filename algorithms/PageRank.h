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
#pragma once
#include "../EdgeMap.hpp"
#include "../VertexMap.hpp"
#include "../VertexSubset.hpp"
#include "ParallelTools/parallel.h"
namespace EdgeMapVertexMap {
// template <class vertex>
template <typename T> struct PR_F {
  static constexpr bool cond_true = true;
  T *p_curr, *p_next;
  // vertex* V;
  // PR_F(double* _p_curr, double* _p_next, vertex* _V) :
  PR_F(T *_p_curr, T *_p_next) : p_curr(_p_curr), p_next(_p_next) {}
  inline bool update(uint32_t s, uint32_t d) {
    p_next[d] += p_curr[s];

    return true;
  }
  inline bool updateAtomic([[maybe_unused]] uint32_t s,
                           [[maybe_unused]] uint32_t d) { // atomic Update
    printf("should never be called for now since its always dense\n");

    return true;
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; }
}; // from ligra readme: for cond which always ret true, ret cond_true// return
   // cond_true(d); }};

template <typename T> struct PR_Vertex {
  T *p_curr;
  uint32_t *degree;
  PR_Vertex(T *_p_curr, uint32_t *_degree) : p_curr(_p_curr), degree(_degree) {}
  inline bool operator()(uint32_t i) {
    p_curr[i] = p_curr[i] / degree[i]; // damping*p_next[i] + addedConstant;
    return true;
  }
};

// TODO(wheatman) maybe assume things have a getDegree function
struct PR_get_degree {
  static constexpr bool cond_true = true;
  uint32_t *degree;
  PR_get_degree(uint32_t *_degree) : degree(_degree) {}
  inline bool update([[maybe_unused]] uint32_t s, uint32_t d) {
    degree[d]++;
    return true;
  }
  inline bool updateAtomic([[maybe_unused]] uint32_t s,
                           [[maybe_unused]] uint32_t d) { // atomic Update
    printf("should never be called for now since its always dense\n");

    return true;
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; }
};

// resets p
template <typename T> struct PR_Vertex_Reset {
  T *p;
  explicit PR_Vertex_Reset(T *_p) : p(_p) {}
  inline bool operator()(uint32_t i) {
    p[i] = 0.0;
    return true;
  }
};

template <typename T, typename Graph>
T *PR_S(const Graph &G, int64_t maxIters) {
  size_t n = G.num_nodes();

  T one_over_n = 1 / (double)n;

  size_t size = n + (4 - (n % 4));
  T *p_curr = (T *)memalign(32, size * sizeof(T));
  T *p_next = (T *)memalign(32, size * sizeof(T));
  uint32_t *degree = (uint32_t *)memalign(32, size * sizeof(uint32_t));

  ParallelTools::parallel_for(0, n,
                              [&](uint64_t i) { p_curr[i] = one_over_n; });
  ParallelTools::parallel_for(0, n, [&](uint64_t i) { degree[i] = 0; });
  // passing in a flag here is becuase the examples of extra data I currently
  // have found don't need to run in PageRank, so we skip it
  const auto data = G.getExtraData(true);
  VertexSubset<uint32_t> Frontier = VertexSubset<uint32_t>(0, n, true);
  edgeMap(G, Frontier, PR_get_degree(degree), data, false);

  int64_t iter = 0;
  // printf("max iters %lu\n", maxIters);
  while (iter++ < maxIters) {
    // using flat snapshot
    vertexMap(Frontier, PR_Vertex(p_curr, degree), false);
    vertexMap(Frontier, PR_Vertex_Reset<T>(p_next), false);
    edgeMap(G, Frontier, PR_F<T>(p_curr, p_next), data, false, 20);

    std::swap(p_curr, p_next);
  }
  Frontier.del();
  free(p_next);
  free(degree);
  return p_curr;
}
} // namespace EdgeMapVertexMap