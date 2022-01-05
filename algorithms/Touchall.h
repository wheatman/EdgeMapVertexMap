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
// #include "ligra.h"
#pragma once
#include "../EdgeMap.hpp"
#include "../VertexMap.hpp"
#include "../VertexSubset.hpp"
#include "ParallelTools/parallel.h"

#include <cstdint>
#include <vector>

// template <class vertex>
struct TOUCH_F {
  uint64_t *count_vector;
  explicit TOUCH_F(uint64_t *count_vector_) : count_vector(count_vector_) {}
  inline bool update([[maybe_unused]] uint32_t s, uint32_t d) {
    count_vector[d] += s;
    return true;
  }
  inline bool updateAtomic([[maybe_unused]] uint32_t s,
                           [[maybe_unused]] uint32_t d) { // atomic Update
    printf("should never be called for now since its always dense\n");
    count_vector[s] += d;
    return true;
  }
  inline bool cond([[maybe_unused]] el_t d) { return true; }
}; // from ligra readme: for cond which always ret true, ret cond_true// return
   // cond_true(d); }};

template <typename Graph> uint64_t TouchAll(const Graph &G) {
  size_t n = G.get_rows();
  VertexSubset<uint32_t> Frontier = VertexSubset<uint32_t>(0, n, true);
  std::vector<uint64_t> count_vector(n, 0);
  edgeMap(G, Frontier, TOUCH_F(count_vector.data()), false);
  uint64_t count = 0;
  std::vector<uint64_t> count_vector2(ParallelTools::getWorkers() * 8, 0);
  ParallelTools::parallel_for(0, n, [&](uint64_t i) {
    uint32_t worker_num = getWorkerNum();
    count_vector2[8 * worker_num] += count_vector[i];
  });
  for (auto c : count_vector2) {
    count += c;
  }
  return count;
}
