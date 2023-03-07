#pragma once

#include <assert.h>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "ParallelTools/reducer.h"

#include "EdgeMapVertexMap/internal/EdgeMap.hpp"
#include "EdgeMapVertexMap/internal/VertexMap.hpp"
#include "EdgeMapVertexMap/internal/VertexSubset.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"
#include "ParallelTools/parallel.h"

namespace EdgeMapVertexMap {

struct GEE_F {

  static constexpr bool cond_true = true;
  double *z;

  const int *Y; // Supervised labels for each vertex.

  const int *nk;
  const int n;
  GEE_F(double *z_, const int n_, const int *Y_, const int *nk_)
      : z(z_), Y(Y_), nk(nk_), n(n_) {}

  inline bool update(uint32_t s, uint32_t d) {
    // -1 or negative label means don't know - ignored

    if (Y[s] >= 0) {
      z[Y[s] * n + d] += 1.0 / nk[Y[s]];
    }
    if (Y[d] >= 0 && s != d) {
      z[Y[d] * n + s] += 1.0 / nk[Y[d]];
    }
    return 1;
  }

  inline bool updateAtomic([[maybe_unused]] uint32_t s,
                           [[maybe_unused]] uint32_t d) { // atomic Update
    assert(false);
    return 1;
  }

  inline bool cond([[maybe_unused]] uint32_t d) { return true; }
};

// Embedding Matrix is kxN - map each vertex to a label. GEE iterates over edges

// Run GEE
template <class Graph>
double *GEE(const Graph &G, const int nClusters, std::string_view y_location) {
  if (nClusters <= 0) {
    std::cerr << "you must specify a positive number of clusters\n";
    exit(-1);
  }
  const uint64_t n = G.num_nodes();

  //    in parallel
  double *Z = (double *)malloc((n * nClusters + 1) * sizeof(double));
  ParallelTools::parallel_for(0, n * nClusters, [&](size_t i) { Z[i] = 0; });
  Z[n * nClusters] = NAN;

  int *Y =
      (int *)malloc(n * sizeof(int)); // TODO maybe set some classes to 1. GEE
                                      // chooses 2 of 5 vertices in class 1

  if (y_location != "") {
    int64_t length = 0;
    char *S = readStringFromFile(y_location.data(), &length);
    words W = stringToWords(S, length);
    uint64_t len = W.m;
    if (len == 0) {
      printf("the file appears to have no data, exiting\n");
      exit(-1);
    }
    ParallelTools::parallel_for(
        0, len, [&](size_t i) { Y[i] = std::stoi(W.Strings[i], nullptr, 10); });
  } else {
    std::cerr << "You must specify the location of the Y file\n";
    exit(-1);
  }

  // nk: 1*n array, contains the number of observations in each class

  std::vector<ParallelTools::Reducer_sum<int>> nk_reduce(nClusters);

  ParallelTools::parallel_for(0, n, [&](size_t j) {
    if (Y[j] >= 0) {
      nk_reduce[Y[j]].inc();
    }
  });
  std::vector<int> nk(nClusters);
  // TODO parallelize if nClusters is ever large
  for (int i = 0; i < nClusters; i++) {
    nk[i] = nk_reduce[i];
  }

  auto Frontier = VertexSubset<uint32_t>(0, n, true);

  const auto data = getExtraData(G, true);

  edgeMap(G, Frontier, GEE_F(Z, n, Y, nk.data()), data, false);

  Frontier.del();
  free(Y);
  return Z;
}

} // namespace EdgeMapVertexMap
