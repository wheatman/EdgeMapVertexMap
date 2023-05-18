#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "ParallelTools/sort.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/BellmanFord.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/GraphEncoderEmbedding.hpp"
#include "EdgeMapVertexMap/algorithms/PageRank.h"
#include "EdgeMapVertexMap/algorithms/TC.h"
#include "EdgeMapVertexMap/algorithms/WeightedGraphEncoderEmbedding.hpp"
#include "EdgeMapVertexMap/internal/utils.hpp"

#include "cxxopts.hpp"

namespace EdgeMapVertexMap {

template <typename T>
std::vector<std::tuple<T, T>>
uniform_random_sym_edges(T nodes, T expected_edges_per_node) {
  uint64_t expected_edges = nodes * expected_edges_per_node;
  std::vector<std::random_device> rds(ParallelTools::getWorkers());

  std::vector<std::tuple<T, T>> edges(expected_edges);
  ParallelTools::parallel_for(0, nodes, [&](uint64_t j) {
    uint64_t start = j * expected_edges_per_node;
    uint64_t end = (j + 1) * expected_edges_per_node;
    std::mt19937_64 eng(
        rds[ParallelTools::getWorkerNum()]()); // a source of random data

    std::uniform_int_distribution<T> dist(0, nodes - 1);
    for (size_t i = start; i < end; i += 2) {
      T other = dist(eng);
      edges[i] = {j, other};
      edges[i + 1] = {other, j};
    }
  });
  return edges;
}

template <typename T>
std::vector<std::tuple<T, T>>
very_skewed_graph(T nodes, T edges_with_high_degree, T edges_per_high_degree) {
  uint64_t expected_edges = edges_with_high_degree * edges_per_high_degree * 2;
  std::vector<std::random_device> rds(ParallelTools::getWorkers());

  std::vector<std::tuple<T, T>> edges(expected_edges);
  ParallelTools::parallel_for(0, edges_with_high_degree, [&](uint64_t j) {
    uint64_t start = j * edges_per_high_degree * 2;
    uint64_t end = (j + 1) * edges_per_high_degree * 2;
    std::mt19937_64 eng(
        rds[ParallelTools::getWorkerNum()]()); // a source of random data

    std::uniform_int_distribution<T> dist(0, nodes - 1);
    for (size_t i = start; i < end; i += 2) {
      T other = dist(eng);
      edges[i] = {j, other};
      edges[i + 1] = {other, j};
    }
  });
  return edges;
}

template <typename G, typename edge_t>
void parallel_batch_insert(G &g, const std::vector<edge_t> &edges) {
  static constexpr bool binary = std::tuple_size_v<edge_t> == 2;
  uint64_t n_workers = ParallelTools::getWorkers() * 10;
  uint64_t p = std::min(std::max(1UL, edges.size() / 100), n_workers);
  std::vector<uint64_t> indxs(p + 1);
  indxs[0] = 0;
  indxs[p] = edges.size();
  for (uint64_t i = 1; i < p; i++) {
    uint64_t start = std::max((i * edges.size()) / p, indxs[i - 1]);
    if (start >= edges.size()) {
      indxs[i] = edges.size();
      continue;
    }
    auto start_val = std::get<0>(edges[start]);
    while (start < edges.size() && std::get<0>(edges[start]) == start_val) {
      start += 1;
    }
    indxs[i] = start;
  }
  ParallelTools::parallel_for(0, p, [&](size_t i) {
    uint64_t idx = indxs[i];
    uint64_t end = indxs[i + 1];

    for (; idx < end; idx++) {

      // Not including self loops to compare to aspen
      auto x = std::get<0>(edges[idx]);
      auto y = std::get<1>(edges[idx]);
      if (x == y) {
        continue;
      }
      if constexpr (binary) {
        g.add_edge(x, y);
      }
      if constexpr (!binary) {
        auto w = std::get<2>(edges[idx]);
        g.add_edge(x, y, w);
      }
    }
  });
}

template <class G, class T>
uint64_t sum_all_edges_with_order(const G &g, const std::vector<T> &order) {
  ParallelTools::Reducer_sum<uint64_t> sum;
  ParallelTools::parallel_for(0, order.size(), [&](uint64_t i) {
    uint64_t local_sum = 0;
    g.map_neighbors(
        order[i],
        [&local_sum]([[maybe_unused]] auto src, auto dest) {
          local_sum += dest;
        },
        nullptr, false);
    sum.add(local_sum);
  });
  return sum.get();
}

template <class T>
void write_array_to_file(std::string_view filename, const T *data,
                         uint64_t length) {
  std::ofstream myfile;
  myfile.open(filename.data());
  for (unsigned int i = 0; i < length; i++) {
    myfile << data[i] << std::endl;
  }
  myfile.close();
}
void print_stats_on_times(std::vector<uint64_t> &times, std::string_view exp) {
  std::sort(times.begin(), times.end());
  uint64_t median = times[times.size() / 2];
  uint64_t total = 0;
  for (const auto &time : times) {
    total += time;
  }
  uint64_t average = total / times.size();
  std::cout << exp << " had a median of " << median << " and an average of "
            << average << std::endl;
}

template <bool run_tc, class G>
void run_unweighted_algorithms(const G &g, const std::string &algorithm_to_run,
                               uint64_t src, uint64_t iters, uint64_t pr_rounds,
                               uint64_t nClusters = 0,
                               std::string_view y_location = "",
                               bool dump_output = true) {
  uint64_t node_count = g.num_nodes();

  if (algorithm_to_run == "bfs") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      int32_t *bfs_out = BFS(g, src);
      uint64_t end = get_usecs();
      printf("running bfs tool %lu micros, %d\n", end - start, bfs_out[0]);
      times.push_back(end - start);
      if (i == 0 && dump_output) {
        std::vector<uint32_t> depths(node_count,
                                     std::numeric_limits<uint32_t>::max());
        ParallelTools::parallel_for(0, node_count, [&](uint32_t j) {
          uint32_t current_depth = 0;
          int32_t current_parent = j;
          if (bfs_out[j] < 0) {
            return;
          }
          while (current_parent != bfs_out[current_parent]) {
            current_depth += 1;
            current_parent = bfs_out[current_parent];
          }
          depths[j] = current_depth;
        });
        write_array_to_file("bfs.out", depths.data(), node_count);
      }
      free(bfs_out);
    }
    print_stats_on_times(times, "bfs");
  }
  if (algorithm_to_run == "bc") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      double *bc_out = BC(g, src);
      uint64_t end = get_usecs();
      times.push_back(end - start);
      printf("running bc tool %lu micros, %f\n", end - start, bc_out[0]);
      if (i == 0 && dump_output) {
        write_array_to_file("bc.out", bc_out, node_count);
      }
      free(bc_out);
    }
    print_stats_on_times(times, "bc");
  }
  if (algorithm_to_run == "pr") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      double *pr_out = PR_S<double>(g, pr_rounds);
      uint64_t end = get_usecs();
      times.push_back(end - start);
      printf("running pr tool %lu micros, %f\n", end - start, pr_out[0]);
      if (i == 0 && dump_output) {
        write_array_to_file("pr.out", pr_out, node_count);
      }
      free(pr_out);
    }
    print_stats_on_times(times, "pr");
  }
  if (algorithm_to_run == "cc") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      auto *cc_out = CC(g);
      uint64_t end = get_usecs();
      times.push_back(end - start);
      printf("running cc tool %lu micros, %lu\n", end - start,
             (uint64_t)cc_out[0]);
      if (i == 0 && dump_output) {
        write_array_to_file("cc.out", cc_out, node_count);
      }
      free(cc_out);
    }
    print_stats_on_times(times, "cc");
  }
  if (algorithm_to_run == "gee") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      auto *Z = GEE<float>(g, nClusters, y_location);
      uint64_t end = get_usecs();
      times.push_back(end - start);
      printf("running gee tool %lu micros, %f\n", end - start, Z[0]);
      if (i == 0 && dump_output) {
        std::ofstream myfile;
        myfile.open("gee.out");
        for (uint64_t i = 0; i < node_count; i++) {
          for (uint64_t j = 0; j < nClusters; j++) {
            myfile << std::fixed << std::setprecision(6)
                   << Z[j * node_count + i];
            if (j != nClusters - 1) {
              myfile << " ";
            }
          }
          myfile << "\n";
        }
        myfile.close();
      }
      free(Z);
    }
    print_stats_on_times(times, "gee");
  }
  if constexpr (run_tc) {
    if (algorithm_to_run == "tc") {
      std::vector<uint64_t> times;
      for (size_t i = 0; i < iters; i++) {
        uint64_t start = get_usecs();
        uint64_t tris = TC(g);
        uint64_t end = get_usecs();
        times.push_back(end - start);
        printf("triangle count = %ld, took %lu\n", tris, end - start);
      }
      print_stats_on_times(times, "tc");
    }
  }
}

template <class G>
void run_weighted_algorithms(const G &g, const std::string &algorithm_to_run,
                             uint64_t src, uint64_t iters,
                             uint64_t nClusters = 0,
                             std::string_view y_location = "",
                             bool laplacian = false, bool dump_output = true) {
  uint64_t node_count = g.num_nodes();
  if (algorithm_to_run == "bf") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      int32_t *bf_out = BF(g, src);
      uint64_t end = get_usecs();
      times.push_back(end - start);
      printf("running bf tool %lu micros, %d\n", end - start, bf_out[0]);
      if (i == 0 && dump_output) {
        write_array_to_file("bf.out", bf_out, node_count);
      }
      free(bf_out);
    }
    print_stats_on_times(times, "bf");
  }
  if (algorithm_to_run == "gee") {
    std::vector<uint64_t> times;
    for (size_t i = 0; i < iters; i++) {
      uint64_t start = get_usecs();
      double *Z = GEE_Weighted(g, nClusters, y_location, laplacian);
      uint64_t end = get_usecs();
      printf("running wgee tool %lu micros, %f\n", end - start, Z[0]);
      times.push_back(end - start);
      if (i == 0 && dump_output) {
        std::ofstream myfile;
        myfile.open("gee_weighted.out");
        for (uint64_t i = 0; i < node_count; i++) {
          for (uint64_t j = 0; j < nClusters; j++) {
            myfile << std::fixed << std::setprecision(6)
                   << Z[j * node_count + i];
            if (j != nClusters - 1) {
              myfile << " ";
            }
          }
          myfile << "\n";
        }
        myfile.close();
      }
      free(Z);
    }
    print_stats_on_times(times, "wgee");
  }
}

inline void add_options_to_parser(cxxopts::Options &options) {
  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("src","what node to start from",cxxopts::value<uint64_t>()->default_value("0"))
    ("iters","how many iters for for each algorithm",cxxopts::value<uint64_t>()->default_value("10"))
    ("pr_rounds","how many rounds for pr",cxxopts::value<uint64_t>()->default_value("10"))
    ("g,graph", "graph file path", cxxopts::value<std::string>())
    ("algorithm", "which algorithm to run", cxxopts::value<std::string>())
    ("w,weights", "run with a weighted graph", cxxopts::value<bool>()->default_value("false")) 
    ("s,symetric", "symeterizes the graph while reading it in and then run on an undirected graph", cxxopts::value<bool>()->default_value("true")) 
    ("nClusters", "number of clusters for algorithms that need it, currently only gee", cxxopts::value<uint64_t>()->default_value("0")) 
    ("y_location", "path to the y vector of GEE", cxxopts::value<std::string>()->default_value("")) 
    ("laplacian", "use the laplacian in weighted GEE", cxxopts::value<bool>()->default_value("false")) 
    ("dump_output", "write the output arrays to a file", cxxopts::value<bool>()->default_value("true")) 
    ("help","Print help");
  // clang-format on
}
} // namespace EdgeMapVertexMap
