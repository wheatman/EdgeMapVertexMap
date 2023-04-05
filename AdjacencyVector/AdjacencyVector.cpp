/*
 * adjacency matrix
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "EdgeMapVertexMap/internal/GraphHelpers.hpp"
#include "EdgeMapVertexMap/internal/io_util.hpp"

#include "cxxopts.hpp"

using namespace EdgeMapVertexMap;

template <class node_t, class weight_t = bool> class AdjacencyVector {

  static constexpr bool binary = std::is_same_v<weight_t, bool>;
  // data members
  std::vector<typename std::conditional<
      binary, std::vector<node_t>,
      std::pair<std::vector<node_t>, std::vector<weight_t>>>::type>
      nodes;

public:
  // function headings

  AdjacencyVector(node_t n, std::vector<std::tuple<node_t, node_t>> &edges_list)
      : nodes(n) {
    static_assert(binary);
    // shuffle map ensures that the different vectors are not likely to be laid
    // out next to each other in memory
    std::vector<node_t> shuffle_map(n);
    ParallelTools::parallel_for(0, n, [&](size_t i) { shuffle_map[i] = i; });

    std::random_device rd;
    std::shuffle(shuffle_map.begin(), shuffle_map.end(), rd);
    ParallelTools::sort(
        edges_list.begin(), edges_list.end(), [&shuffle_map](auto &a, auto &b) {
          return std::tie(shuffle_map[std::get<0>(a)], std::get<1>(a)) <
                 std::tie(shuffle_map[std::get<0>(b)], std::get<1>(b));
          ;
        });
    auto new_end = std::unique(edges_list.begin(), edges_list.end());
    edges_list.resize(std::distance(edges_list.begin(), new_end));

    uint64_t n_workers = ParallelTools::getWorkers();
    uint64_t p = std::min(std::max(1UL, edges_list.size() / 100), n_workers);
    std::vector<uint64_t> indxs(p + 1);
    indxs[0] = 0;
    indxs[p] = edges_list.size();
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * edges_list.size()) / p;
      node_t start_val = std::get<0>(edges_list[start]);
      while (std::get<0>(edges_list[start]) == start_val) {
        start += 1;
        if (start == edges_list.size()) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](size_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        node_t x = std::get<0>(edges_list[idx]);
        node_t y = std::get<1>(edges_list[idx]);
        if (x == y) {
          continue;
        }
        nodes[x].push_back(y);
      }
    });
  }

  AdjacencyVector(node_t n,
                  std::vector<std::tuple<node_t, node_t, weight_t>> &edges_list)
      : nodes(n) {
    static_assert(!binary);
    // shuffle map ensures that the different vectors are not likely to be laid
    // out next to each other in memory
    std::vector<node_t> shuffle_map(n);
    ParallelTools::parallel_for(0, n, [&](size_t i) { shuffle_map[i] = i; });

    std::random_device rd;
    std::shuffle(shuffle_map.begin(), shuffle_map.end(), rd);
    ParallelTools::sort(
        edges_list.begin(), edges_list.end(), [&shuffle_map](auto &a, auto &b) {
          return std::tie(shuffle_map[std::get<0>(a)], std::get<1>(a)) <
                 std::tie(shuffle_map[std::get<0>(b)], std::get<1>(b));
          ;
        });
    auto new_end =
        std::unique(edges_list.begin(), edges_list.end(),
                    [](const auto &tup1, const auto &tup2) {
                      return std::get<0>(tup1) == std::get<0>(tup2) &&
                             std::get<1>(tup1) == std::get<1>(tup2);
                    });
    edges_list.resize(std::distance(edges_list.begin(), new_end));

    uint64_t n_workers = ParallelTools::getWorkers();
    uint64_t p = std::min(std::max(1UL, edges_list.size() / 100), n_workers);
    std::vector<uint64_t> indxs(p + 1);
    indxs[0] = 0;
    indxs[p] = edges_list.size();
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * edges_list.size()) / p;
      node_t start_val = std::get<0>(edges_list[start]);
      while (std::get<0>(edges_list[start]) == start_val) {
        start += 1;
        if (start == edges_list.size()) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](size_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        node_t x = std::get<0>(edges_list[idx]);
        node_t y = std::get<1>(edges_list[idx]);
        if (x == y) {
          continue;
        }
        weight_t w = std::get<2>(edges_list[idx]);
        nodes[x].first.push_back(y);
        nodes[x].second.push_back(w);
      }
    });
  }

  size_t num_nodes() const { return nodes.size(); }

  template <class F>
  void map_neighbors(node_t node, F f, [[maybe_unused]] void *d,
                     bool parallel) const {
    if constexpr (binary) {
      if (parallel) {
        ParallelTools::parallel_for(0, nodes[node].size(),
                                    [&](size_t i) { f(node, nodes[node][i]); });
      } else {
        for (size_t i = 0; i < nodes[node].size(); i++) {
          f(node, nodes[node][i]);
        }
      }
    } else {
      if (parallel) {
        ParallelTools::parallel_for(0, nodes[node].first.size(), [&](size_t i) {
          f(node, nodes[node].first[i], nodes[node].second[i]);
        });
      } else {
        for (size_t i = 0; i < nodes[node].first.size(); i++) {
          f(node, nodes[node].first[i], nodes[node].second[i]);
        }
      }
    }
  }
};

int main(int32_t argc, char *argv[]) {
  cxxopts::Options options("Graph tester",
                           "Runs different algorithms on a Graph");
  add_options_to_parser(options);
  auto result = options.parse(argc, argv);

  std::string graph_filename = result["graph"].as<std::string>();
  uint64_t src = result["src"].as<uint64_t>();
  uint64_t iters = result["iters"].as<uint64_t>();
  uint64_t pr_rounds = result["pr_rounds"].as<uint64_t>();
  uint64_t nClusters = result["nClusters"].as<uint64_t>();
  std::string algorithm_to_run = result["algorithm"].as<std::string>();
  std::string y_location = result["y_location"].as<std::string>();
  bool use_weights = result["weights"].as<bool>();
  bool laplacian = result["laplacian"].as<bool>();
  bool dump_output = result["dump_output"].as<bool>();
  uint64_t edge_count;
  uint32_t node_count;

  if (!use_weights) {
    auto edges =
        get_edges_from_file_adj(graph_filename, &edge_count, &node_count, true);
    AdjacencyVector<uint32_t> g(node_count, edges);
    run_unweighted_algorithms<false>(g, algorithm_to_run, src, iters, pr_rounds,
                                     nClusters, y_location, dump_output);
  } else {
    using weight_type = uint32_t;
    auto edges = get_edges_from_file_adj<uint32_t, weight_type>(
        graph_filename, &edge_count, &node_count, true);
    AdjacencyVector<uint32_t, weight_type> g(node_count, edges);
    run_weighted_algorithms(g, algorithm_to_run, src, iters, nClusters,
                            y_location, laplacian, dump_output);
  }
}