#pragma once
#include "ParallelTools/parallel.h"
#include "ParallelTools/sort.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace EdgeMapVertexMap {

// A structure that keeps a sequence of strings all allocated from
// the same block of memory
struct words {
  char *Chars;    // array storing all strings
  long n;         // total number of characters
  char **Strings; // pointers to strings (all should be null terminated)
  long m;         // number of substrings
  words() {}
  words(char *C, long nn, char **S, long mm)
      : Chars(C), n(nn), Strings(S), m(mm) {}
  void del() {
    free(Chars);
    free(Strings);
  }
};

inline bool isSpace(char c) {
  switch (c) {
  case '\r':
  case '\t':
  case '\n':
  case 0:
  case ' ':
    return true;
  default:
    return false;
  }
}
// parallel code for converting a string to words
words stringToWords(char *Str, uint64_t n) {
  ParallelTools::parallel_for(0, n, [&](size_t i) {
    if (isSpace(Str[i]))
      Str[i] = 0;
  });

  // mark start of words
  bool *FL = (bool *)malloc(n);
  FL[0] = Str[0];
  ParallelTools::parallel_for(1, n,
                              [&](size_t i) { FL[i] = Str[i] && !Str[i - 1]; });

  uint32_t worker_count = ParallelTools::getWorkers();
  std::vector<uint64_t> sub_counts(worker_count, 0);
  uint64_t section_count = (n / worker_count) + 1;
  ParallelTools::parallel_for(0, worker_count, [&](size_t i) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i + 1) * section_count, n);
    uint64_t local_count = 0;
    for (uint64_t j = start; j < end; j++) {
      if (FL[j]) {
        local_count += 1;
      }
    }
    sub_counts[i] = local_count;
  });
  // count and prefix sum
  for (uint32_t i = 1; i < worker_count; i++) {
    sub_counts[i] += sub_counts[i - 1];
  }
  uint64_t m = sub_counts[worker_count - 1];
  uint64_t *offsets = (uint64_t *)malloc(m * sizeof(uint64_t));
  ParallelTools::parallel_for(0, worker_count, [&](size_t i) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i + 1) * section_count, n);
    uint64_t offset;
    if (i == 0)
      offset = 0;
    else
      offset = sub_counts[i - 1];
    for (uint64_t j = start; j < end; j++) {
      if (FL[j] == 1) {
        offsets[offset++] = j;
      }
    }
  });

  // pointer to each start of word
  char **SA = (char **)malloc(m * sizeof(char *));
  ParallelTools::parallel_for(0, m,
                              [&](size_t j) { SA[j] = Str + offsets[j]; });

  free(offsets);
  free(FL);
  return words(Str, n, SA, m);
}
char *readStringFromFile(const char *fileName, long *length) {
  std::ifstream file(fileName, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << fileName << std::endl;
    abort();
  }
  long end = file.tellg();
  file.seekg(0, std::ios::beg);
  long n = end - file.tellg();
  char *bytes = (char *)malloc(n + 1);
  file.read(bytes, n);
  file.close();
  *length = n;
  return bytes;
}

template <typename node_t = uint32_t, typename weight_t = bool>
auto get_edges_from_file_adj(const std::string &filename, uint64_t *edge_count,
                             uint32_t *node_count, bool symmetrize = true) {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;
  using edge_type =
      typename std::conditional<binary, std::tuple<node_t, node_t>,
                                std::tuple<node_t, node_t, weight_t>>::type;

  if constexpr (!std::is_integral_v<node_t>) {
    printf("get_edges_from_file_adj can only do integral node identifiers\n");
    exit(-1);
  }
  int64_t length = 0;
  char *S = readStringFromFile(filename.c_str(), &length);
  if (length == 0) {
    printf("file has 0 length, exiting\n");
    exit(-1);
  }
  words W = stringToWords(S, length);
  if (strcmp(W.Strings[0], "AdjacencyGraph") &&
      strcmp(W.Strings[0], "WeightedAdjacencyGraph")) {
    std::cout << "Bad input file: missing header, got " << W.Strings[0]
              << std::endl;
    exit(-1);
  }
  if constexpr (binary) {
    if (!strcmp(W.Strings[0], "WeightedAdjacencyGraph")) {
      std::cerr
          << "reading in a weighted graph to binary format, ignoring weights"
          << std::endl;
    }
  }
  bool pattern = false;
  if constexpr (!binary) {
    if (!strcmp(W.Strings[0], "AdjacencyGraph")) {
      std::cerr << "trying reading in a binary graph to weighted format, using "
                   "1 for all weights"
                << std::endl;
      pattern = true;
    }
  }

  uint64_t len = W.m - 1;
  if (len == 0) {
    printf("the file appears to have no data, exiting\n");
    exit(-1);
  }
  uint64_t n = strtoul(W.Strings[1], nullptr, 10);
  uint64_t m = strtoul(W.Strings[2], nullptr, 10);
  uint64_t *offsets = (uint64_t *)malloc(n * sizeof(uint64_t));
  ParallelTools::parallel_for(0, n, [&](size_t i) {
    offsets[i] = strtoul(W.Strings[i + 3], nullptr, 10);
  });
  uint64_t *destinations = (uint64_t *)malloc(m * sizeof(uint64_t));
  ParallelTools::parallel_for(0, m, [&](size_t i) {
    destinations[i] = strtoul(W.Strings[i + 3 + n], nullptr, 10);
  });
  weight_t *weights = nullptr;
  if constexpr (!binary) {
    if (!pattern) {
      weights = (weight_t *)malloc(m * sizeof(weight_t));
    }
    ParallelTools::parallel_for(0, m, [&](size_t i) {
      if (pattern) {
        weights[i] = 1;
      } else {
        if constexpr (std::is_integral_v<weight_t>) {
          weights[i] = strtoul(W.Strings[i + 3 + n + m], nullptr, 10);
        } else {
          weights[i] = strtold(W.Strings[i + 3 + n + m], nullptr);
        }
      }
    });
  }
  W.del();

  if (n == 0 || m == 0) {
    printf("the file says we have no edges or vertices, exiting\n");
    free(offsets);
    free(destinations);
    free(weights);
    exit(-1);
  }

  if (len != n + m + 2 && len != n + 2 * m + 2) {
    std::cout << "n = " << n << " m = " << m << std::endl;
    std::cout << "Bad input file: length = " << len << " n+m+2 = " << n + m + 2
              << std::endl;
    std::cout << "or: length = " << len << " n+2*m+2 = " << n + 2 * m + 2
              << std::endl;
    free(offsets);
    free(destinations);
    free(weights);
    exit(-1);
  }
  uint64_t num_edges = m;
  if (symmetrize) {
    num_edges *= 2;
  }
  std::vector<edge_type> edges_array(num_edges);
  ParallelTools::parallel_for(0, n, [&](size_t i) {
    uint64_t o = offsets[i];
    uint64_t l = ((i == n - 1) ? m : offsets[i + 1]) - offsets[i];
    for (uint64_t j = o; j < o + l; j++) {
      if (i == destinations[j]) {
        std::cerr << "self loop with " << i << "\n";
      }
      if constexpr (binary) {
        edges_array[j] = {i, destinations[j]};
        if (symmetrize) {
          edges_array[j + m] = {destinations[j], i};
        }
      } else {
        edges_array[j] = {i, destinations[j], weights[j]};
        if (symmetrize) {
          edges_array[j + m] = {destinations[j], i, weights[j]};
        }
      }
    }
  });

  ParallelTools::sort(edges_array.begin(), edges_array.end());
  // TODO(wheatman) this stuff could be done in parallel
  if (symmetrize) {
    auto new_end = std::unique(
        edges_array.begin(), edges_array.end(),
        [](auto const &t1, auto const &t2) {
          return std::make_tuple(std::get<0>(t1), std::get<1>(t1)) ==
                 std::make_tuple(std::get<0>(t2), std::get<1>(t2));
        });
    edges_array.erase(new_end, edges_array.end());
  }
  *edge_count = edges_array.size();

  *node_count = n;
  free(offsets);
  free(destinations);
  free(weights);
  return edges_array;
}

template <typename node_t = uint32_t, typename timestamp_t = uint32_t,
          typename weight_t = bool>
auto get_edges_from_file_edges_ts(const std::string &filename,
                                  uint64_t *edge_count, node_t *node_count,
                                  bool symmetrize = true) {
  static constexpr bool binary = std::is_same_v<weight_t, bool>;
  static_assert(std::is_integral_v<weight_t>,
                "io function is only impleented for interger weights\n");
  int64_t length = 0;
  char *S = readStringFromFile(filename.c_str(), &length);
  if (length == 0) {
    printf("file has 0 length, exiting\n");
    exit(-1);
  }
  words W = stringToWords(S, length);
  uint64_t len = W.m;
  if (len == 0) {
    printf("the file appears to have no data, exiting\n");
    exit(-1);
  }
  uint64_t *In = (uint64_t *)malloc(len * sizeof(uint64_t));
  ParallelTools::parallel_for(
      0, len, [&](size_t i) { In[i] = strtoul(W.Strings[i], nullptr, 10); });
  W.del();

  if constexpr (binary) {
    if (len % 3 != 0) {
      std::cout << "len % 3 = " << len % 3 << std::endl;
      free(In);
      exit(-1);
    }
  } else {
    if (len % 4 != 0) {
      std::cout << "len % 4 = " << len % 4 << std::endl;
      free(In);
      exit(-1);
    }
  }
  using edge_t = typename std::conditional<
      binary, std::tuple<node_t, node_t, timestamp_t>,
      std::tuple<node_t, node_t, timestamp_t, weight_t>>::type;
  size_t divisor = 3;
  if constexpr (!binary) {
    divisor = 4;
  }
  size_t m = len / divisor;
  uint64_t mul_factor = 1;
  if (symmetrize) {
    m *= 2;
    mul_factor = 2;
  }
  std::vector<edge_t> edges_array(m);
  ParallelTools::parallel_for(0, len / divisor, [&](size_t i) {
    if constexpr (binary) {
      edges_array[mul_factor * i] = std::make_tuple(
          In[i * divisor], In[i * divisor + 1], In[i * divisor + 2]);
      if (symmetrize) {
        edges_array[2 * i + 1] = std::make_tuple(
            In[i * divisor + 1], In[i * divisor], In[i * divisor + 2]);
      }
    } else {
      edges_array[mul_factor * i] =
          std::make_tuple(In[i * divisor], In[i * divisor + 1],
                          In[i * divisor + 2], In[i * divisor + 3]);
      if (symmetrize) {
        edges_array[2 * i + 1] =
            std::make_tuple(In[i * divisor + 1], In[i * divisor],
                            In[i * divisor + 2], In[i * divisor + 3]);
      }
    }
  });

  ParallelTools::sort(edges_array.begin(), edges_array.end(),
                      [](auto const &t1, auto const &t2) {
                        return std::make_tuple(std::get<2>(t1), std::get<0>(t1),
                                               std::get<1>(t1)) <
                               std::make_tuple(std::get<2>(t2), std::get<0>(t2),
                                               std::get<1>(t2));
                      });
  // TODO(wheatman) this stuff could be done in parallel
  auto new_end = std::unique(edges_array.begin(), edges_array.end());
  edges_array.erase(new_end, edges_array.end());
  *edge_count = edges_array.size();
  node_t max_node = 0;
  for (const auto &edge : edges_array) {
    max_node = std::max(max_node, std::get<0>(edge));
    max_node = std::max(max_node, std::get<1>(edge));
  }

  *node_count = max_node + 1;
  free(In);
  return edges_array;
}

template <typename node_t = uint32_t, typename timestamp_t = uint32_t,
          typename weight_t = bool>
auto get_edges_from_file_adj_ts(const std::string &filename,
                                bool random_timstamps, uint64_t *edge_count,
                                uint32_t *node_count, bool symmetrize = true) {

  auto untimed_edges = get_edges_from_file_adj<node_t, weight_t>(
      filename, edge_count, node_count, symmetrize);

  static constexpr bool binary = std::is_same_v<weight_t, bool>;
  using edge_t = typename std::conditional<
      binary, std::tuple<node_t, node_t, timestamp_t>,
      std::tuple<node_t, node_t, timestamp_t, weight_t>>::type;
  static_assert(std::is_integral_v<weight_t>,
                "io function is only implemented for interger weights\n");

  std::vector<edge_t> edges_array(untimed_edges.size());

  std::uniform_int_distribution<> distrib(0, 100000);
  std::random_device rd;
  std::mt19937 gen(rd());
  for (uint64_t i = 0; i < untimed_edges.size(); i++) {
    timestamp_t ts = 0;
    if (random_timstamps) {
      ts = distrib(gen);
    }
    auto edge = untimed_edges[i];
    if (symmetrize) {
      if (i > 0) {
        if (std::get<0>(edge) == std::get<1>(untimed_edges[i - 1]) &&
            std::get<1>(edge) == std::get<0>(untimed_edges[i - 1])) {
          // symetric edges should have the same timestep
          ts = std::get<2>(edges_array[i - 1]);
        }
      }
    }
    if constexpr (binary) {
      edges_array[i] = {std::get<0>(edge), std::get<1>(edge), ts};
    } else {
      edges_array[i] = {std::get<0>(edge), std::get<1>(edge), ts,
                        std::get<2>(edge)};
    }
  }

  ParallelTools::sort(edges_array.begin(), edges_array.end(),
                      [](auto const &t1, auto const &t2) {
                        return std::make_tuple(std::get<2>(t1), std::get<0>(t1),
                                               std::get<1>(t1)) <
                               std::make_tuple(std::get<2>(t2), std::get<0>(t2),
                                               std::get<1>(t2));
                      });

  return edges_array;
}

template <typename node_t = uint32_t, typename timestamp_t = uint32_t,
          typename weight_t = bool>
auto get_edges_from_file_ts(const std::string &filename, bool random_timstamps,
                            uint64_t *edge_count, uint32_t *node_count,
                            bool symmetrize = true) {
  if (filename.ends_with(std::string("adj"))) {
    return get_edges_from_file_adj_ts<node_t, timestamp_t, weight_t>(
        filename, random_timstamps, edge_count, node_count, symmetrize);
  } else if (filename.ends_with(std::string("edges"))) {
    return get_edges_from_file_edges_ts<node_t, timestamp_t, weight_t>(
        filename, edge_count, node_count, symmetrize);
  }
  std::cout << "file type not implemented\n";
  exit(-1);
}

} // namespace EdgeMapVertexMap
