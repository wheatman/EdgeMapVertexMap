#pragma once

#include "VertexSubset.hpp"
#include "utils.hpp"

#include <type_traits>

namespace EdgeMapVertexMap {

template <class Graph, class argument = std::nullptr_t>
auto getExtraData(const Graph &G, argument arg = argument()) {
  constexpr bool has_getExtraData = requires(const Graph &g) {
    g.getExtraData(arg);
  };
  if constexpr (has_getExtraData) {
    return G.getExtraData(arg);
  } else {
    return nullptr;
  }
}

template <class Graph, class F>
auto map_out_neighbors(const Graph &G, typename Graph::node_t node, F f,
                       const typename Graph::extra_data_t &d, bool parallel) {
  constexpr bool has_map_out_neighbors = requires(const Graph &g) {
    g.map_out_neighbors(node, f, d, parallel);
  };
  constexpr bool has_map_in_neighbors = requires(const Graph &g) {
    g.map_in_neighbors(node, f, d, parallel);
  };
  if constexpr (has_map_out_neighbors) {
    static_assert(has_map_in_neighbors,
                  "if the graph is directed, symbolized by having "
                  "map_out_neighbors, it must also have map_in_neighbors");
    return G.map_out_neighbors(node, f, d, parallel);
  } else {
    return G.map_neighbors(node, f, d, parallel);
  }
}

template <class F> class flip_args {
public:
  static constexpr bool no_early_exit = F::no_early_exit;
  F &f;
  flip_args(F &f_) : f(f_){};
  auto operator()(auto a, auto b, auto... args) { return f(b, a, args...); }
};

template <class Graph, class F>
auto map_in_neighbors(const Graph &G, typename Graph::node_t node, F f,
                      const typename Graph::extra_data_t &d, bool parallel) {
  constexpr bool has_map_out_neighbors = requires(const Graph &g) {
    g.map_out_neighbors(node, f, d, parallel);
  };
  constexpr bool has_map_in_neighbors = requires(const Graph &g) {
    g.map_in_neighbors(node, f, d, parallel);
  };
  if constexpr (has_map_in_neighbors) {
    static_assert(has_map_out_neighbors,
                  "if the graph is directed, symbolized by having "
                  "map_in_neighbors, it must also have map_out_neighbors");
    return G.map_in_neighbors(node, f, d, parallel);
  } else {
    auto f2 = flip_args(f);
    return G.map_neighbors(node, f2, d, parallel);
  }
}

template <bool output, class Graph, class F> struct MAP_SPARSE {
private:
  using node_t = typename Graph::node_t;
  using value_t = typename Graph::weight_t;
  const node_t src;
  F &f;
  VertexSubset<node_t> &output_vs;

public:
  static constexpr bool no_early_exit = true;

  MAP_SPARSE(const node_t src_, F &f_, VertexSubset<node_t> &output_vs_)
      : src(src_), f(f_), output_vs(output_vs_) {}

  inline bool operator()([[maybe_unused]] node_t source, node_t dest,
                         [[maybe_unused]] value_t val = {}) {
    constexpr bool no_vals =
        std::is_invocable_v<decltype(&F::update), F &, node_t, node_t>;
    if (f.cond(dest) == 1) {
      if constexpr (output) {
        bool r;
        if constexpr (no_vals) {
          r = f.updateAtomic(src, dest);
        } else {
          r = f.updateAtomic(src, dest, val);
        }
        if (r) {
          output_vs.insert_sparse(dest);
        }
      } else {
        if constexpr (no_vals) {
          f.updateAtomic(src, dest);
        } else {
          f.updateAtomic(src, dest, val);
        }
      }
    }
    return false;
  }
};

template <bool output, class Graph, class F> struct EDGE_MAP_SPARSE {

private:
  using node_t = typename Graph::node_t;
  using value_t = typename Graph::weight_t;
  using extra_data_t = typename Graph::extra_data_t;
  const Graph &G;
  VertexSubset<node_t> &output_vs;
  F f;
  const extra_data_t &d;

public:
  EDGE_MAP_SPARSE(const Graph &G_, VertexSubset<node_t> &output_vs_, F f_,
                  const extra_data_t &d_)
      : G(G_), output_vs(output_vs_), f(f_), d(d_) {}
  inline bool operator()(node_t val) {
    struct MAP_SPARSE<output, Graph, F> ms(val, f, output_vs);
    bool parallel = true;
#if OPENMP == 1
    // openmp is doing wrong things with nested parallelism so disable the
    // inner parallel portion
    parallel = false;

#else
    map_out_neighbors(G, val, ms, d, parallel);
#endif
    return false;
  }
};

template <bool output, class Graph, class F>
VertexSubset<typename Graph::node_t>
EdgeMapSparse(const Graph &G,
              const VertexSubset<typename Graph::node_t> &vertext_subset, F f,
              const typename Graph::extra_data_t &d) {
  VertexSubset vs = (vertext_subset.sparse())
                        ? vertext_subset
                        : vertext_subset.convert_to_sparse();
  if constexpr (output) {
    VertexSubset<typename Graph::node_t> output_vs =
        vs.empty_version_for_insert();
    struct EDGE_MAP_SPARSE<output, Graph, F> v(G, output_vs, f, d);
    vs.map_sparse(v);
    if (!vertext_subset.sparse()) {
      vs.del();
    }
    return output_vs;
  } else {
    VertexSubset<typename Graph::node_t> null_vs{};
    struct EDGE_MAP_SPARSE<output, Graph, F> v(G, null_vs, f, d);
    vs.map_sparse(v);
    if (!vertext_subset.sparse()) {
      vs.del();
    }
    return null_vs;
  }
}

template <bool output, bool vs_all, class Graph, class F> struct MAP_DENSE {
private:
  using node_t = typename Graph::node_t;
  using value_t = typename Graph::weight_t;
  F &f;
  const VertexSubset<node_t> &vs;
  VertexSubset<node_t> &output_vs;

public:
  static constexpr bool no_early_exit = F::cond_true;

  MAP_DENSE(F &f_, const VertexSubset<node_t> &vs_,
            VertexSubset<node_t> &output_vs_)
      : f(f_), vs(vs_), output_vs(output_vs_) {}

  inline bool operator()(node_t source, node_t dest,
                         [[maybe_unused]] value_t val = {}) {
    constexpr bool no_vals =
        std::is_invocable_v<decltype(&F::update), F &, node_t, node_t>;

    bool has = true;
    if constexpr (!vs_all) {
      has = vs.has_dense_no_all(source);
    }
    if (has) {
      bool r;
      if constexpr (no_vals) {
        r = f.update(source, dest);
      } else {
        r = f.update(source, dest, val);
      }

      if constexpr (output) {
        if (r) {
          output_vs.insert_dense(dest);
        }
      }
      if constexpr (!F::cond_true) {
        if (f.cond(dest) == 0) {
          return true;
        }
      }
    }
    return false;
  }
};

template <class Graph, class F>
void map_range_in(const Graph &G, F f, typename Graph::node_t node_start,
                  typename Graph::node_t node_end,
                  [[maybe_unused]] const typename Graph::extra_data_t &d) {
  constexpr bool has_map_range_in = requires(const Graph &g) {
    g.map_range_in(f, node_start, node_end, d);
  };
  constexpr bool has_map_range = requires(const Graph &g) {
    g.map_range(f, node_start, node_end, d);
  };
  if constexpr (has_map_range_in) {
    G.map_range_in(f, node_start, node_end, d);
  } else if constexpr (has_map_range) {
    auto f2 = flip_args(f);
    G.map_range(f2, node_start, node_end, d);
  } else {
    for (typename Graph::node_t i = node_start; i < node_end; i++) {
      map_in_neighbors(G, i, f, d, false);
    }
  }
}

template <bool output, bool vs_all, class Graph, class F>
VertexSubset<typename Graph::node_t>
EdgeMapDense(const Graph &G,
             const VertexSubset<typename Graph::node_t> &vertext_subset, F f,
             const typename Graph::extra_data_t &d) {
  VertexSubset vs = (vertext_subset.sparse())
                        ? vertext_subset.convert_to_dense()
                        : vertext_subset;
  if constexpr (output) {
    VertexSubset output_vs = vs.empty_version_for_insert();
    // needs a grainsize of at least 512
    // so writes to the bitvector storing the next vertex set are going to
    // different cache lines
    typename Graph::node_t num_nodes = G.num_nodes();
    ParallelTools::parallel_for(0, num_nodes, 512, [&](size_t i) {
      uint64_t end = std::min(i + 512, (uint64_t)num_nodes);
      if constexpr (F::cond_true) {
        MAP_DENSE<output, vs_all, Graph, F> md(f, vs, output_vs);
        map_range_in(G, md, i, end, d);
      } else {
        for (uint64_t j = i; j < end; j++) {
          if (f.cond(j) == 1) {
            MAP_DENSE<output, vs_all, Graph, F> md(f, vs, output_vs);
            map_in_neighbors(G, j, md, d, false);
          }
        }
      }
    });
    if (vertext_subset.sparse()) {
      vs.del();
    }
    return output_vs;
  } else {
    VertexSubset<typename Graph::node_t> null_vs{};
    // needs a grainsize of at least 512
    // so writes to the bitvector storing the next vertex set are going to
    // different cache lines
    ParallelTools::parallel_for(0, G.num_nodes(), 512, [&](size_t i) {
      uint64_t end = std::min(i + 512, (uint64_t)G.num_nodes());
      if constexpr (F::cond_true) {
        MAP_DENSE<output, vs_all, Graph, F> md(f, vs, null_vs);
        map_range_in(G, md, i, end, d);
      } else {
        for (uint64_t j = i; j < end; j++) {
          if (f.cond(j) == 1) {
            MAP_DENSE<output, vs_all, Graph, F> md(f, vs, null_vs);
            map_in_neighbors(G, j, md, d, false);
          }
        }
      }
    });
    if (vertext_subset.sparse()) {
      vs.del();
    }
    return null_vs;
  }
}

template <class Graph>
bool run_dense(const Graph &G, VertexSubset<typename Graph::node_t> &vs,
               const typename Graph::extra_data_t &d, uint32_t threshold) {
  constexpr bool has_num_edges = requires(const Graph &g) { g.num_edges(); };
  constexpr bool has_get_degree = requires(const Graph &g) {
    g.get_degree(typename Graph::node_t(), d);
  };
  if constexpr (has_num_edges && has_get_degree) {
    return G.num_edges() / threshold <= vs.get_n() + vs.get_out_degree(G, d) ||
           (!vs.sparse() && vs.get_n() > G.num_nodes() / 10);
  } else {
    return G.num_nodes() / threshold <= vs.get_n();
  }
}

template <GraphConcept Graph, class F>
VertexSubset<typename Graph::node_t>
edgeMap(const Graph &G, VertexSubset<typename Graph::node_t> &vs, F f,
        const typename Graph::extra_data_t &d, bool output = true,
        uint32_t threshold = 20) {
  if (output) {
    if (vs.complete()) {
      // if complete always run in dense mode
      auto out = EdgeMapDense<true, true>(G, vs, f, d);
      return out;

    } else {
      if (run_dense(G, vs, d, threshold)) {
        auto out = EdgeMapDense<true, false>(G, vs, f, d);
        return out;
      } else {
        auto out = EdgeMapSparse<true>(G, vs, f, d);
        return out;
      }
    }
  } else {
    if (vs.complete()) {
      // if complete always run in dense mode
      auto out = EdgeMapDense<false, true>(G, vs, f, d);
      return out;
    } else {
      if (run_dense(G, vs, d, threshold)) {
        auto out = EdgeMapDense<false, false>(G, vs, f, d);
        return out;
      } else {
        auto out = EdgeMapSparse<false>(G, vs, f, d);
        return out;
      }
    }
  }
}
} // namespace EdgeMapVertexMap
