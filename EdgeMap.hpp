#pragma once

#include "VertexSubset.hpp"
#include <type_traits>

template <class F, class node_t, bool output, class value_t = bool>
struct MAP_SPARSE {
private:
  const node_t src;
  F &f;
  VertexSubset<node_t> &output_vs;
  static constexpr bool binary = std::is_same<value_t, bool>::value;

public:
  static constexpr bool no_early_exit = true;

  MAP_SPARSE(const node_t src, F &f, VertexSubset<node_t> &output_vs)
      : src(src), f(f), output_vs(output_vs) {}

  inline bool update([[maybe_unused]] node_t source, node_t dest,
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

template <class F, class Graph, class extra_data_t, class node_t, bool output,
          class value_t = bool>
struct EDGE_MAP_SPARSE {

private:
  const Graph &G;
  VertexSubset<node_t> &output_vs;
  F f;
  const extra_data_t &d;

public:
  EDGE_MAP_SPARSE(const Graph &G_, VertexSubset<node_t> &output_vs_, F f_,
                  const extra_data_t &d_)
      : G(G_), output_vs(output_vs_), f(f_), d(d_) {}
  inline bool update(node_t val) {
    struct MAP_SPARSE<F, node_t, output, value_t> ms(val, f, output_vs);
    // openmp is doing wrong things with nested parallelism so disable the
    // inner parallel portion
#if OPENMP == 1
    G.template map_neighbors<MAP_SPARSE<F, node_t, output, value_t>>(val, ms, d,
                                                                     false);
#else
    G.template map_neighbors<MAP_SPARSE<F, node_t, output, value_t>>(val, ms, d,
                                                                     true);
#endif
    return false;
  }
};

template <class F, class Graph, class extra_data_t, class node_t, bool output,
          class value_t = bool>
VertexSubset<node_t> EdgeMapSparse(const Graph &G,
                                   const VertexSubset<node_t> &vertext_subset,
                                   F f, const extra_data_t &d) {
  VertexSubset vs = (vertext_subset.sparse())
                        ? vertext_subset
                        : vertext_subset.convert_to_sparse();
  if constexpr (output) {
    VertexSubset<node_t> output_vs = VertexSubset(vs, false);
    struct EDGE_MAP_SPARSE<F, Graph, extra_data_t, node_t, output, value_t> v(
        G, output_vs, f, d);
    vs.map_sparse(v);
    output_vs.finalize();
    if (!vertext_subset.sparse()) {
      vs.del();
    }
    return output_vs;
  } else {
    VertexSubset<node_t> null_vs = VertexSubset<node_t>();
    struct EDGE_MAP_SPARSE<F, Graph, extra_data_t, node_t, output, value_t> v(
        G, null_vs, f, d);
    vs.map_sparse(v);
    if (!vertext_subset.sparse()) {
      vs.del();
    }
    return null_vs;
  }
}

template <class F, class node_t, bool output, bool vs_all, class value_t = bool>
struct MAP_DENSE {
private:
  F &f;
  const VertexSubset<node_t> &vs;
  VertexSubset<node_t> &output_vs;

public:
  static constexpr bool no_early_exit = false;

  MAP_DENSE(F &f, const VertexSubset<node_t> &vs,
            VertexSubset<node_t> &output_vs)
      : f(f), vs(vs), output_vs(output_vs) {}

  inline bool update(node_t dest, node_t source,
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

template <class F, class Graph, class extra_data_t, class node_t>
void map_range(const Graph &G, F f, node_t node_start, node_t node_end,
               [[maybe_unused]] const extra_data_t &d) {
  constexpr bool has_map_range = requires(const Graph &g) {
    g.template map_range<F, node_t>(f, node_start, node_end, d);
  };
  if constexpr (has_map_range) {
    G.template map_range<F, node_t>(f, node_start, node_end, d);
  } else {
    for (node_t i = node_start; i < node_end; i++) {
      G.map_neighbors(i, f, d, false);
    }
  }
}

template <class F, class Graph, class extra_data_t, class node_t, bool output,
          bool vs_all, class value_t = bool>
VertexSubset<node_t> EdgeMapDense(const Graph &G,
                                  const VertexSubset<node_t> &vertext_subset,
                                  F f, const extra_data_t &d) {
  VertexSubset vs = (vertext_subset.sparse())
                        ? vertext_subset.convert_to_dense()
                        : vertext_subset;
  if constexpr (output) {
    VertexSubset output_vs = VertexSubset(vs, false);
    // needs a grainsize of at least 512
    // so writes to the bitvector storing the next vertex set are going to
    // different cache lines
    node_t num_nodes = G.num_nodes();
    parallel_for(uint64_t i = 0; i < num_nodes; i += 512) {
      uint64_t end = std::min(i + 512, (uint64_t)num_nodes);
      if constexpr (F::cond_true) {
        MAP_DENSE<F, node_t, output, vs_all, value_t> md(f, vs, output_vs);
        map_range<MAP_DENSE<F, node_t, output, vs_all, value_t>, Graph,
                  extra_data_t, node_t>(G, md, i, end, d);
      } else {
        for (uint64_t j = i; j < end; j++) {
          if (f.cond(j) == 1) {
            MAP_DENSE<F, node_t, output, vs_all, value_t> md(f, vs, output_vs);
            G.template map_neighbors<
                MAP_DENSE<F, node_t, output, vs_all, value_t>>(j, md, d, false);
          }
        }
      }
    }
    if (vertext_subset.sparse()) {
      vs.del();
    }
    return output_vs;
  } else {
    VertexSubset<node_t> null_vs = VertexSubset<node_t>();
    // needs a grainsize of at least 512
    // so writes to the bitvector storing the next vertex set are going to
    // different cache lines
    parallel_for(uint64_t i = 0; i < G.num_nodes(); i += 512) {

      uint64_t end = std::min(i + 512, (uint64_t)G.num_nodes());
      if constexpr (F::cond_true) {
        MAP_DENSE<F, node_t, output, vs_all, value_t> md(f, vs, null_vs);
        map_range<MAP_DENSE<F, node_t, output, vs_all, value_t>, Graph,
                  extra_data_t, node_t>(G, md, i, end, d);
      } else {
        for (uint64_t j = i; j < end; j++) {
          if (f.cond(j) == 1) {
            MAP_DENSE<F, node_t, output, vs_all, value_t> md(f, vs, null_vs);
            G.template map_neighbors<
                MAP_DENSE<F, node_t, output, vs_all, value_t>>(j, md, d, false);
          }
        }
      }
    }
    if (vertext_subset.sparse()) {
      vs.del();
    }
    return null_vs;
  }
}

template <class F, class Graph, class extra_data_t, class node_t,
          class value_t = bool>
VertexSubset<node_t> edgeMap(const Graph &G, VertexSubset<node_t> &vs, F f,
                             const extra_data_t &d, bool output = true,
                             uint32_t threshold = 20) {
  if (output) {
    if (vs.complete()) {
      if (G.num_nodes() / threshold <= vs.get_n()) {
        auto out =
            EdgeMapDense<F, Graph, extra_data_t, node_t, true, true, value_t>(
                G, vs, f, d);
        return out;
      } else {
        auto out = EdgeMapSparse<F, Graph, extra_data_t, node_t, true, value_t>(
            G, vs, f, d);
        return out;
      }
    } else {
      if (G.num_nodes() / threshold <= vs.get_n()) {
        auto out =
            EdgeMapDense<F, Graph, extra_data_t, node_t, true, false, value_t>(
                G, vs, f, d);
        return out;
      } else {
        auto out = EdgeMapSparse<F, Graph, extra_data_t, node_t, true, value_t>(
            G, vs, f, d);
        return out;
      }
    }
  } else {
    if (vs.complete()) {
      if (G.num_nodes() / threshold <= vs.get_n()) {
        auto out =
            EdgeMapDense<F, Graph, extra_data_t, node_t, false, true, value_t>(
                G, vs, f, d);
        return out;
      } else {
        auto out =
            EdgeMapSparse<F, Graph, extra_data_t, node_t, false, value_t>(G, vs,
                                                                          f, d);
        return out;
      }
    } else {
      if (G.num_nodes() / threshold <= vs.get_n()) {
        auto out =
            EdgeMapDense<F, Graph, extra_data_t, node_t, false, false, value_t>(
                G, vs, f, d);
        return out;
      } else {
        auto out =
            EdgeMapSparse<F, Graph, extra_data_t, node_t, false, value_t>(G, vs,
                                                                          f, d);
        return out;
      }
    }
  }
}