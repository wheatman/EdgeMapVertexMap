#pragma once

#include "VertexSubset.hpp"

namespace EdgeMapVertexMap {
template <class F, class node_t, bool output> struct VERTEX_MAP {
private:
  const VertexSubset<node_t> &vs;
  VertexSubset<node_t> &output_vs;
  F f;

public:
  VERTEX_MAP(const VertexSubset<node_t> &vs_, VertexSubset<node_t> &output_vs_,
             F f_)
      : vs(vs_), output_vs(output_vs_), f(f_) {}
  inline bool operator()(node_t val) {
    if constexpr (output) {
      if (f(val) == 1) {
        output_vs.insert(val);
      }
    } else {
      f(val);
    }
    return false;
  }
};

template <class F, class node_t>
VertexSubset<node_t> vertexMap(VertexSubset<node_t> &vs, F f,
                               bool output = true) {

  if (output) {
    VertexSubset<node_t> output_vs = VertexSubset<node_t>(vs, false);
    struct VERTEX_MAP<F, node_t, true> v(vs, output_vs, f);
    vs.map(v);
    return output_vs;
  } else {
    // output is empty
    VertexSubset<node_t> null_vs = VertexSubset<node_t>();
    struct VERTEX_MAP<F, node_t, false> v(vs, null_vs, f);
    vs.map(v);
    return null_vs;
  }
}

} // namespace EdgeMapVertexMap
