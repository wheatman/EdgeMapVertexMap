#pragma once

#include "VertexSubset.hpp"

namespace EdgeMapVertexMap {

template <class F, class node_t>
VertexSubset<node_t> vertexMap(VertexSubset<node_t> &vs, F f,
                               bool output = true) {

  if (output) {
    VertexSubset<node_t> output_vs = vs.empty_version_for_insert();
    vs.map([&](node_t val) {
      if (f(val) == 1) {
        output_vs.insert(val);
      }
    });
    return output_vs;
  } else {
    // output is empty
    VertexSubset<node_t> null_vs = VertexSubset<node_t>();
    vs.map([&](node_t val) { f(val); });
    return null_vs;
  }
}

} // namespace EdgeMapVertexMap
