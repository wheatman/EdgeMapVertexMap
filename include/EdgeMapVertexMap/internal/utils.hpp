#pragma once
#include <cstdint>
#include <limits>
#include <random>
#include <sys/time.h>
#include <type_traits>

namespace EdgeMapVertexMap {

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

template <typename G>
concept GraphConcept = requires(G g) {
  typename G::node_t;
  typename G::weight_t;
  typename G::extra_data_t;
};

} // namespace EdgeMapVertexMap