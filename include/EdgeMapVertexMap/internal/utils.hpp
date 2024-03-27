#pragma once
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
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

template <class T>
void write_array_to_file(std::string_view filename, const T *data,
                         uint64_t length) {
  std::ofstream myfile;
  myfile.open(filename.data());
  for (uint64_t i = 0; i < length; i++) {
    if constexpr (std::is_unsigned_v<T>) {
      if (data[i] == std::numeric_limits<T>::max()) {
        myfile << -1 << std::endl;
        continue;
      }
    }
    myfile << data[i] << std::endl;
  }
  myfile.close();
}

} // namespace EdgeMapVertexMap