#ifndef BRACKET_H
#define BRACKET_H

#include "byte-model.h"

#include "../ds/emhash_map.hpp"
#include <vector>
#include <utility>

class Bracket : public ByteModel {
 public:
  Bracket(const unsigned int& bit_context, int distance_limit, int stack_limit,
      int stats_limit, const std::vector<bool>& vocab);
  void ByteUpdate();

 private:
  emhash6::HashMap<unsigned char, unsigned char> brackets_;
  unsigned int distance_limit_, stack_limit_, stats_limit_;
  std::vector<unsigned int> active_, distance_;
  const unsigned int& byte_;
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> stats_;
};

#endif

