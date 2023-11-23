#ifndef BRACKET_CONTEXT_H
#define BRACKET_CONTEXT_H

#include "context.h"

#include "../ds/emhash_map.hpp"
#include <vector>

class BracketContext : public Context {
 public:
  BracketContext(const unsigned int& bit_context, int distance_limit,
      int stack_limit);
  void Update();
  bool IsEqual(Context* c) const;

 private:
  const unsigned int& byte_;
  emhash6::HashMap<unsigned char, unsigned char> brackets_;
  unsigned int distance_limit_, stack_limit_;
  std::vector<unsigned int> active_, distance_;
};

#endif

