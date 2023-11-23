#ifndef INDIRECT_H
#define INDIRECT_H

#include "model.h"
#include "../states/state.h"

#include <vector>
#include <array>
#include <stdlib.h>

template<typename StateType>
class Indirect : public Model {
 public:
  Indirect(const StateType& state,
      const unsigned long long& byte_context,
      const unsigned int& bit_context, float delta,
      std::vector<unsigned char>& map);
  const std::valarray<float>& Predict() const;
  void Perceive(int bit);
  void ByteUpdate();

 private:
  const unsigned long long& byte_context_;
  const unsigned int& bit_context_;
  unsigned long long map_index_, map_offset_;
  float divisor_;
  const StateType& state_;
  std::vector<unsigned char>& map_;
  std::array<float, 256> predictions_;
};


template<typename StateType>
Indirect<StateType>::Indirect(const StateType& state,
    const unsigned long long& byte_context,
    const unsigned int& bit_context, float delta,
    std::vector<unsigned char>& map) :  byte_context_(byte_context),
    bit_context_(bit_context), map_index_(0), map_offset_(0),
    divisor_(1.0 / delta), state_(state), map_(map) {
  map_offset_ = rand() % (map_.size() - 257);
  for (int i = 0; i < 256; ++i) {
    predictions_[i] = state_.InitProbability(i);
  }
}

template<typename StateType>
const std::valarray<float>& Indirect<StateType>::Predict() const {
  outputs_[0] = predictions_[map_[map_index_ + bit_context_]];
  return outputs_;
}

template<typename StateType>
void Indirect<StateType>::Perceive(int bit) {
  map_index_ += bit_context_;
  int state = map_[map_index_];
  predictions_[state] += (bit - predictions_[state]) * divisor_;
  map_[map_index_] = state_.Next(state, bit);
  map_index_ -= bit_context_;
}

template<typename StateType>
void Indirect<StateType>::ByteUpdate() {
  map_index_ = (257 * byte_context_ + map_offset_) % (map_.size() - 257);
}


#endif

