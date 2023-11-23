#include "mixer.h"

#include "sigmoid.h"

#include <numeric>
#include <utility>
#include <math.h>

Mixer::Mixer(const std::valarray<float>& inputs,
    const std::valarray<float>& extra_inputs,
    const unsigned long long& context, float learning_rate,
    unsigned int extra_input_size) : inputs_(inputs),
    extra_inputs_vec_(extra_inputs), extra_inputs_size_(extra_input_size),/*extra_inputs_(extra_input_size),*/ p_(0.5),
    learning_rate_(learning_rate), context_(context), max_steps_(1), steps_(0),
    context_base_(inputs.size(), extra_inputs_size_)
    {}

ContextData* Mixer::GetContextData() {
  ContextData* data;
  unsigned long long limit = 10000;
  auto it = context_map_.find(context_); 
  if (context_map_.size() >= limit && it == context_map_.end()) {
    data = &context_base_;
    // data = context_map_[0xDEADBEEF].get();
    // if (data == nullptr) {
    //   context_map_[0xDEADBEEF] = std::unique_ptr<ContextData>(
    //       new ContextData(inputs_.size(), extra_inputs_.size()));
    //   data = context_map_[0xDEADBEEF].get();
    // }
  } else {
    if (it != context_map_.end()) {
      data = &it->second;
    } else {
      //auto [it, success] = context_map_.emplace(std::piecewise_construct, std::make_tuple(context_), std::make_tuple(inputs_.size(), extra_inputs_.size()));
      auto [it, success] = context_map_.insert({context_, ContextData(inputs_.size(), extra_inputs_size_)});
      data = &it->second;
    }
  }

  return data;
}

float Mixer::Mix() {
  ContextData* data = GetContextData();
  float p = 0;
  for (int i = 0; i < inputs_.size(); ++i) {
    p += inputs_[i] * data->weights[i];
  }
  p_ = p;
  // for (unsigned int i = 0; i < extra_inputs_.size(); ++i) {
  //   extra_inputs_[i] = extra_inputs_vec_[i];
  // }
  float e = 0;
  for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
    e += extra_inputs_vec_[i] * data->extra_weights[i];
  }
  p_ += e;
  return p_;
}

void Mixer::Perceive(int bit) {
  ContextData* data = GetContextData();
//  float decay = 0.9f / pow(0.0000001f * steps_ + 0.8f, 0.8f);
//  decay *= 1.5f - ((1.0f * data->steps) / max_steps_);

  float decay;
  if ( steps_ < 1000000) {
    decay = 1;
  } else {
    if ( steps_ < 5000000) {
      decay = 0.7;
    } else {
      if ( steps_ < 25000000) {
        decay = 0.3;
      } else {
        decay = 0.2;
      }
    }
  }
  float update = decay * learning_rate_ * (Sigmoid::Logistic(p_) - bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  data->weights -= update * inputs_;
  data->extra_weights -= update * extra_inputs_vec_[std::slice(0,extra_inputs_size_,1)];
//  if (data->steps % 1000 == 0) {
  if ((data->steps & 1023) == 0) {
    data->weights *= 1.0f - 3.0e-6f;
    data->extra_weights *= 1.0f - 3.0e-6f;
  }
}

