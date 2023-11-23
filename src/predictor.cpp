#include "predictor.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

//#include <iostream>

Predictor::Predictor(const std::vector<bool>& vocab) : manager_(),
    sigmoid_(100001), vocab_(vocab) {
//  srand(0xDEADBEEF);
  srand(SEED);

  AddBracket();
  AddFXCM();
  AddPPMD();
  AddWord();
  AddMatch();
  AddDoubleIndirect();
  AddMixers();
  auxiliary_size_ = 2;
}

unsigned long long Predictor::GetNumModels() {
  unsigned long long num = 0;

  // models
  num += bracket_model_->NumOutputs(); // bracket
  num += fxcm_model_->NumOutputs();
  num += direct_models_.size();
  num += match_models_.size();
  num += indirect_ns_models_.size();
  num += indirect_r_models_.size();
  num += byte_model_->NumOutputs();
  num += byte_mixer_->NumOutputs();
  return num;
}

void Predictor::AddMixer(int layer, const unsigned long long& context,
    float learning_rate) {
  if (layer == 0) {
    mixer_0_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_0_.size());
  } else {
    mixer_1_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_1_.size());
  }
}

void Predictor::AddFXCM() {
  fxcm_model_.emplace();
}

void Predictor::AddBracket() {
  bracket_model_.emplace(manager_.bit_context_, 200, 10, 100000, vocab_);
  const Context& context = manager_.AddBracketContext(manager_.bit_context_, 256, 15);
  direct_models_.emplace_back(context.GetContext(), manager_.bit_context_, 30, 0,
      context.Size());
  indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
      manager_.bit_context_, 300, manager_.shared_map_);
}

void Predictor::AddPPMD() {
  byte_model_.emplace(25, 14000, manager_.bit_context_, vocab_);
}

void Predictor::AddWord() {
  float delta = 200;
  std::vector<std::vector<unsigned int>> model_params = {{0}, {0, 1}, {7, 2},
      {7}, {1}, {1, 2}, {1, 2, 3}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {3, 4},
      {1, 2, 4}, {1, 2, 3, 4}, {2, 3, 4}, {2}, {1, 2, 3, 4, 5},
      {1, 2, 3, 4, 5, 6}};
  for (const auto& params : model_params) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
        manager_.bit_context_, delta, manager_.shared_map_);
  }

  std::vector<std::vector<unsigned int>> model_params2 = {{0}, {1}, {7},
      {1, 3}, {1, 2, 3}, {7, 2}};
  for (const auto& params : model_params2) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, 2000000, &(manager_.longest_match_));
    if (params[0] == 1 && params.size() == 1) {
      indirect_r_models_.emplace_back(manager_.run_map_, context.GetContext(),
          manager_.bit_context_, delta, manager_.shared_map_);
    }
  }
}

void Predictor::AddMatch() {
  float delta = 0.5;
  int limit = 200;
  unsigned long long max_size = 2000000;
  std::vector<std::vector<int>> model_params = {{0, 8}, {1, 8}, {2, 8}, {7, 4},
      {11, 3}, {13, 2}, {15, 2}, {17, 2}, {20, 1}, {25, 1}};

  for (const auto& params : model_params) {
    const Context& context = manager_.AddContextHashContext(manager_.bit_context_,params[0], params[1]);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, limit, delta, std::min(max_size, context.Size()),
        &(manager_.longest_match_));
  }
}

void Predictor::AddDoubleIndirect() {
  float delta = 400;
  std::vector<std::vector<unsigned int>> model_params = {{1, 8, 1, 8},
      {2, 8, 1, 8}, {1, 8, 2, 8}, {2, 8, 2, 8}, {1, 8, 3, 8}, {3, 8, 1, 8},
      {4, 6, 4, 8}, {5, 5, 5, 5}, {1, 8, 4, 8}, {1, 8, 5, 6}, {6, 4, 6, 4}};
  for (const auto& params : model_params) {
    const Context& context = manager_.AddIndirectHashContext(manager_.bit_context_, params[0], params[1],
        params[2], params[3]);
    indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
        manager_.bit_context_, delta, manager_.shared_map_);
  }
}

void Predictor::AddMixers() {
  unsigned int vocab_size = 0;
  for (unsigned int i = 0; i < vocab_.size(); ++i) {
    if (vocab_[i]) ++vocab_size;
  }
  byte_mixer_.emplace(1, manager_.bit_context_, vocab_,
      vocab_size, new Lstm(vocab_size, vocab_size, 200, 1, 128, 0.03, 10));

  for (int i = 0; i < 2; ++i) {
    layers_.emplace_back(sigmoid_,
        1.0e-4);
  }

  unsigned long long input_size = GetNumModels();
  std::cout << "num models " << input_size << "\n";
  layers_[0].SetNumModels(input_size);
  std::vector<std::vector<double>> model_params = {{0, 8, 0.005},
      {0, 8, 0.0005}, {1, 8, 0.005}, {1, 8, 0.0005}, {2, 4, 0.005},
      {3, 2, 0.002}};
  for (const auto& params : model_params) {
    const Context& context = manager_.AddContextHashContext(manager_.bit_context_, params[0], params[1]);
    const BitContext& bit_context = manager_.AddBitContext(manager_.long_bit_context_,
        context.GetContext(), context.Size());
    AddMixer(0, bit_context.GetContext(), params[2]);
  }

  AddMixer(0, manager_.recent_bytes_[2], 0.002);
  AddMixer(0, manager_.zero_context_, 0.00005);
  AddMixer(0, manager_.line_break_, 0.0007);
  AddMixer(0, manager_.longest_match_, 0.0005);
  AddMixer(0, manager_.wrt_context_, 0.002);
  AddMixer(0, manager_.auxiliary_context_, 0.0005);

  std::vector<int> map(256, 0);
  for (int i = 0; i < 256; ++i) {
    map[i] = (i < 1) + (i < 32) + (i < 64) + (i < 128) + (i < 255) +
      (i < 142) + (i < 138) + (i < 140) + (i < 137) + (i < 97);
  }
  const Context& interval1 = manager_.AddIntervalContext(manager_.bit_context_, map, 8);
  AddMixer(0, interval1.GetContext(), 0.001);

  for (int i = 0; i < 256; ++i) {
    map[i] = (i < 41) + (i < 92) + (i < 124) + (i < 58) +
        (i < 11) + (i < 46) + (i < 36) + (i < 47) +
        (i < 64) + (i < 4) + (i < 61) + (i < 97) +
        (i < 125) + (i < 45) + (i < 48);
  }
  const Context& interval2 = manager_.AddIntervalContext(manager_.bit_context_, map, 8);
  AddMixer(0, interval2.GetContext(), 0.001);

  for (int i = 0; i < 256; ++i) map[i] = 0;
  for (int i = 'a'; i <= 'z'; ++i) map[i] = 1;
  for (int i = '0'; i <= '9'; ++i) map[i] = 1;
  for (int i = 0x80; i < 256; ++i) map[i] = 1;
  const Context& interval3 = manager_.AddIntervalContext(manager_.bit_context_, map, 7);
  AddMixer(0, interval3.GetContext(), 0.001);
  const BitContext& bit_context5 = manager_.AddBitContext(manager_.long_bit_context_,
      interval3.GetContext(), interval3.Size());
  AddMixer(0, bit_context5.GetContext(), 0.005);

  map = {
    2, 3, 1, 3, 3, 0, 1, 2, 3, 3, 0, 0, 1, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 
    3, 2, 0, 2, 1, 3, 2, 1, 3, 3, 3, 3, 2, 3, 0, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2,
    2, 2, 0, 0, 2, 3, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 3, 2, 0, 2, 3,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const Context& interval4 = manager_.AddIntervalContext(manager_.bit_context_, map, 10);
  AddMixer(0, interval4.GetContext(), 0.001);
  const Context& interval5 = manager_.AddIntervalContext(manager_.bit_context_, map, 15);
  AddMixer(0, interval5.GetContext(), 0.001);
  const Context& interval8 = manager_.AddIntervalContext(manager_.bit_context_, map, 7);
  const BitContext& bit_context4 = manager_.AddBitContext(manager_.long_bit_context_,
      interval8.GetContext(), interval8.Size());
  AddMixer(0, bit_context4.GetContext(), 0.005);

  map = {
    0, 0, 2, 0, 5, 6, 0, 6, 0, 2, 0, 4, 3, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    2, 4, 1, 4, 4, 7, 4, 7, 3, 7, 2, 2, 3, 5, 3, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 5, 3, 3, 5, 5, 
    0, 5, 5, 7, 5, 0, 1, 5, 4, 5, 0, 0, 6, 0, 7, 1, 
    3, 3, 7, 4, 5, 5, 7, 0, 2, 2, 5, 4, 4, 7, 4, 6, 
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  const Context& interval6 = manager_.AddIntervalContext(manager_.bit_context_, map, 9);
  AddMixer(0, interval6.GetContext(), 0.001);
  const Context& interval7 = manager_.AddIntervalHashContext(manager_.bit_context_, map, 8, 7, 2);
  AddMixer(0, interval7.GetContext(), 0.001);
  const Context& interval9 = manager_.AddIntervalContext(manager_.bit_context_, map, 7);
  const BitContext& bit_context6 = manager_.AddBitContext(manager_.long_bit_context_,
      interval9.GetContext(), interval9.Size());
  AddMixer(0, bit_context6.GetContext(), 0.005);

  const BitContext& bit_context1 = manager_.AddBitContext(manager_.long_bit_context_,
      manager_.recent_bytes_[1], 256);
  AddMixer(0, bit_context1.GetContext(), 0.005);

  const Context& combined1 = manager_.AddCombinedContext(manager_.recent_bytes_[1],
      manager_.recent_bytes_[0], 256, 256);
  AddMixer(0, combined1.GetContext(), 0.005);

  const Context& combined2 = manager_.AddCombinedContext(manager_.recent_bytes_[2],
      manager_.recent_bytes_[1], 256, 256);
  AddMixer(0, combined2.GetContext(), 0.003);

  input_size = mixer_0_.size() + auxiliary_size_;
  layers_[1].SetNumModels(input_size);
  AddMixer(1, manager_.zero_context_, 0.0003);

  layers_[0].SetExtraInputSize(mixer_0_.size());
}

float Predictor::Predict() {
  unsigned int input_index = 0;
  auto bracket_model_output = bracket_model_->Predict()[0];
  layers_[0].SetInput(input_index++, bracket_model_output);

  const auto& fxcm_model_outputs = fxcm_model_->Predict();
  for (unsigned int j = 0; j < fxcm_model_outputs.size(); ++j) {
    layers_[0].SetInput(input_index, fxcm_model_outputs[j]);
    ++input_index;
  }
  auto fxcm_model_index = input_index - 1;

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    const std::valarray<float>& outputs = direct_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }

  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    const std::valarray<float>& outputs = match_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
 
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_ns_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
 
 for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_r_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
  layers_[0].SetInput(input_index++, byte_model_->Predict()[0]);

  float byte_mixer_override = -1;
  auto byte_mixer_output = byte_mixer_->Predict()[0];
  if (byte_mixer_output == 0 || byte_mixer_output == 1) byte_mixer_override = byte_mixer_output;
  layers_[0].SetInput(input_index++, byte_mixer_output);
  auto byte_mixer_index = input_index - 1;

  float auxiliary_average = Sigmoid::Logistic(layers_[0].Inputs()[fxcm_model_index]) + Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
  auxiliary_average /= auxiliary_size_;
  manager_.auxiliary_context_ = auxiliary_average * 15;

  for (unsigned int i = 0; i < mixer_0_.size(); ++i) {
    float p = mixer_0_[i].Mix();
    layers_[0].SetExtraInput(i, p);
    layers_[1].SetStretchedInput(i, p);
  }
  layers_[1].SetStretchedInput(mixer_0_.size(), layers_[0].Inputs()[fxcm_model_index]);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[byte_mixer_index]);

  float p = Sigmoid::Logistic(mixer_1_[0].Mix());
  p = sse_.Predict(p);
  if (byte_mixer_override >= 0) {
    return byte_mixer_override;
  }
  return p;
}

void Predictor::Perceive(int bit) {
  bracket_model_->Perceive(bit);
  fxcm_model_->Perceive(bit);
    
  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }


  byte_model_->Perceive(bit);

  byte_mixer_->Perceive(bit);

  for (auto& mixer: mixer_0_) {
    mixer.Perceive(bit);
  }
  for (auto& mixer: mixer_1_) {
    mixer.Perceive(bit);
  }

  sse_.Perceive(bit);

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;

  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();
    fxcm_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }

    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }


    byte_model_->ByteUpdate();


    const std::valarray<float>& p = byte_model_->BytePredict();
    for (unsigned int j = 0; j < 256; ++j) {
      byte_mixer_->SetInput(j, p[j]);
    }
    
    byte_mixer_->ByteUpdate();
 
    manager_.bit_context_ = 1;
  }
}

void Predictor::Pretrain(int bit) {
  bracket_model_->Predict();
  fxcm_model_->Predict();
    
  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Predict();
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Predict();
  }


  bracket_model_->Perceive(bit);
  fxcm_model_->Perceive(bit);
    
  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }


  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;
  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();
    fxcm_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
    manager_.bit_context_ = 1;
  }
}

