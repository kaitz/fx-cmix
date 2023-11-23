#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>
#include <vector>
#include <memory>
#include <string>

#include "lstm-layer.h"

#include "../ds/emhash_set.hpp"
#include "../ds/SmallVector.h"

class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  ~Lstm();
  std::valarray<float>& Perceive(unsigned int input);
  std::valarray<float>& Predict(unsigned int input);
  void SetInput(const std::valarray<float>& input);
  //void SaveToDisk(const std::string& path);
  //void LoadFromDisk(const std::string& path);

 private:
  llvm::SmallVector<LstmLayer, 1> layers_;
  std::vector<uint8_t> input_history_; // horizon
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_,
      output_layer_;
  std::valarray<std::valarray<float>> output_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  int last_input_ = -1;
};
#include "lstm.hpp"
#endif


