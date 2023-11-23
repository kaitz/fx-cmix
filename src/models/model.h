#ifndef MODEL_H
#define MODEL_H

#include <valarray>

class Model {
 public:
  Model() : outputs_(0.5, 1) {}
  Model(int size) : outputs_(0.5, size) {}
  ~Model() {}
  const std::valarray<float>& Predict() const {return outputs_;}
  unsigned int NumOutputs() {return outputs_.size();}
  void Perceive(int bit) {}
  void ByteUpdate() {}

 protected:
  mutable std::valarray<float> outputs_;
};

#endif

