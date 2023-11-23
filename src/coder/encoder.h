#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>
#include <vector>

#include "../predictor.h"

class Encoder {
 public:
  Encoder(std::ofstream* os, Predictor* p);
  void Encode(int bit);
  void Flush();
  size_t OutputSize() { return out_.size();}
 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::vector<char> out_;
  std::ofstream* os_;
  unsigned int x1_, x2_;
  Predictor* p_;
};
#endif
