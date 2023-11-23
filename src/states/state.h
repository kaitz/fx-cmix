#ifndef STATE_H
#define STATE_H

class State {
 public:
  virtual ~State() {}
  float InitProbability(int state) const { return 0.5; }
  int Next(int state, int bit) const { return 0; }
};

#endif


