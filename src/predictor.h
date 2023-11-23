#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mixer/sigmoid.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/byte-mixer.h"
#include "mixer/sse.h"
#include "models/model.h"
#include "models/byte-model.h"
#include "context-manager.h"
#include "models/direct.h"
#include "models/direct-hash.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#include "mixer/lstm.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/sparse.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/bit-context.h"
#include "contexts/combined-context.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include <vector>
#include <set>
#include <memory>
#include <optional>

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab);
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);

 private:
  unsigned long long GetNumModels();
  void AddMixer(int layer, const unsigned long long& context,
      float learning_rate);
  void AddAuxiliary();
  void AddFXCM();
  //void AddPAQ8();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddDirect();
  void AddMatch();
  void AddDoubleIndirect();
  void AddMixers();

  llvm::SmallVector<Indirect<Nonstationary>, 30> indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_; // run map
  llvm::SmallVector<Direct, 1> direct_models_;
  llvm::SmallVector<Match, 16> match_models_;
  std::optional<FXCM> fxcm_model_;
  std::optional<Bracket> bracket_model_;
  size_t auxiliary_size_ = 2; // 0 -> fxcm, 1 -> byte_mixer
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  llvm::SmallVector<Mixer, 25> mixer_0_;
  llvm::SmallVector<Mixer, 25> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  Sigmoid sigmoid_;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<ByteMixer> byte_mixer_;
  std::vector<bool> vocab_;
};

#endif

