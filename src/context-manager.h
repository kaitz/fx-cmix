#ifndef CONTEXT_MANAGER_H
#define CONTEXT_MANAGER_H

#include "states/nonstationary.h"
#include "states/run-map.h"
#include "contexts/context.h"
#include "contexts/bit-context.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/combined-context.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/indirect-hash.h"
#include "contexts/sparse.h"

#include <cmath>
#include "ds/SmallVector.h"

#include <vector>

struct ContextManager {
  ContextManager();

  template<typename ... Args>
  const IndirectHash& AddIndirectHashContext(Args&& ...args) {
    IndirectHash tmp(std::forward<Args>(args)...);
    for (const auto& old : indirect_hash_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    indirect_hash_contexts_.emplace_back(std::forward<Args>(args)...);
    return indirect_hash_contexts_[indirect_hash_contexts_.size() - 1];
  }

  template<typename ... Args>
  const BracketContext& AddBracketContext(Args&& ...args) {
    BracketContext tmp(std::forward<Args>(args)...);
    for (const auto& old : bracket_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    bracket_contexts_.emplace_back(std::forward<Args>(args)...);
    return bracket_contexts_[bracket_contexts_.size() - 1];
  }

  template<typename ... Args>
  const CombinedContext& AddCombinedContext(Args&& ...args) {
    CombinedContext tmp(std::forward<Args>(args)...);
    for (const auto& old : combined_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    combined_contexts_.emplace_back(std::forward<Args>(args)...);
    return combined_contexts_[combined_contexts_.size() - 1];
  }

  template<typename ... Args>
  const ContextHash& AddContextHashContext(Args&& ...args) {
    ContextHash tmp(std::forward<Args>(args)...);
    for (const auto& old : context_hash_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    context_hash_contexts_.emplace_back(std::forward<Args>(args)...);
    return context_hash_contexts_[context_hash_contexts_.size() - 1];
  }

  template<typename ... Args>
  const IntervalHash& AddIntervalHashContext(Args&& ...args) {
    IntervalHash tmp(std::forward<Args>(args)...);
    for (const auto& old : interval_hash_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    interval_hash_contexts_.emplace_back(std::forward<Args>(args)...);
    return interval_hash_contexts_[interval_hash_contexts_.size() - 1];
  }

  template<typename ... Args>
  const Interval& AddIntervalContext(Args&& ...args) {
    Interval tmp(std::forward<Args>(args)...);
    for (const auto& old : interval_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    interval_contexts_.emplace_back(std::forward<Args>(args)...);
    return interval_contexts_[interval_contexts_.size() - 1];
  }


  template<typename ... Args>
  const Sparse& AddSparseContext(Args&& ...args) {
    Sparse tmp(std::forward<Args>(args)...);
    for (const auto& old : sparse_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    sparse_contexts_.emplace_back(std::forward<Args>(args)...);
    return sparse_contexts_[sparse_contexts_.size() - 1];
  }

  template<typename ... Args>
  const BitContext& AddBitContext(Args&& ...args) {
    BitContext tmp(std::forward<Args>(args)...);
    for (const auto& old : bit_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    bit_contexts_.emplace_back(std::forward<Args>(args)...);
    return bit_contexts_[bit_contexts_.size() - 1];
  }
  void UpdateContexts(int bit);
  void UpdateHistory();
  void UpdateWords();
  void UpdateRecentBytes();
  void UpdateWRTContext();

  unsigned int bit_context_ = 1, wrt_state_ = 0;
  unsigned long long long_bit_context_ = 1, zero_context_ = 0, history_pos_ = 0,
      line_break_ = 0, longest_match_ = 0, auxiliary_context_ = 0,
      wrt_context_ = 0;
  std::vector<unsigned char> history_, shared_map_;
  std::vector<unsigned long long> words_, recent_bytes_;
  llvm::SmallVector<Interval, 8> interval_contexts_;
  llvm::SmallVector<IntervalHash, 1> interval_hash_contexts_;
  llvm::SmallVector<IndirectHash, 11> indirect_hash_contexts_;
  llvm::SmallVector<ContextHash, 12> context_hash_contexts_;
  llvm::SmallVector<CombinedContext, 2> combined_contexts_;
  llvm::SmallVector<Sparse, 18> sparse_contexts_;
  llvm::SmallVector<BracketContext, 1> bracket_contexts_;
  llvm::SmallVector<BitContext, 10> bit_contexts_;

  RunMap run_map_;
  Nonstationary nonstationary_;
};

#endif

