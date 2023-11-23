#include "context-manager.h"

ContextManager::ContextManager() : history_(100000000, 0),
    shared_map_(256*500000, 0), words_(8, 0), recent_bytes_(8, 0) {}

void ContextManager::UpdateHistory() {
  history_[history_pos_] = bit_context_;
  ++history_pos_;
  if (history_pos_ == history_.size()) history_pos_ = 0;
}

void ContextManager::UpdateWords() {
  unsigned char c = bit_context_;
  if ((c >= 'a' && c <= 'z') || c >= 0x80) {
    words_[7] = words_[7] * 997*16 + c;
  } else {
    words_[7] = 0;
  }
  if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == 8 || c == 6 ||
      c >= 0x80) {
    words_[0] = words_[0] * 997*16 + c;
    words_[0] &= 0xfffffff;
    words_[1] = words_[1] * 263*32 + c;
  } else {
    for (int i = 6; i >= 2; --i) {
      words_[i] = words_[i-1];
    }
    words_[1] = 0;
  }
}

void ContextManager::UpdateRecentBytes() {
  for (int i = 7; i >= 1; --i) {
    recent_bytes_[i] = recent_bytes_[i-1];
  }
  recent_bytes_[0] = bit_context_;
}

void ContextManager::UpdateWRTContext() {
  if (bit_context_ < 0x80) {
    wrt_state_ = 0;
  } else {
    if (wrt_state_ == 0) wrt_context_ = 0;
    wrt_state_ = 1;
    wrt_context_ <<= 8;
    wrt_context_ += bit_context_;
    if (wrt_context_ > 0xFFEFCF) wrt_context_ = 0;
  }
}

void ContextManager::UpdateContexts(int bit) {
  bit_context_ += bit_context_ + bit;
  long_bit_context_ = bit_context_;
  if (bit_context_ >= 256) {
    bit_context_ -= 256;
    long_bit_context_ = 1;
    longest_match_ = 0;

    if (bit_context_ == '\n') {
      line_break_ = 0;
    } else if (line_break_ < 99) {
      ++line_break_;
    }

    UpdateHistory();
    UpdateWords();
    UpdateRecentBytes();
    UpdateWRTContext();
    for (auto& context : interval_contexts_) {
      context.Update();
    }
    for (auto& context : interval_hash_contexts_) {
      context.Update();
    }
    for (auto& context : indirect_hash_contexts_) {
      context.Update();
    }
    for (auto& context : context_hash_contexts_) {
      context.Update();
    }
    for (auto& context : combined_contexts_) {
      context.Update();
    }
    for (auto& context : sparse_contexts_) {
      context.Update();
    }
    for (auto& context : bracket_contexts_) {
      context.Update();
    }
  }
  for (auto& context : bit_contexts_) {
    context.Update();
  }
}

