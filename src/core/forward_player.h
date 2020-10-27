#pragma once

#include "actor_player.h"

namespace core {

class ForwardPlayer : public ActorPlayer {
 public:
  void batchResize(size_t n) {
    actor_->batchResize(n);
  }
  void batchPrepare(size_t index, const State& s, torch::Tensor rnnState) {
    actor_->batchPrepare(index, s, rnnState);
  }
  void batchEvaluate(size_t n) {
    actor_->batchEvaluate(n);
  }
  void batchResult(size_t index, const State& s, PiVal& pival) {
    actor_->batchResult(index, s, pival);
  }
};

}  // namespace core
