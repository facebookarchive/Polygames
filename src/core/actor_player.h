#pragma once

#include "actor.h"
#include "player.h"

namespace core {

class ActorPlayer : public Player {
public:
  ActorPlayer() : Player(false) {}

  void setActor(std::shared_ptr<Actor> actor) {
    actor_ = std::move(actor);
  }

  void recordMove(const core::State* state) {
    actor_->recordMove(state);
  }

  void result(const core::State* state, float reward) {
    actor_->result(state, reward);
  }

  void forget(const core::State* state) {
    actor_->forget(state);
  }

  bool isTournamentOpponent() const {
    return actor_->isTournamentOpponent();
  }

  bool wantsTournamentResult() const {
    return actor_->wantsTournamentResult();
  }

  std::string getModelId() const {
    return actor_->getModelId();
  }

  float calculateValue(const core::State& state) {
    PiVal result;
    actor_->evaluate(state, result);
    return result.value;
  }

  std::vector<torch::Tensor> nextRnnState(
      const std::vector<const core::State*>& state,
      const std::vector<torch::Tensor>& rnnState) {
    PiVal result;
    actor_->batchResize(state.size());
    for (size_t i = 0; i != state.size(); ++i) {
      actor_->batchPrepare(i, *state[i], rnnState[i]);
    }
    actor_->batchEvaluate(state.size());
    std::vector<torch::Tensor> r;
    for (size_t i = 0; i != state.size(); ++i) {
      actor_->batchResult(i, *state[i], result);
      r.push_back(result.rnnState);
    }
    return r;
  }

  std::vector<int64_t> rnnStateSize() const {
    return actor_->rnnStateSize();
  }

  int rnnSeqlen() const {
    return actor_->rnnSeqlen();
  }

  int vOutputs() const {
    return actor_->vOutputs();
  }

  int findBatchSize(const core::State& state) const {
    return actor_->findBatchSize(state);
  }

  virtual void terminate() override {
    if (actor_) {
      actor_->terminate();
    }
  }

protected:
  std::shared_ptr<Actor> actor_;

};

}
