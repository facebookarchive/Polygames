/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <thread>
#include <vector>

#include "mcts/state.h"
#include "mcts/types.h"
#include "mcts/utils.h"

namespace mcts {

class Storage;

class Node {
 public:
  Node()
      : storage_(nullptr)
      , id_(0) {
    reset();
  }

  void setStorageAndId(Storage* storage, NodeId id) {
    storage_ = storage;
    id_ = id;
  }

  // Node(Storage* storage, NodeId id)
  //     : storage_(storage), id_(id) {}

  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  void init(Node* parent, std::unique_ptr<State> s, uint64_t stateHash);

  void acquire();

  void release();

  std::mutex& getMutex() {
    return mSelf_;
  }

  // caller is responsible for holding locks in case of multi-threading
  Node* getOrAddChild(const Action& action,
                      bool storeState,
                      bool stochastic,
                      uint64_t stateHash);

  const std::vector<Node*>& getChild(const Action& action) const;

  MctsStats& getMctsStats() {
    return mctsStats_;
  }

  const MctsStats& getMctsStats() const {
    return mctsStats_;
  }

  const State& getState() const {
    return *state_;
  }

  NodeId getId() const {
    return id_;
  }

  const std::unordered_map<Action, std::vector<Node*>> getChildren() const {
    return children_;
  }

  Node* getParent() const {
    return parent_;
  }

  // int getDepth() const {
  //   return depth_;
  // }

  const PiVal& getPiVal() const {
    return piVal_;
  }

  void settle(int rootPlayerId, PiVal piVal) {
    // Only called when the node is locked.
    if (parent_ != nullptr) {
      auto& stats = parent_->getMctsStats();
      float upValue = rootPlayerId == piVal.playerId
                          ? piVal.value
                          : -piVal.value;
      stats.atomicUpdateChildV(upValue);
    }
    piVal_ = std::move(piVal);
    visited_ = true;
  }

  // free the entire tree rooted at this node
  void freeTree();

  bool isVisited() {
    return visited_;
  }

  uint64_t getStateHash() {
    return stateHash_;
  }

  void printTree(int level, int maxLevel, int action) const;

 private:
  // called by constructor and freeTree
  void reset();

  // set in constructor, should never be changed
  Storage* storage_;
  NodeId id_;

  // sync tools
  std::mutex mSelf_;
  std::thread::id holderThreadId_;

  // actual attributes
  Node* parent_;
  std::unique_ptr<State> state_;
  uint64_t stateHash_;
  std::unordered_map<Action, std::vector<Node*>> children_;
  // int depth_;
  bool visited_;

  MctsStats mctsStats_;
  PiVal piVal_;
};

}  // namespace mcts
