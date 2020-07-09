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
  }

  void setStorageAndId(Storage* storage, NodeId id) {
    storage_ = storage;
    id_ = id;
  }

  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  void init(Node* parent);

  void acquire();

  void release();

  // caller is responsible for holding locks in case of multi-threading
  Node* newChild(Node* childNode, Action action);

  Node* getChild(Action action) const;

  MctsStats& getMctsStats() {
    return mctsStats_;
  }

  const MctsStats& getMctsStats() const {
    return mctsStats_;
  }

  const State& getState() const {
    return *state_;
  }

  bool hasState() const {
    return state_ != nullptr;
  }

  void setState(State* state) {
    state_ = state;
  }

  std::unique_ptr<State>& localState() {
    return localState_;
  }

  NodeId getId() const {
    return id_;
  }

  const auto& getChildren() const {
    return children_;
  }

  Node* getParent() const {
    return parent_;
  }

  const PiVal& getPiVal() const {
    return piVal_;
  }

  void settle(int rootPlayerId) {
    // Only called when the node is locked.
    if (parent_ != nullptr) {
      auto& stats = parent_->getMctsStats();
      float upValue =
          rootPlayerId == piVal_.playerId ? piVal_.value : -piVal_.value;
      stats.atomicUpdateChildV(upValue);
    }
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

  // private:

  // std::pair<Node*, Node*> link;

  // set in constructor, should never be changed
  Storage* storage_;
  NodeId id_;

  // sync tools
  // std::mutex mSelf_;
  // std::thread::id holderThreadId_;

  // actual attributes
  Node* parent_;
  std::unique_ptr<State> localState_;
  State* state_;
  uint64_t stateHash_;
  // std::unordered_map<Action, std::vector<Node*>> children_;
  std::vector<std::pair<Action, Node*>> children_;
  // int depth_;
  bool visited_;

  MctsStats mctsStats_;
  PiVal piVal_;
};

}  // namespace mcts
