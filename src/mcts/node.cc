/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include "mcts/node.h"
#include "mcts/storage.h"

using namespace mcts;

void Node::init(Node* parent, std::unique_ptr<State> s, uint64_t stateHash) {
  assert(parent_ == nullptr);
  assert(state_ == nullptr);
  assert(children_.empty());
  // assert(depth_ == 0);
  assert(visited_ == false);

  parent_ = parent;
  state_ = std::move(s);
  stateHash_ = stateHash;
  // depth_ = parent == nullptr ? 0 : parent->getDepth() + 1;
}

void Node::reset() {
  parent_ = nullptr;
  state_ = nullptr;
  children_.clear();
  // depth_ = 0;
  visited_ = false;

  mctsStats_.reset();
  piVal_.reset();
}

void Node::acquire() {
  mSelf_.lock();
  holderThreadId_ = std::this_thread::get_id();
}

void Node::release() {
  assert(holderThreadId_ == std::this_thread::get_id());
  mSelf_.unlock();
}

Node* Node::getOrAddChild(const Action& action,
                          bool storeState,
                          bool stochastic,
                          uint64_t stateHash) {
  auto it = children_.find(action);

  std::unique_ptr<State> childState;
  if (it != children_.end()) {
    // std::cout << "child found" << std::endl;
    if (!stochastic) {  // if not stochastic we keep one child in the list,
                        // only.
      assert(it->second.size() == 1);
      // std::cout << "not stochastic" << std::endl;
      return it->second[0];
    } else {
      // stochastic games always need to forward state to determine which child
      // it lands, does not make sense to store state.
      assert(!storeState);
      for (const auto& node : it->second) {
        // TODO:[qucheng] Do we want to compare against all nodes' state?
        // answer[oteytaud] only the nodes corresponding to the right action.
        if (node->getStateHash() == stateHash) {
          // std::cout << "hash found" << std::endl;
          return node;
        }
      }

      // this is a new stochastic choice
      Node* child = storage_->newNode();
      child->init(this, std::move(childState), stateHash);
      it->second.push_back(child);
      // std::cout << " creating stochastic child" << std::endl;
      return child;
    }
  }

  uint64_t hash = stateHash;
  if (storeState) {
    childState = state_->clone();
    childState->forward(action);
    hash = childState->getHash();
  }

  std::vector<Node*> childList;
  Node* child = storage_->newNode();
  child->init(this, std::move(childState), hash);
  childList.push_back(child);
  children_.insert(it, {action, childList});
  //std::cout << " new decision child" << std::endl;
  return child;
}

namespace {
const std::vector<Node*> emptyList;
}

const std::vector<Node*>& Node::getChild(const Action& action) const {
  auto it = children_.find(action);
  if (it == children_.end()) {
    return emptyList;
  }
  return it->second;
}

void Node::freeTree() {
  if (!children_.empty()) {
    for (auto& actionNode : children_) {
      for (size_t u = 0; u < actionNode.second.size(); u++) {
        actionNode.second[u]->freeTree();
      }
    }
  }
  reset();
  storage_->freeNode(this);
}

void Node::printTree(int level, int maxLevel, int action) const {
  if (level > maxLevel) {
    return;
  }
  for (int i = 0; i < level; ++i) {
    std::cout << "    ";
  }
  std::cout << action << " " << mctsStats_.getValue() << "/"
            << mctsStats_.getNumVisit();
  std::cout << "(" << mctsStats_.getValue() / mctsStats_.getNumVisit() << ")";
  std::cout << ", vloss:" << mctsStats_.getVirtualLoss() << std::endl;
  for (const auto& pair : getChildren()) {
    pair.second[0]->printTree(level + 1, maxLevel, pair.first);
  }
}
