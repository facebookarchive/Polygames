/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include "mcts/node.h"
#include "mcts/storage.h"

namespace mcts {

void Node::init(Node* parent) {

  parent_ = nullptr;
  state_ = nullptr;
  children_.clear();
  visited_ = false;

  mctsStats_.reset();
  piVal_.reset();
  legalPolicy_.clear();

  parent_ = parent;
}

void Node::acquire() {
  //  mSelf_.lock();
  //  holderThreadId_ = std::this_thread::get_id();
}

void Node::release() {
  //  assert(holderThreadId_ == std::this_thread::get_id());
  //  mSelf_.unlock();
}

Node* Node::newChild(Node* child, Action action) {
  child->init(this);
  auto i = std::lower_bound(children_.begin(), children_.end(), action,
                            [](auto& a, Action b) { return a.first < b; });
  children_.insert(i, std::make_pair(action, child));
  return child;
}

namespace {
const std::vector<Node*> emptyList;
}

Node* Node::getChild(Action action) const {
  auto i = std::lower_bound(children_.begin(), children_.end(), action,
                            [](auto& a, Action b) { return a.first < b; });
  return i == children_.end() || i->first != action ? nullptr : i->second;
}

void Node::freeTree() {
  piVal_.rnnState.reset();
  for (auto& v : children_) {
    v.second->freeTree();
  }
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
  for (auto& v : getChildren()) {
    v.second->printTree(level + 1, maxLevel, v.first);
  }
}

}  // namespace mcts
