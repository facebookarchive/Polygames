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

void Node::init(Node* parent) {

  parent_ = nullptr;
  state_ = nullptr;
  children_.clear();
  //  for (auto& v : children_) {
  //    v.clear();
  //  }
  visited_ = false;

  mctsStats_.reset();
  piVal_.reset();

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

  //  size_t index = action % children_.size();
  //  auto& list = children_[index];
  //  auto i = std::lower_bound(list.begin(), list.end(), action, [](auto& a,
  //  Action b) {
  //    return a.first < b;
  //  });
  //  list.insert(i, std::make_pair(action, child));

  auto i = std::lower_bound(
      children_.begin(), children_.end(), action, [](auto& a, Action b) {
        return a.first < b;
      });
  children_.insert(i, std::make_pair(action, child));
  //  children_.emplace_back(action, child);
  return child;
}

namespace {
const std::vector<Node*> emptyList;
}

Node* Node::getChild(Action action) const {
  //  size_t index = action % children_.size();
  //  auto& list = children_[index];
  //  auto i = std::lower_bound(list.begin(), list.end(), action, [](auto& a,
  //  Action b) {
  //    return a.first < b;
  //  });
  //  return i == list.end() || i->first != action ? nullptr : i->second;

  //  for (auto& v : children_) {
  //    if (v.first == action) return v.second;
  //  }
  //  return nullptr;
  auto i = std::lower_bound(
      children_.begin(), children_.end(), action, [](auto& a, Action b) {
        return a.first < b;
      });
  return i == children_.end() || i->first != action ? nullptr : i->second;
  // return children_.size() > (size_t)action ? children_[action] : nullptr;
}

void Node::freeTree() {
  for (auto& v : children_) {
    v.second->freeTree();
  }
  //  for (auto& v : children_) {
  //    for (auto& v2 : v) {
  //      v2.second->freeTree();
  //    }
  //  }
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
  // for (const Node* node : getChildren()) {
  // pair.second[0]->printTree(level + 1, maxLevel, pair.first);
  //}
}
