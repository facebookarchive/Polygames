/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "mcts/node.h"

#include <deque>
#include <list>

namespace mcts {

extern std::mutex freeStoragesMutex;
extern std::list<Storage*> freeStorages;

class Storage {
 public:
  Storage(size_t capacity) {
  }

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  std::vector<Node*> chunks;
  size_t chunkIndex = 0;
  size_t subIndex = 0;
  size_t allocated = 0;

  const size_t chunkSize = 16;

  Node* newNode() {
    if (chunkIndex >= chunks.size()) {
      Node* newChunk = (Node*)std::aligned_alloc(128, sizeof(Node) * chunkSize);
      new (newChunk) Node[chunkSize];
      for (size_t i = 0; i != chunkSize; ++i) {
        newChunk[i].setStorageAndId(this, i);
      }
      chunks.push_back(newChunk);
    }
    Node* r = chunks[chunkIndex] + subIndex;
    ++subIndex;
    if (subIndex == chunkSize) {
      subIndex = 0;
      ++chunkIndex;
    }
    ++allocated;
    return r;
  }

  void freeNode(Node* node) {
    --allocated;
    if (allocated == 0) {
      chunkIndex = 0;
      subIndex = 0;
      std::lock_guard l(freeStoragesMutex);
      freeStorages.push_back(this);
    }
  }

  static Storage* getStorage() {
    std::lock_guard l(freeStoragesMutex);
    if (freeStorages.empty()) {
      return new Storage(0);
    }
    Storage* r = freeStorages.back();
    freeStorages.pop_back();
    return r;
  }
};

//// helper class for managing storage space for tree
// class Storage {
// public:
//  Storage(size_t capacity) {
//    capacity =
//        (capacity + storageGroupSize - 1) / storageGroupSize *
//        storageGroupSize;
//    storage_.resize(capacity / storageGroupSize);
//    assignedNodes_.resize(capacity);
//    freeNodes_.reserve(capacity);
//    for (size_t i = 0; i < capacity; ++i) {
//      freeNodes_.push_back(i);
//      storage_[i / storageGroupSize][i % storageGroupSize].setStorageAndId(
//          this, i);
//    }
//  }

//  Storage(const Storage&) = delete;
//  Storage& operator=(const Storage&) = delete;

//  Node* newNode() {
//    std::lock_guard<std::mutex> lock(mFreeNodes_);
//    if (freeNodes_.empty()) {
//      auto nextId = storageGroupSize * storage_.size();
//      storage_.emplace_back();
//      assignedNodes_.resize(assignedNodes_.size() + storageGroupSize);
//      for (size_t i = 0; i != storageGroupSize; ++i, ++nextId) {
//        freeNodes_.push_back(nextId);
//        storage_[nextId / storageGroupSize][nextId % storageGroupSize]
//            .setStorageAndId(this, nextId);
//      }
//    }

//    auto id = freeNodes_.back();
//    freeNodes_.pop_back();
//    assert(!assignedNodes_[id]);
//    assignedNodes_[id] = true;

//    return &storage_[id / storageGroupSize][id % storageGroupSize];
//  }

//  int getNumFreeNode() {
//    return freeNodes_.size();
//  }

//  void freeNode(Node* node) {
//    const NodeId id = node->getId();
//    std::lock_guard<std::mutex> lock(mFreeNodes_);
//    assert(assignedNodes_[id]);
//    assignedNodes_[id] = false;
//    freeNodes_.push_back(id);
//  }

// private:
//  std::mutex mFreeNodes_;
//  std::vector<NodeId> freeNodes_;

//  static const size_t storageGroupSize = 0x100;

//  std::vector<bool> assignedNodes_;
//  std::deque<std::array<Node, storageGroupSize>> storage_;
//};

// using Storage = NewStorage;

}  // namespace mcts
