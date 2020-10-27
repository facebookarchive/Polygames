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

class Storage {
  std::vector<Node*> chunks;
  size_t chunkIndex = 0;
  size_t subIndex = 0;
  size_t allocated = 0;
  const size_t chunkSize = 16;

 public:
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Node* newNode();
  void freeNode(Node* node);
  static Storage* getStorage();
};

}  // namespace mcts
