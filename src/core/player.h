/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

namespace core {

class State;
class Player {
 public:
  Player(bool isHuman)
      : isHuman_(isHuman) {
    isTP_ = false;
  }

  bool isHuman() const {
    return isHuman_;
  }
  bool isTP() const {
    return isTP_;
  }

  virtual void setName(std::string name) {
    name_ = std::move(name);
  }
  const std::string& getName() {
    return name_;
  }

  virtual void terminate() {}

 private:
  std::string name_ = "unnamed";
  bool isHuman_;

 protected:
  bool isTP_;
};

}
