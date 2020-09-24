/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: Dennis Soemers
// - Affiliation: Maastricht University, DKE, Digital Ludeme Project (Ludii
// developer)
// - Github: https://github.com/DennisSoemers/
// - Email: dennis.soemers@maastrichtuniversity.nl (or d.soemers@gmail.com)

#pragma once

#include <array>
#include <jni.h>
#include <memory>
#include <string>
#include <vector>

namespace Ludii {

/**
 * C++ wrapper around Ludii's "LudiiGameWrapper" class.
 *
 * This class takes care of calling all the required Java methods from Ludii
 * games.
 */
class LudiiGameWrapper {

 public:
  /**
   * Constructor; calls the LudiiGameWrapper Java constructor
   *
   * @param lud_path String describing the path of the game to load. Should end
   * in .lud
   */
  LudiiGameWrapper(const std::string lud_path);

  /**
   * Constructor; calls the LudiiGameWrapper Java constructor
   *
   * @param lud_path String describing the path of the game to load. Should end
   * in .lud
   * @param game_options Vector of additiona options to pass into Ludii,
   * describing variant of game to load.
   */
  LudiiGameWrapper(const std::string lud_path,
                   const std::vector<std::string> game_options);

  /**
   * Copy constructor. Re-uses the same Java LudiiGameWrapper object.
   */
  LudiiGameWrapper(LudiiGameWrapper const&);

  /**
   * Copy-assignment operator. Re-uses the same Java LudiiGameWrapper object.
   */
  LudiiGameWrapper& operator=(LudiiGameWrapper const& other);

  /**
   * Destructor
   */
  ~LudiiGameWrapper();

  /**
   * @return Array of 3 ints describing the shape of state tensors; [channels,
   * x, y]
   */
  const std::array<int, 3>& StateTensorsShape();

  /**
   * @return Array of 3 ints describing the shape of move tensors; [channels, x,
   * y]
   */
  const std::array<int, 3>& MoveTensorsShape();

  /**
   * @return Vector with, for every channel in state tensors, a name describing
   * what data we have in that channel.
   */
  const std::vector<std::string> stateTensorChannelNames();

  /** Our object of Java's LudiiGameWrapper type */
  jobject ludiiGameWrapperJavaObject;

 private:
  /** Method ID for the stateTensorsShape() method in Java */
  jmethodID stateTensorsShapeMethodID;

  /** Method ID for the moveTensorsShape() method in Java */
  jmethodID moveTensorsShapeMethodID;

  /** Method ID for the stateTensorChannelNames() method in Java */
  jmethodID stateTensorChannelNamesMethodID;

  /**
   * Shape for state tensors.
   * This remains constant throughout episodes, so can just compute it once and
   * store
   */
  std::unique_ptr<std::array<int, 3>> stateTensorsShape;

  /**
   * Shape for state tensors.
   * This remains constant throughout episodes, so can just compute it once and
   * store
   */
  std::unique_ptr<std::array<int, 3>> moveTensorsShape;
};

}  // namespace Ludii
