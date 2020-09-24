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

#include <algorithm>
#include <array>
#include <jni.h>
#include <memory>
#include <string>
#include <vector>

#include "../../core/state.h"
#include "ludii_game_wrapper.h"

namespace Ludii {

class Action : public ::_Action {
 public:
  Action(int i, int j, int k);
};

/**
 * C++ wrapper around Ludii's "LudiiStateWrapper" class.
 *
 * This class takes care of calling all the required Java methods from Ludii
 * states.
 */
class LudiiStateWrapper : public ::State {

 public:
  void Initialize();
  std::unique_ptr<mcts::State> clone_() const;
  void ApplyAction(const _Action& action);
  void DoGoodAction();

 public:
  /**
   * Constructor; calls the LudiiStateWrapper Java constructor
   */
  LudiiStateWrapper(int seed, LudiiGameWrapper&& inLudiiGameWrapper);

  /**
   * Copy constructor; calls the Java copy constructor for LudiiStateWrapper
   *
   * @param other The LudiiStateWrapper object of which we wish to create a deep
   * copy
   */
  LudiiStateWrapper(const LudiiStateWrapper& other);

  /**
   * Destructor
   */
  ~LudiiStateWrapper();

  /**
   * @return 2D int array; for every legal move, we have an array of
   * 	length 3 containing [channel, x, y]
   */
  std::vector<std::array<int, 3>> LegalMovesTensors() const;

  /**
   * @return Number of legal moves in current state
   */
  int NumLegalMoves() const;

  /**
   * Applies the n'th legal move in current game state
   */
  void ApplyNthMove(const int n) const;

  /**
   * NOTE: The Java method that we call for this actually first computes
   * the array of scores for all players, and then only returns the score
   * for the queried player. If we often want to do this inside a loop
   * through all players, it'd be more efficient to call a Java method
   * that instantly returns the full array once.
   *
   * @return Score in [-1.0, 1.0] for given player index (starting at 0).
   * Will always return 0.0 for non-terminal game states.
   */
  double Returns(const int player) const;

  /**
   * @return True if and only if the current game state is terminal; false
   * otherwise.
   */
  bool IsTerminal() const;

  /**
   * @return The current player to move (0 for first, 1 for second, etc.)
   */
  int CurrentPlayer() const;

  /**
   * Calls the Java reset() method on the Java game state object
   */
  void Reset() const;

  /**
   * @return State representated by a game-dependent number of channels, with
   * each channel having X and Y coordinates.
   */
  std::vector<std::vector<std::vector<float>>> ToTensor() const;

 private:
  void findFeatures();
  void findActions();

  // We don't want to be accidentally coyping objects of this class
  // (without having implemented our own, correct copy constructor or assignment
  // operator)
  LudiiStateWrapper& operator=(LudiiStateWrapper const&) = delete;

  /** Pointer to our Game wrapper */
  std::shared_ptr<LudiiGameWrapper> ludiiGameWrapper;

  /** Our object of Java's LudiiStateWrapper type */
  jobject ludiiStateWrapperJavaObject;

  /** Method ID for the legalMovesTensors() method in Java */
  jmethodID legalMovesTensorsMethodID;

  /** Method ID for the numLegalMoves() method in Java */
  jmethodID numLegalMovesMethodID;

  /** Method ID for the applyNthMove() method in Java */
  jmethodID applyNthMoveMethodID;

  /** Method ID for the returns() method in Java */
  jmethodID returnsMethodID;

  /** Method ID for the isTerminal() method in Java */
  jmethodID isTerminalMethodID;

  /** Method ID for the toTensor() method in Java */
  jmethodID toTensorMethodID;

  /** Method ID for the currentPlayer() method in Java */
  jmethodID currentPlayerMethodID;

  /** Method ID for the reset() method in Java */
  jmethodID resetMethodID;
};

}  // namespace Ludii
