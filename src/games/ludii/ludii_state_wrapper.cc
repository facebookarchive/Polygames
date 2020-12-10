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

#include "ludii_state_wrapper.h"
#include "jni_utils.h"

namespace Ludii {

// Action::Action(int i, int j, int k) {
//  _loc[0] = i;
//  _loc[1] = j;
//  _loc[2] = k;
//  _hash = uint32_t(0);  // TODO implement hash for stochastic games
//}

void LudiiStateWrapper::Initialize() {
  Reset();

  _hash = 0;  // TODO implement hash for stochastic games
  _status = GameStatus::player0Turn;

  // Initializes Features.
  _featSize.resize(3);
  const std::array<int, 3>& sts = ludiiGameWrapper->StateTensorsShape();
  std::copy(sts.begin(), sts.end(), _featSize.begin());
  _features = std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2]);
  findFeatures();
  fillFullFeatures();

  // Initializes Actions.
  _actionSize.resize(3);
  const std::array<int, 3>& mts = ludiiGameWrapper->MoveTensorsShape();
  std::copy(mts.begin(), mts.end(), _actionSize.begin());
  findActions();
}

void LudiiStateWrapper::findFeatures() {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const jfloatArray flatTensorArray =
      static_cast<jfloatArray>(jenv->CallObjectMethod(
          ludiiStateWrapperJavaObject, toTensorFlatMethodID));
  JNIUtils::CheckJniException(jenv);
  const jsize numEntries = jenv->GetArrayLength(flatTensorArray);
  jfloat* jfloats =
      (jfloat*)jenv->GetPrimitiveArrayCritical(flatTensorArray, nullptr);
  std::copy(jfloats, jfloats + numEntries, _features.begin());

  // Allow JVM to clean up memory now that we have our own floats
  jenv->ReleasePrimitiveArrayCritical(flatTensorArray, jfloats, JNI_ABORT);
  jenv->DeleteLocalRef(flatTensorArray);
}

void LudiiStateWrapper::findActions() {
  const std::vector<std::array<int, 3>> moves = LegalMovesTensors();
  size_t nbMoves = moves.size();
  _legalActions.clear();
  _legalActions.reserve(nbMoves);
  for (size_t i = 0; i < nbMoves; ++i) {
    const std::array<int, 3>& move = moves[i];
    _legalActions.emplace_back(i, move[0], move[1], move[2]);
  }
}

std::unique_ptr<core::State> LudiiStateWrapper::clone_() const {
  return std::make_unique<LudiiStateWrapper>(*this);
}

void LudiiStateWrapper::ApplyAction(const _Action& action) {

  assert(not IsTerminal());

  // play move
  ApplyNthMove(action.GetIndex());

  // update game status
  if (IsTerminal()) {

    if (isOnePlayerGame()) {
      const double score = Returns(0);
      if (score >= 0.99)  // Probably just 1.0
        _status = GameStatus::player0Win;
      else if (score <= -0.99)  // Probably just -1.0
        _status = GameStatus::player1Win;
      else
        _status = GameStatus::tie;
    } else {
      const double score_0 = Returns(0);
      const double score_1 = Returns(1);
      if (score_0 > score_1)
        _status = score_0 > 0.0 ? GameStatus::player0Win : GameStatus::tie;
      else
        _status = score_1 > 0.0 ? GameStatus::player1Win : GameStatus::tie;
    }
  } else {
    const int player = CurrentPlayer();
    _status = player == 0 ? GameStatus::player0Turn : GameStatus::player1Turn;
  }

  // update features
  findFeatures();
  fillFullFeatures();

  // update actions
  findActions();

  // update hash  // TODO
}

void LudiiStateWrapper::DoGoodAction() {
  return DoRandomAction();
}

// NOTE: String descriptions of signatures of Java methods can be found by
// navigating to directory containing the .class files and using:
//
// javap -s <ClassName.class>

LudiiStateWrapper::LudiiStateWrapper(int seed,
                                     LudiiGameWrapper&& inLudiiGameWrapper)
    : core::State(seed) {

  JNIEnv* jenv = JNIUtils::GetEnv();
  ludiiGameWrapper =
      std::make_shared<LudiiGameWrapper>(std::move(inLudiiGameWrapper));
  jclass ludiiStateWrapperClass = JNIUtils::LudiiStateWrapperClass();

  // Find the LudiiStateWrapper Java constructor
  jmethodID ludiiStateWrapperConstructor = jenv->GetMethodID(
      ludiiStateWrapperClass, "<init>", "(Lutils/LudiiGameWrapper;)V");
  JNIUtils::CheckJniException(jenv);

  // Call our Java constructor to instantiate new object
  jobject local_ref =
      jenv->NewObject(ludiiStateWrapperClass, ludiiStateWrapperConstructor,
                      ludiiGameWrapper->ludiiGameWrapperJavaObject);
  JNIUtils::CheckJniException(jenv);
  ludiiStateWrapperJavaObject = jenv->NewGlobalRef(local_ref);
  jenv->DeleteLocalRef(local_ref);

  // Find method IDs for all the Java methods we may want to call
  legalMovesTensorsMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "legalMovesTensors", "()[[I");
  JNIUtils::CheckJniException(jenv);
  numLegalMovesMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "numLegalMoves", "()I");
  JNIUtils::CheckJniException(jenv);
  applyNthMoveMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "applyNthMove", "(I)V");
  JNIUtils::CheckJniException(jenv);
  returnsMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "returns", "(I)D");
  JNIUtils::CheckJniException(jenv);
  isTerminalMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "isTerminal", "()Z");
  JNIUtils::CheckJniException(jenv);
  toTensorFlatMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "toTensorFlat", "()[F");
  JNIUtils::CheckJniException(jenv);
  currentPlayerMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "currentPlayer", "()I");
  JNIUtils::CheckJniException(jenv);
  resetMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "reset", "()V");
  JNIUtils::CheckJniException(jenv);
  copyFromMethodID = 
      jenv->GetMethodID(ludiiStateWrapperClass, "copyFrom", "(Lutils/LudiiStateWrapper;)V");
  JNIUtils::CheckJniException(jenv);
  getRandomRolloutsRewardMethodID =
      jenv->GetMethodID(ludiiStateWrapperClass, "getRandomRolloutsReward", "(III)D");
  JNIUtils::CheckJniException(jenv);
}

LudiiStateWrapper::LudiiStateWrapper(const LudiiStateWrapper& other)
    : core::State(other)
    , ludiiGameWrapper(other.ludiiGameWrapper) {

  JNIEnv* jenv = JNIUtils::GetEnv();
  jclass ludiiStateWrapperClass = JNIUtils::LudiiStateWrapperClass();

  // Find the LudiiStateWrapper Java copy constructor
  jmethodID ludiiStateWrapperCopyConstructor = jenv->GetMethodID(
      ludiiStateWrapperClass, "<init>", "(Lutils/LudiiStateWrapper;)V");
  JNIUtils::CheckJniException(jenv);

  // Call our Java constructor to instantiate new object
  jobject local_ref =
      jenv->NewObject(ludiiStateWrapperClass, ludiiStateWrapperCopyConstructor,
                      other.ludiiStateWrapperJavaObject);
  JNIUtils::CheckJniException(jenv);
  ludiiStateWrapperJavaObject = jenv->NewGlobalRef(local_ref);
  jenv->DeleteLocalRef(local_ref);

  // We can just copy all the pointers to methods
  legalMovesTensorsMethodID = other.legalMovesTensorsMethodID;
  numLegalMovesMethodID = other.numLegalMovesMethodID;
  applyNthMoveMethodID = other.applyNthMoveMethodID;
  returnsMethodID = other.returnsMethodID;
  isTerminalMethodID = other.isTerminalMethodID;
  toTensorFlatMethodID = other.toTensorFlatMethodID;
  currentPlayerMethodID = other.currentPlayerMethodID;
  resetMethodID = other.resetMethodID;
  copyFromMethodID = other.copyFromMethodID;
  getRandomRolloutsRewardMethodID = other.getRandomRolloutsRewardMethodID;
}

LudiiStateWrapper& LudiiStateWrapper::operator=(LudiiStateWrapper const& other) {
  if (&other == this)
    return *this;

  core::State::operator=(other);

  JNIEnv* jenv = JNIUtils::GetEnv();
  jenv->CallVoidMethod(ludiiStateWrapperJavaObject, copyFromMethodID, other.ludiiStateWrapperJavaObject);
  JNIUtils::CheckJniException(jenv);
  
  // We can just copy all the pointers to methods
  legalMovesTensorsMethodID = other.legalMovesTensorsMethodID;
  numLegalMovesMethodID = other.numLegalMovesMethodID;
  applyNthMoveMethodID = other.applyNthMoveMethodID;
  returnsMethodID = other.returnsMethodID;
  isTerminalMethodID = other.isTerminalMethodID;
  toTensorFlatMethodID = other.toTensorFlatMethodID;
  currentPlayerMethodID = other.currentPlayerMethodID;
  resetMethodID = other.resetMethodID;
  copyFromMethodID = other.copyFromMethodID;
  getRandomRolloutsRewardMethodID = other.getRandomRolloutsRewardMethodID;
  
  return *this;
}

LudiiStateWrapper::~LudiiStateWrapper() {
  JNIEnv* jenv = JNIUtils::GetEnv();
  if (jenv) {
    jenv->DeleteGlobalRef(ludiiStateWrapperJavaObject);
  }
}

std::vector<std::array<int, 3>> LudiiStateWrapper::LegalMovesTensors() const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const jobjectArray javaArrOuter =
      static_cast<jobjectArray>(jenv->CallObjectMethod(
          ludiiStateWrapperJavaObject, legalMovesTensorsMethodID));
  JNIUtils::CheckJniException(jenv);
  const jsize numLegalMoves = jenv->GetArrayLength(javaArrOuter);

  std::vector<std::array<int, 3>> matrix(numLegalMoves);
  for (jsize i = 0; i < numLegalMoves; ++i) {
    const jintArray inner =
        static_cast<jintArray>(jenv->GetObjectArrayElement(javaArrOuter, i));
    jint* jints = (jint*)jenv->GetPrimitiveArrayCritical(inner, nullptr);

    matrix[i] = {jints[0], jints[1], jints[2]};

    // Allow JVM to clean up memory now that we have our own ints
    jenv->ReleasePrimitiveArrayCritical(inner, jints, JNI_ABORT);
    jenv->DeleteLocalRef(inner);
  }

  jenv->DeleteLocalRef(javaArrOuter);

  return matrix;
}

int LudiiStateWrapper::NumLegalMoves() const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const int num_legal_moves = (int)jenv->CallIntMethod(
      ludiiStateWrapperJavaObject, numLegalMovesMethodID);
  JNIUtils::CheckJniException(jenv);
  return num_legal_moves;
}

void LudiiStateWrapper::ApplyNthMove(const int n) const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  jenv->CallVoidMethod(ludiiStateWrapperJavaObject, applyNthMoveMethodID, n);
  JNIUtils::CheckJniException(jenv);
}

double LudiiStateWrapper::Returns(const int player) const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const double returns = (double)jenv->CallDoubleMethod(
      ludiiStateWrapperJavaObject, returnsMethodID, player);
  JNIUtils::CheckJniException(jenv);
  return returns;
}

bool LudiiStateWrapper::IsTerminal() const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const bool is_terminal = (bool)jenv->CallBooleanMethod(
      ludiiStateWrapperJavaObject, isTerminalMethodID);
  JNIUtils::CheckJniException(jenv);
  return is_terminal;
}

int LudiiStateWrapper::CurrentPlayer() const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  const int current_player = (int)jenv->CallIntMethod(
      ludiiStateWrapperJavaObject, currentPlayerMethodID);
  JNIUtils::CheckJniException(jenv);
  return current_player;
}

void LudiiStateWrapper::Reset() const {
  JNIEnv* jenv = JNIUtils::GetEnv();
  jenv->CallVoidMethod(ludiiStateWrapperJavaObject, resetMethodID);
  JNIUtils::CheckJniException(jenv);
}

bool LudiiStateWrapper::isOnePlayerGame() const {
  return (ludiiGameWrapper->NumPlayers() == 1);
}

float LudiiStateWrapper::getRandomRolloutReward(int player) const {
  const int numSimulation = 10;
  const int rolloutRandomMovesCap = 200;	// Use -1 for no cap on num moves in rollout
  JNIEnv* jenv = JNIUtils::GetEnv();
  const double avgReward = (double)jenv->CallDoubleMethod(
      ludiiStateWrapperJavaObject, getRandomRolloutsRewardMethodID, player, numSimulation, rolloutRandomMovesCap);
  JNIUtils::CheckJniException(jenv);
  return (float) avgReward;
}

}  // namespace Ludii
