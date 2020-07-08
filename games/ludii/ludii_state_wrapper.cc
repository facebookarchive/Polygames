/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: Dennis Soemers
// - Affiliation: Maastricht University, DKE, Digital Ludeme Project (Ludii developer)
// - Github: https://github.com/DennisSoemers/
// - Email: dennis.soemers@maastrichtuniversity.nl (or d.soemers@gmail.com)

#include "jni_utils.h"
#include "ludii_state_wrapper.h"

namespace Ludii {

Action::Action(int i, int j, int k) {
    _loc[0] = i;
    _loc[1] = j;
    _loc[2] = k;
    _hash = uint32_t(0);  // TODO implement hash for stochastic games
}

void LudiiStateWrapper::Initialize() {

	// In case we're calling this from a new thread, make sure that we're attached to JVM
	// (following few lines based on: https://stackoverflow.com/a/12901159/6735980)
	JavaVM* jvm;
	jenv->GetJavaVM(&jvm);
	int jenvStatus = jvm->GetEnv((void **)&jenv, JNI_VERSION_1_6);

	if (jenvStatus == JNI_EDETACHED) {
//		std::cout << "Was not yet attached" << std::endl;
		if (jvm->AttachCurrentThread((void **) &jenv, NULL) != 0) {
			std::cout << "Failed to attach" << std::endl;
		}
		else if (jenvStatus == JNI_EVERSION) {
			std::cout << "JVM GetEnv: version not supported" << std::endl;
		}
//		else {
//			std::cout << "Should be attached fine now" << std::endl;
//		}
	}

	// Now do the normal initialisation
	Reset();

    _hash = 0;  // TODO implement hash for stochastic games
    _status = GameStatus::player0Turn;

    // Initializes Features.
    _featSize.resize(3);
    const std::array<int,3> & sts = ludiiGameWrapper.StateTensorsShape();
    std::copy(sts.begin(), sts.end(), _featSize.begin());
    _features = std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2]);
    findFeatures();
    fillFullFeatures();

    // Initializes Actions.
    _actionSize.resize(3);
    const std::array<int,3> & mts = ludiiGameWrapper.MoveTensorsShape();
    std::copy(mts.begin(), mts.end(), _actionSize.begin());
    findActions();
}

void LudiiStateWrapper::findFeatures() {
    const auto tensor = ToTensor();
    int k=0;
    for (int x=0; x<_featSize[0]; ++x) {
        for (int y=0; y<_featSize[1]; ++y) {
            for (int z=0; z<_featSize[2]; ++z) {
                _features[k] = tensor[x][y][z];
                ++k;
            }
        }
    }
}

void LudiiStateWrapper::findActions() {
    const std::vector<std::array<int, 3>> moves = LegalMovesTensors();
    int nbMoves = NumLegalMoves();
    _legalActions.clear();
    _legalActions.reserve(nbMoves);
    for (int i=0; i<nbMoves; ++i) {
        const std::array<int,3> & move = moves[i];
        _legalActions.push_back(
                std::make_shared<Ludii::Action>(move[0], move[1], move[2]));
        _legalActions.back()->SetIndex(i);
    }
}

std::unique_ptr<mcts::State> LudiiStateWrapper::clone_() const {
  return std::make_unique<LudiiStateWrapper>(*this);
}

void LudiiStateWrapper::ApplyAction(const _Action& action) {

  assert(not IsTerminal());

  // find move from action
  const std::array<int, 3> move { action.GetX(), action.GetY(), action.GetZ() }; 
  const auto moves = LegalMovesTensors();
  const auto it = std::find(moves.begin(), moves.end(), move);
  assert(it != moves.end());

  // play move
  const int n = std::distance(moves.begin(), it);
  ApplyNthMove(n);

  // update game status
  // TODO only 2-player games ?
  // TODO I have no idea how to get the current player/winner !!!

  // We don't want this assert: We have more than 2 channels, so also more than 2 names!
  //assert(ludiiGameWrapper.stateTensorChannelNames().size() == 2);

  const double score_0 = Returns(0);
  const double score_1 = Returns(1);
  if (IsTerminal()) {
      if (score_0 > score_1)
          _status = score_0 > 0.0 ? GameStatus::player0Win : GameStatus::tie;
      else
          _status = score_1 > 0.0 ? GameStatus::player1Win : GameStatus::tie;
  }
  else {
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

// NOTE: String descriptions of signatures of Java methods can be found by navigating
// to directory containing the .class files and using:
//
// javap -s <ClassName.class>

LudiiStateWrapper::LudiiStateWrapper(int seed, JNIEnv* jenv, LudiiGameWrapper && ludiiGameWrapper)
	: ::State(seed), jenv(jenv), ludiiGameWrapper(ludiiGameWrapper) {

	jclass ludiiStateWrapperClass = JNIUtils::LudiiStateWrapperClass();

	// Find the LudiiStateWrapper Java constructor
	jmethodID ludiiStateWrapperConstructor =
			jenv->GetMethodID(ludiiStateWrapperClass, "<init>", "(Lutils/LudiiGameWrapper;)V");

	// Call our Java constructor to instantiate new object
	ludiiStateWrapperJavaObject =
			jenv->NewObject(ludiiStateWrapperClass, ludiiStateWrapperConstructor, ludiiGameWrapper.ludiiGameWrapperJavaObject);

	// Find method IDs for all the Java methods we may want to call
	legalMovesTensorsMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "legalMovesTensors", "()[[I");
	numLegalMovesMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "numLegalMoves", "()I");
	applyNthMoveMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "applyNthMove", "(I)V");
	returnsMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "returns", "(I)D");
	isTerminalMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "isTerminal", "()Z");
	toTensorMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "toTensor", "()[[[F");
	currentPlayerMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "currentPlayer", "()I");
	resetMethodID = jenv->GetMethodID(ludiiStateWrapperClass, "reset", "()V");
}

LudiiStateWrapper::LudiiStateWrapper(const LudiiStateWrapper& other)
	: ::State(other), jenv(other.jenv), ludiiGameWrapper(other.ludiiGameWrapper) {

	jclass ludiiStateWrapperClass = JNIUtils::LudiiStateWrapperClass();

	// Find the LudiiStateWrapper Java copy constructor
	jmethodID ludiiStateWrapperCopyConstructor =
			jenv->GetMethodID(ludiiStateWrapperClass, "<init>", "(Lutils/LudiiStateWrapper;)V");

	// Call our Java constructor to instantiate new object
	ludiiStateWrapperJavaObject =
			jenv->NewObject(ludiiStateWrapperClass, ludiiStateWrapperCopyConstructor, other.ludiiStateWrapperJavaObject);

	// We can just copy all the pointers to methods
	legalMovesTensorsMethodID = other.legalMovesTensorsMethodID;
	numLegalMovesMethodID = other.numLegalMovesMethodID;
	applyNthMoveMethodID = other.applyNthMoveMethodID;
	returnsMethodID = other.returnsMethodID;
	isTerminalMethodID = other.isTerminalMethodID;
	toTensorMethodID = other.toTensorMethodID;
	currentPlayerMethodID = other.currentPlayerMethodID;
	resetMethodID = other.resetMethodID;
}

std::vector<std::array<int, 3>> LudiiStateWrapper::LegalMovesTensors() const {
	const jobjectArray javaArrOuter = static_cast<jobjectArray>(jenv->CallObjectMethod(ludiiStateWrapperJavaObject, legalMovesTensorsMethodID));
	const jsize numLegalMoves = jenv->GetArrayLength(javaArrOuter);

	std::vector<std::array<int, 3>> matrix(numLegalMoves);
	for (jsize i = 0; i < numLegalMoves; ++i) {
		const jintArray inner = static_cast<jintArray>(jenv->GetObjectArrayElement(javaArrOuter, i));
		jint* jints = jenv->GetIntArrayElements(inner, nullptr);

        matrix[i] = {jints[0], jints[1], jints[2]};

		// Allow JVM to clean up memory now that we have our own ints
		jenv->ReleaseIntArrayElements(inner, jints, 0);
	}

	return matrix;
}

int LudiiStateWrapper::NumLegalMoves() const {
	return (int) jenv->CallIntMethod(ludiiStateWrapperJavaObject, numLegalMovesMethodID);
}

void LudiiStateWrapper::ApplyNthMove(const int n) const {
	jenv->CallVoidMethod(ludiiStateWrapperJavaObject, applyNthMoveMethodID, n);
}

double LudiiStateWrapper::Returns(const int player) const {
	return (double) jenv->CallDoubleMethod(ludiiStateWrapperJavaObject, returnsMethodID, player);
}

bool LudiiStateWrapper::IsTerminal() const {
	return (bool) jenv->CallBooleanMethod(ludiiStateWrapperJavaObject, isTerminalMethodID);
}

int LudiiStateWrapper::CurrentPlayer() const {
	return (int) jenv->CallIntMethod(ludiiStateWrapperJavaObject, currentPlayerMethodID);
}

void LudiiStateWrapper::Reset() const {
	jenv->CallVoidMethod(ludiiStateWrapperJavaObject, resetMethodID);
}

std::vector<std::vector<std::vector<float>>> LudiiStateWrapper::ToTensor() const {
	const jobjectArray channelsArray = static_cast<jobjectArray>(jenv->CallObjectMethod(ludiiStateWrapperJavaObject, toTensorMethodID));
	const jsize numChannels = jenv->GetArrayLength(channelsArray);

	std::vector<std::vector<std::vector<float>>> tensor(numChannels);
	for (jsize c = 0; c < numChannels; ++c) {
		const jobjectArray xArray = static_cast<jobjectArray>(jenv->GetObjectArrayElement(channelsArray, c));
		const jsize numXCoords = jenv->GetArrayLength(xArray);

		tensor[c].resize(numXCoords);
		for (jsize x = 0; x < numXCoords; ++x) {
			const jfloatArray yArray = static_cast<jfloatArray>(jenv->GetObjectArrayElement(xArray, x));
			const jsize numYCoords = jenv->GetArrayLength(yArray);
			jfloat* jfloats = jenv->GetFloatArrayElements(yArray, nullptr);

			tensor[c][x].resize(numYCoords);
			std::copy(jfloats, jfloats + numYCoords, tensor[c][x].begin());

			// Allow JVM to clean up memory now that we have our own ints
			jenv->ReleaseFloatArrayElements(yArray, jfloats, 0);
		}
	}

	return tensor;
}

}	// namespace Ludii
