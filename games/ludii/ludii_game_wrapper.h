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


#pragma once

#include <array>
#include <jni.h>  // NOLINT
#include <memory>
#include <string>
#include <vector>

namespace Ludii {

/**
 * C++ wrapper around Ludii's "LudiiGameWrapper" class.
 *
 * This class takes care of calling all the required Java methods from Ludii games.
 */
class LudiiGameWrapper {

public:

	/**
	 * Constructor; calls the LudiiGameWrapper Java constructor
	 *
	 * @param jenv Our JNI environment
	 * @param lud_path String describing the path of the game to load. Should end in .lud
	 */
	LudiiGameWrapper(JNIEnv* jenv, const std::string lud_path);

	/**
	 * Constructor; calls the LudiiGameWrapper Java constructor
	 *
	 * @param jenv Our JNI environment
	 * @param lud_path String describing the path of the game to load. Should end in .lud
	 * @param game_options Vector of additiona options to pass into Ludii, describing variant of game to load.
	 */
	LudiiGameWrapper(JNIEnv* jenv, const std::string lud_path, const std::vector<std::string> game_options);
	/**
	 * Copy constructor; calls the Java copy constructor for LudiiGameWrapper
	 *
	 * @param other The LudiiGameWrapper object of which we wish to create a deep copy
	 */
	LudiiGameWrapper(LudiiGameWrapper const&);

	/**
	 * @return Array of 3 ints describing the shape of state tensors; [channels, x, y]
	 */
	int* StateTensorsShape();

	/**
	 * @return Array of 3 ints describing the shape of move tensors; [channels, x, y]
	 */
	int* MoveTensorsShape();

	/** Our object of Java's LudiiGameWrapper type */
	jobject ludiiGameWrapperJavaObject;

private:
	// We don't want to be accidentally coyping objects of this class
	// (without having implemented our own, correct copy constructor or assignment operator)
	LudiiGameWrapper& operator=(LudiiGameWrapper const&) = delete;

	/** Pointer to the JNI environment, allows for communication with Ludii's Java code */
	JNIEnv* jenv;

	/** Our LudiiGameWrapper class in Java */
	jclass ludiiGameWrapperClass;

	/** Method ID for the stateTensorsShape() method in Java */
	jmethodID stateTensorsShapeMethodID;

	/** Method ID for the moveTensorsShape() method in Java */
	jmethodID moveTensorsShapeMethodID;

	/**
	 * Shape for state tensors.
	 * This remains constant throughout episodes, so can just compute it once and store
	 */
	std::unique_ptr<std::array<int,3>> stateTensorsShape;

	/**
	 * Shape for state tensors.
	 * This remains constant throughout episodes, so can just compute it once and store
	 */
	std::unique_ptr<std::array<int,3>> moveTensorsShape;

};

}	// namespace Ludii
