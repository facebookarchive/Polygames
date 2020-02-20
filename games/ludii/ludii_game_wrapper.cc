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

#include "ludii_game_wrapper.h"

namespace Ludii {

// NOTE: String descriptions of signatures of Java methods can be found by navigating
// to directory containing the .class files and using:
//
// javap -s <ClassName.class>

LudiiGameWrapper::LudiiGameWrapper(const JNIEnv* const jenv, const std::string lud_path) : jenv(jenv) {
	// Find our LudiiGameWrapper Java class
	ludiiGameWrapperClass = jenv->FindClass("utils/LudiiGameWrapper");

	// Find the LudiiGameWrapper Java constructor
	jmethodID ludiiGameWrapperConstructor =
			jenv->GetMethodID(ludiiGameWrapperClass, "<init>", "(Ljava/lang/String;)V");

	// Convert our lud path into a Java string
	jstring java_lud_path = jenv->NewStringUTF(lud_path.c_str());

	// Call our Java constructor to instantiate new object
	ludiiGameWrapperJavaObject =
			jenv->NewObject(ludiiGameWrapperClass, ludiiGameWrapperConstructor, java_lud_path);

	// Find method IDs for the two tensor shape Java methods that we may be calling frequently
	stateTensorsShapeMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorsShape", "()[I");
	moveTensorsShapeMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "moveTensorsShape", "()[I");
}

LudiiGameWrapper::LudiiGameWrapper(
		const JNIEnv* const jenv, const std::string lud_path,
		const std::vector<std::string> game_options) : jenv(jenv) {

	// Find our LudiiGameWrapper Java class
	ludiiGameWrapperClass = jenv->FindClass("utils/LudiiGameWrapper");

	// Find the LudiiGameWrapper Java constructor (with extra argument for options)
	const jmethodID ludiiGameWrapperConstructor =
			jenv->GetMethodID(ludiiGameWrapperClass, "<init>", "(Ljava/lang/String;[Ljava/lang/String;)V");

	// Convert our lud path into a Java string
	const jstring java_lud_path = jenv->NewStringUTF(lud_path.c_str());

	// Convert vector of game options into array of Java strings
	const jobjectArray java_game_options = (jobjectArray) jenv->NewObjectArray(
			game_options.size(), jenv->FindClass("java/lang/String"), nullptr);
	for (int i = 0; i < game_options.size(); ++i) {
		jenv->SetObjectArrayElement(java_game_options, i, jenv->NewStringUTF(game_options[i].c_str()));
	}

	// Call our Java constructor to instantiate new object
	ludiiGameWrapperJavaObject =
			jenv->NewObject(ludiiGameWrapperClass, ludiiGameWrapperConstructor, java_lud_path, java_game_options);

	// Find method IDs for the two tensor shape Java methods that we may be calling frequently
	stateTensorsShapeMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorsShape", "()[I");
	moveTensorsShapeMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "moveTensorsShape", "()[I");
}

std::unique_ptr<int[]> LudiiGameWrapper::StateTensorsShape() const {
	if (stateTensorsShape.get() == nullptr) {
		// Get our array of Java ints
		const jintArray jint_array = jenv->CallObjectMethod(ludiiGameWrapperJavaObject, stateTensorsShapeMethodID);
		const jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);

		// Create our C++ array of 3 ints
		stateTensorsShape = std::unique_ptr<int[]>(new int[3]);
		for (size_t i = 0; i < 3; ++i) {
			stateTensorsShape.get()[i] = (int) jints[i];
		}

		// Allow JVM to clean up memory now that we have our own ints
		jenv->ReleaseIntArrayElements(jint_array, jints, nullptr);
	}

	return stateTensorsShape;
}

std::unique_ptr<int[]> LudiiGameWrapper::MoveTensorsShape() const {
	if (moveTensorsShape.get() == nullptr) {
		// Get our array of Java ints
		const jintArray jint_array = jenv->CallObjectMethod(ludiiGameWrapperJavaObject, moveTensorsShapeMethodID);
		const jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);

		// Create our C++ array of 3 ints
		moveTensorsShape(std::make_shared<int[]>(new int[3]));
		for (size_t i = 0; i < 3; ++i) {
			shape.get()[i] = (int) jints[i];
		}

		// Allow JVM to clean up memory now that we have our own ints
		jenv->ReleaseIntArrayElements(jint_array, jints, nullptr);
	}

	return moveTensorsShape;
}

}	// namespace Ludii
