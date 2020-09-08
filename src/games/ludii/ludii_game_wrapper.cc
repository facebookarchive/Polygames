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
#include "ludii_game_wrapper.h"

namespace Ludii {

// NOTE: String descriptions of signatures of Java methods can be found by navigating
// to directory containing the .class files and using:
//
// javap -s <ClassName.class>

LudiiGameWrapper::LudiiGameWrapper(JNIEnv* jenv, const std::string lud_path) : jenv(jenv) {
	jclass ludiiGameWrapperClass = JNIUtils::LudiiGameWrapperClass();

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

	// Find the method ID for the stateTensorChannelNames() method in Java
	stateTensorChannelNamesMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorChannelNames", "()[Ljava/lang/String;");
}

LudiiGameWrapper::LudiiGameWrapper(
		JNIEnv* jenv, const std::string lud_path,
		const std::vector<std::string> game_options) : jenv(jenv) {

	jclass ludiiGameWrapperClass = JNIUtils::LudiiGameWrapperClass();

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

	// Find the method ID for the stateTensorChannelNames() method in Java
	stateTensorChannelNamesMethodID = jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorChannelNames", "()[Ljava/lang/String;");
}

LudiiGameWrapper::LudiiGameWrapper(LudiiGameWrapper const& other)
    : jenv(other.jenv) {

	jclass ludiiGameWrapperClass = JNIUtils::LudiiGameWrapperClass();

	// We can just copy the pointer to the same Java Game object
	ludiiGameWrapperJavaObject = other.ludiiGameWrapperJavaObject;

	// We can just copy all the pointers to methods
	stateTensorsShapeMethodID = other.stateTensorsShapeMethodID;
	moveTensorsShapeMethodID = other.moveTensorsShapeMethodID;
	stateTensorChannelNamesMethodID = other.stateTensorChannelNamesMethodID;
}

const std::array<int,3> & LudiiGameWrapper::StateTensorsShape() {
	if (not stateTensorsShape) {
		// Get our array of Java ints
		const jintArray jint_array = static_cast<jintArray>(jenv->CallObjectMethod(ludiiGameWrapperJavaObject, stateTensorsShapeMethodID));
		jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);

		// Create our C++ array of 3 ints
		stateTensorsShape = std::make_unique<std::array<int,3>>(std::array<int,3>{jints[0], jints[1], jints[2]});

		// Allow JVM to clean up memory now that we have our own ints
		jenv->ReleaseIntArrayElements(jint_array, jints, 0);
	}

	return *stateTensorsShape;
}

const std::array<int,3> & LudiiGameWrapper::MoveTensorsShape() {
	if (not moveTensorsShape) {
		// Get our array of Java ints
		const jintArray jint_array = static_cast<jintArray>(jenv->CallObjectMethod(ludiiGameWrapperJavaObject, moveTensorsShapeMethodID));
		jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);

		// Create our C++ array of 3 ints
		moveTensorsShape = std::make_unique<std::array<int,3>>(std::array<int,3>{jints[0], jints[1], jints[2]});

		// Allow JVM to clean up memory now that we have our own ints
		jenv->ReleaseIntArrayElements(jint_array, jints, 0);
	}

	return *moveTensorsShape;
}

const std::vector<std::string> LudiiGameWrapper::stateTensorChannelNames() {
	std::vector<std::string> channelNames;

	const jobjectArray java_arr = static_cast<jobjectArray>(jenv->CallObjectMethod(ludiiGameWrapperJavaObject, stateTensorChannelNamesMethodID));
	const int len = jenv->GetArrayLength(java_arr);

	for (int i = 0; i < len; ++i) {
		jstring jstr = (jstring) (jenv->GetObjectArrayElement(java_arr, i));

		// Convert Java string to C++ string
		const jsize jstr_len = jenv->GetStringUTFLength(jstr);
		const char* chars = jenv->GetStringUTFChars(jstr, (jboolean*) 0);
		std::string str(chars, jstr_len);

		channelNames.push_back(str);

		// Allow JVM to clean up memory
		jenv->ReleaseStringUTFChars(jstr, chars);
		jenv->DeleteLocalRef(jstr);
	}

	return channelNames;
}

}	// namespace Ludii
