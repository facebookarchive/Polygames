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

#include "ludii_game_wrapper.h"
#include "jni_utils.h"

namespace Ludii {

// NOTE: String descriptions of signatures of Java methods can be found by
// navigating to directory containing the .class files and using:
//
// javap -s <ClassName.class>

LudiiGameWrapper::LudiiGameWrapper(const std::string lud_path) {

  JNIEnv* jenv = JNIUtils::GetEnv();
  jclass ludiiGameWrapperClass = JNIUtils::LudiiGameWrapperClass();

  // Find the LudiiGameWrapper Java constructor
  jmethodID ludiiGameWrapperConstructor = jenv->GetMethodID(
      ludiiGameWrapperClass, "<init>", "(Ljava/lang/String;)V");
  JNIUtils::CheckJniException(jenv);

  // Convert our lud path into a Java string
  jstring java_lud_path = jenv->NewStringUTF(lud_path.c_str());
  JNIUtils::CheckJniException(jenv);

  // Call our Java constructor to instantiate new object
  jobject local_ref = jenv->NewObject(
      ludiiGameWrapperClass, ludiiGameWrapperConstructor, java_lud_path);
  JNIUtils::CheckJniException(jenv);
  ludiiGameWrapperJavaObject = jenv->NewGlobalRef(local_ref);
  jenv->DeleteLocalRef(local_ref);

  // Find method IDs for the two tensor shape Java methods that we may be
  // calling frequently
  stateTensorsShapeMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorsShape", "()[I");
  JNIUtils::CheckJniException(jenv);
  moveTensorsShapeMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "moveTensorsShape", "()[I");
  JNIUtils::CheckJniException(jenv);

  // Find the method ID for the stateTensorChannelNames() method in Java
  stateTensorChannelNamesMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorChannelNames",
                        "()[Ljava/lang/String;");
  JNIUtils::CheckJniException(jenv);

  // Clean up memory
  jenv->DeleteLocalRef(java_lud_path);
  JNIUtils::CheckJniException(jenv);
}

LudiiGameWrapper::LudiiGameWrapper(
    const std::string lud_path, const std::vector<std::string> game_options) {

  JNIEnv* jenv = JNIUtils::GetEnv();
  jclass ludiiGameWrapperClass = JNIUtils::LudiiGameWrapperClass();

  // Find the LudiiGameWrapper Java constructor (with extra argument for
  // options)
  jmethodID ludiiGameWrapperConstructor =
      jenv->GetMethodID(ludiiGameWrapperClass, "<init>",
                        "(Ljava/lang/String;[Ljava/lang/String;)V");
  JNIUtils::CheckJniException(jenv);

  // Convert our lud path into a Java string
  jstring java_lud_path = jenv->NewStringUTF(lud_path.c_str());
  JNIUtils::CheckJniException(jenv);

  // Convert vector of game options into array of Java strings
  const jobjectArray java_game_options = (jobjectArray)jenv->NewObjectArray(
      game_options.size(), jenv->FindClass("java/lang/String"), nullptr);
  JNIUtils::CheckJniException(jenv);
  for (size_t i = 0; i < game_options.size(); ++i) {
    jenv->SetObjectArrayElement(
        java_game_options, i, jenv->NewStringUTF(game_options[i].c_str()));
    JNIUtils::CheckJniException(jenv);
  }

  // Call our Java constructor to instantiate new object
  jobject local_ref =
      jenv->NewObject(ludiiGameWrapperClass, ludiiGameWrapperConstructor,
                      java_lud_path, java_game_options);
  JNIUtils::CheckJniException(jenv);
  ludiiGameWrapperJavaObject = jenv->NewGlobalRef(local_ref);
  jenv->DeleteLocalRef(local_ref);

  // Find method IDs for the two tensor shape Java methods that we may be
  // calling frequently
  stateTensorsShapeMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorsShape", "()[I");
  JNIUtils::CheckJniException(jenv);
  moveTensorsShapeMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "moveTensorsShape", "()[I");
  JNIUtils::CheckJniException(jenv);

  // Find the method ID for the stateTensorChannelNames() method in Java
  stateTensorChannelNamesMethodID =
      jenv->GetMethodID(ludiiGameWrapperClass, "stateTensorChannelNames",
                        "()[Ljava/lang/String;");
  JNIUtils::CheckJniException(jenv);

  // Clean up memory
  jenv->DeleteLocalRef(java_lud_path);
  jenv->DeleteLocalRef(java_game_options);  // TODO see if we also need to clean
                                            // elements of array first?

  // TODO also handle the Game object caching with options
}

LudiiGameWrapper::LudiiGameWrapper(LudiiGameWrapper const& other) {

  JNIEnv* jenv = JNIUtils::GetEnv();

  // We can just copy the pointer to the same Java Game object
  ludiiGameWrapperJavaObject =
      jenv->NewGlobalRef(other.ludiiGameWrapperJavaObject);
  JNIUtils::CheckJniException(jenv);

  // We can just copy all the pointers to methods
  stateTensorsShapeMethodID = other.stateTensorsShapeMethodID;
  moveTensorsShapeMethodID = other.moveTensorsShapeMethodID;
  stateTensorChannelNamesMethodID = other.stateTensorChannelNamesMethodID;
}

LudiiGameWrapper& LudiiGameWrapper::operator=(LudiiGameWrapper const& other) {
  JNIEnv* jenv = JNIUtils::GetEnv();

  // We can just copy the pointer to the same Java Game object
  ludiiGameWrapperJavaObject =
      jenv->NewGlobalRef(other.ludiiGameWrapperJavaObject);
  JNIUtils::CheckJniException(jenv);

  // We can just copy all the pointers to methods
  stateTensorsShapeMethodID = other.stateTensorsShapeMethodID;
  moveTensorsShapeMethodID = other.moveTensorsShapeMethodID;
  stateTensorChannelNamesMethodID = other.stateTensorChannelNamesMethodID;

  return *this;
}

LudiiGameWrapper::~LudiiGameWrapper() {
  JNIEnv* jenv = JNIUtils::GetEnv();
  if (jenv) {
    jenv->DeleteGlobalRef(ludiiGameWrapperJavaObject);
  }
}

const std::array<int, 3>& LudiiGameWrapper::StateTensorsShape() {
  if (not stateTensorsShape) {
    JNIEnv* jenv = JNIUtils::GetEnv();

    // Get our array of Java ints
    const jintArray jint_array = static_cast<jintArray>(jenv->CallObjectMethod(
        ludiiGameWrapperJavaObject, stateTensorsShapeMethodID));
    JNIUtils::CheckJniException(jenv);
    jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);
    JNIUtils::CheckJniException(jenv);

    // Create our C++ array of 3 ints
    stateTensorsShape = std::make_unique<std::array<int, 3>>(
        std::array<int, 3>{jints[0], jints[1], jints[2]});

    // Allow JVM to clean up memory now that we have our own ints
    jenv->ReleaseIntArrayElements(jint_array, jints, 0);
    jenv->DeleteLocalRef(jint_array);
  }

  return *stateTensorsShape;
}

const std::array<int, 3>& LudiiGameWrapper::MoveTensorsShape() {
  if (not moveTensorsShape) {
    JNIEnv* jenv = JNIUtils::GetEnv();

    // Get our array of Java ints
    const jintArray jint_array = static_cast<jintArray>(jenv->CallObjectMethod(
        ludiiGameWrapperJavaObject, moveTensorsShapeMethodID));
    JNIUtils::CheckJniException(jenv);
    jint* jints = jenv->GetIntArrayElements(jint_array, nullptr);
    JNIUtils::CheckJniException(jenv);

    // Create our C++ array of 3 ints
    moveTensorsShape = std::make_unique<std::array<int, 3>>(
        std::array<int, 3>{jints[0], jints[1], jints[2]});

    // Allow JVM to clean up memory now that we have our own ints
    jenv->ReleaseIntArrayElements(jint_array, jints, 0);
    jenv->DeleteLocalRef(jint_array);
  }

  return *moveTensorsShape;
}

const std::vector<std::string> LudiiGameWrapper::stateTensorChannelNames() {
  JNIEnv* jenv = JNIUtils::GetEnv();
  std::vector<std::string> channelNames;

  const jobjectArray java_arr =
      static_cast<jobjectArray>(jenv->CallObjectMethod(
          ludiiGameWrapperJavaObject, stateTensorChannelNamesMethodID));
  JNIUtils::CheckJniException(jenv);
  const int len = jenv->GetArrayLength(java_arr);
  JNIUtils::CheckJniException(jenv);

  for (int i = 0; i < len; ++i) {
    jstring jstr = (jstring)(jenv->GetObjectArrayElement(java_arr, i));
    JNIUtils::CheckJniException(jenv);

    // Convert Java string to C++ string
    const jsize jstr_len = jenv->GetStringUTFLength(jstr);
    JNIUtils::CheckJniException(jenv);
    const char* chars = jenv->GetStringUTFChars(jstr, (jboolean*)0);
    JNIUtils::CheckJniException(jenv);
    std::string str(chars, jstr_len);

    channelNames.push_back(str);

    // Allow JVM to clean up memory
    jenv->ReleaseStringUTFChars(jstr, chars);
    jenv->DeleteLocalRef(jstr);
  }

  jenv->DeleteLocalRef(java_arr);

  return channelNames;
}

}  // namespace Ludii
