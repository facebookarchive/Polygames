// inspired from
// https://gist.github.com/alexminnaar/90cf1ea3de45e79a1b14081d90d214b7

/*
Copyright (c) 2020 Alex Minnaar
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "jni_utils.h"

#include <cstring>
#include <iostream>
#include <stdio.h>
#include <string>

namespace Ludii {

JavaVM* JNIUtils::jvm = nullptr;
jint JNIUtils::res = 0;

thread_local JNIEnv* JNIUtils::env = nullptr;

JNIEnv* JNIUtils::GetEnv() {
  if (jvm == nullptr)
    return nullptr;

  if (env == nullptr) {
    JavaVMAttachArgs args = {JNI_VERSION_1_2, 0, 0};
    jvm->AttachCurrentThread((void**)&env, &args);
  }

  return env;
}

void JNIUtils::InitJVM(std::string jar_location) {
  if (jvm != nullptr)
    return;  // We've already initialised the JVM

  JNIEnv* env = nullptr;
  std::cout << "intializing JVM" << std::endl;
  if (jar_location.empty())
    jar_location = "ludii/Ludii.jar";

  // Check if we can actually access the JAR file
  if (FILE* file = fopen(jar_location.c_str(), "rb")) {
    // The Ludii.jar file seems to be there, so we're fine!
    fclose(file);
  } else {
    // Can't find the Ludii.jar file
    return;
  }

  //#define CHECK_JNI  // Uncomment this to run extra checks for JNI

#ifdef JNI_VERSION_1_2
  JavaVMInitArgs vm_args;

#ifdef CHECK_JNI
  const size_t num_jvm_args = 2;
#else
  const size_t num_jvm_args = 1;
#endif

  JavaVMOption options[num_jvm_args];
  std::string java_classpath = "-Djava.class.path=" + jar_location;
  options[0].optionString = java_classpath.data();

#ifdef CHECK_JNI
  std::string check_jni = "-Xcheck:jni";
  options[1].optionString = check_jni.data();
#endif

  vm_args.version = 0x00010002;
  vm_args.options = options;
  vm_args.nOptions = num_jvm_args;
  vm_args.ignoreUnrecognized = JNI_TRUE;
  /* Create the Java VM */
  res = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
#else
  JDK1_1InitArgs vm_args;
  std::string classpath = vm_args.classpath + ";" + jar_location;
  vm_args.version = 0x00010001;
  JNI_GetDefaultJavaVMInitArgs(&vm_args);
  /* Append jar location to the default system class path */
  vm_args.classpath = java_classpath.data();
  /* Create the Java VM */
  res = JNI_CreateJavaVM(&jvm, &env, &vm_args);
#endif /* JNI_VERSION_1_2 */

  // Find our LudiiGameWrapper Java class
  ludiiGameWrapperClass =
      (jclass)env->NewGlobalRef(env->FindClass("utils/LudiiGameWrapper"));
  CheckJniException(env);

  // Find our LudiiStateWrapper Java class
  ludiiStateWrapperClass =
      (jclass)env->NewGlobalRef(env->FindClass("utils/LudiiStateWrapper"));
  CheckJniException(env);

  // Find the method ID for the static method giving us the Ludii versio
  ludiiVersionMethodID = env->GetStaticMethodID(
      ludiiGameWrapperClass, "ludiiVersion", "()Ljava/lang/String;");
  CheckJniException(env);

  std::cout << "Using Ludii version " << LudiiVersion() << std::endl;
}

void JNIUtils::CloseJVM() {
  JNIEnv* env = JNIUtils::GetEnv();

  if (env != nullptr) {
    env->DeleteGlobalRef(ludiiStateWrapperClass);
    env->DeleteGlobalRef(ludiiGameWrapperClass);
    jvm->DestroyJavaVM();

    jvm = nullptr;
    res = 0;
  }
}

// These will be assigned proper values by InitJVM() call
jclass JNIUtils::ludiiGameWrapperClass = nullptr;
jclass JNIUtils::ludiiStateWrapperClass = nullptr;
jmethodID JNIUtils::ludiiVersionMethodID = nullptr;

jclass JNIUtils::LudiiGameWrapperClass() {
  return ludiiGameWrapperClass;
}

jclass JNIUtils::LudiiStateWrapperClass() {
  return ludiiStateWrapperClass;
}

const std::string JNIUtils::LudiiVersion() {
  JNIEnv* env = JNIUtils::GetEnv();
  jstring jstr = (jstring)(
      env->CallStaticObjectMethod(ludiiGameWrapperClass, ludiiVersionMethodID));
  CheckJniException(env);
  const char* strReturn = env->GetStringUTFChars(jstr, (jboolean*)0);
  CheckJniException(env);
  const std::string str = strReturn;
  env->ReleaseStringUTFChars(jstr, strReturn);
  CheckJniException(env);
  env->DeleteLocalRef(jstr);
  CheckJniException(env);
  return str;
}

}  // namespace Ludii
