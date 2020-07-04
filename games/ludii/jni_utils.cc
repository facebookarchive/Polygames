// inspired from https://gist.github.com/alexminnaar/90cf1ea3de45e79a1b14081d90d214b7

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
#include <string>

namespace Ludii {

JavaVM* JNIUtils::jvm = nullptr;
JNIEnv* JNIUtils::env = nullptr;
jint JNIUtils::res = 0;

JNIEnv *JNIUtils::GetEnv() { return env; }

void JNIUtils::InitJVM(std::string jar_location) {
	if (env != nullptr)
		return;		// We've already initialised the JVM

	std::cout << "intializing JVM" << std::endl;
	if (jar_location.empty()) jar_location = "ludii/Ludii.jar";
#ifdef JNI_VERSION_1_2
	JavaVMInitArgs vm_args;
	JavaVMOption options[1];
	std::string java_classpath = "-Djava.class.path=" + jar_location;
	char *c_classpath = strdup(java_classpath.c_str());
	options[0].optionString = c_classpath;
	vm_args.version = 0x00010002;
	vm_args.options = options;
	vm_args.nOptions = 1;
	vm_args.ignoreUnrecognized = JNI_TRUE;
	/* Create the Java VM */
	res = JNI_CreateJavaVM(&jvm, (void **)&env, &vm_args);
	free(c_classpath);
#else
	JDK1_1InitArgs vm_args;
	std::string classpath = vm_args.classpath + ";" + jar_location;
	char* c_classpath = strdup(java_classpath.c_str());
	vm_args.version = 0x00010001;
	JNI_GetDefaultJavaVMInitArgs(&vm_args);
	/* Append jar location to the default system class path */
	vm_args.classpath = c_classpath;
	/* Create the Java VM */
	res = JNI_CreateJavaVM(&jvm, &env, &vm_args);
	free(c_classpath);
#endif /* JNI_VERSION_1_2 */

	// Find our LudiiGameWrapper Java class
	ludiiGameWrapperClass = (jclass) env->NewGlobalRef(env->FindClass("utils/LudiiGameWrapper"));

	// Find our LudiiStateWrapper Java class
	ludiiStateWrapperClass = (jclass) env->NewGlobalRef(env->FindClass("utils/LudiiStateWrapper"));

	// Find the method ID for the static method giving us the Ludii versio
	ludiiVersionMethodID = env->GetStaticMethodID(ludiiGameWrapperClass, "ludiiVersion", "()Ljava/lang/String;");

	std::cout << "Using Ludii version " << LudiiVersion() << std::endl;
}

void JNIUtils::CloseJVM() {
	env->DeleteGlobalRef(ludiiStateWrapperClass);
	env->DeleteGlobalRef(ludiiGameWrapperClass);
	jvm->DestroyJavaVM();

	jvm = nullptr;
	env = nullptr;
	res = 0;
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
	jstring jstr = (jstring) (env->CallStaticObjectMethod(ludiiGameWrapperClass, ludiiVersionMethodID));
	const char* strReturn = env->GetStringUTFChars(jstr, (jboolean*) 0);
	const std::string str = strReturn;
	env->ReleaseStringUTFChars(jstr, strReturn);
	return str;
}

}  // namespace Ludii
