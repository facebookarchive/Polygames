// strongly inspired from https://gist.github.com/alexminnaar/90cf1ea3de45e79a1b14081d90d214b7
// might need something like export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/
// maybe also install jvm



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

#pragma once

#include <cstring>
#include <string>

#include <jni.h>  // NOLINT

namespace Ludii {

class JNIUtils {
 public:
  static JNIEnv *GetEnv();

  static void InitJVM(std::string jar_location);
  static void CloseJVM();

  static jclass LudiiGameWrapperClass();
  static jclass LudiiStateWrapperClass();

  /**
   * @return A string description of the version of Ludii that we're working with.
   */
  static const std::string LudiiVersion();

 private:
  static JavaVM *jvm;
  static JNIEnv *env;
  static jint res;

  /** Our LudiiGameWrapper class in Java */
  static jclass ludiiGameWrapperClass;

  /** Our LudiiStateWrapper class in Java */
  static jclass ludiiStateWrapperClass;

  /** Method ID for the ludiiVersion() method in Java */
  static jmethodID ludiiVersionMethodID;
};

}  // namespace Ludii


