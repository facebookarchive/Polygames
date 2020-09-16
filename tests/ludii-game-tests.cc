/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ludii/jni_utils.h>
#include <ludii/ludii_game_wrapper.h>

#include <gtest/gtest.h>
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

extern JNIEnv * JNI_ENV;

TEST(LudiiGameGroup, jni_init_1) {

    if (JNI_ENV) {

        using namespace std;
        auto env = JNI_ENV;

        //find the GameLoader class
        jclass gameLoader = env->FindClass("player/GameLoader");

        if(gameLoader == NULL) {
            cout << "Could not load class!" << endl;
            return;
        }

        //find the listGames method
        jmethodID mid = env->GetStaticMethodID(gameLoader,"listGames","()[Ljava/lang/String;");

        if(mid == NULL) {
            cout << "Could not load method!" << endl;
            return;
        }

        //execute the listGames method
        jobjectArray stringArray = (jobjectArray) env->CallStaticObjectMethod(gameLoader,mid);

        if(stringArray == NULL) {
            cout << "Could not load object!" << endl;
            return;
        }

        //print the result of listGames
        int stringCount = env->GetArrayLength(stringArray);

        for (int i=0; i<stringCount; i++) {
            //get array element and convert it from jstrng
            jstring string = (jstring) (env->GetObjectArrayElement(stringArray, i));
            const char *rawString = env->GetStringUTFChars(string, 0);

            cout<<rawString<<endl;

            env->ReleaseStringUTFChars(string, rawString);
        }
    }
}

TEST(LudiiGameGroup, game_init_1) {
    if (JNI_ENV) {
        ASSERT_TRUE(JNI_ENV != 0);
        Ludii::LudiiGameWrapper game("/lud/board/space/line/Connect6.lud");
    }
}

