# ncnn-android-rvm

![download](https://img.shields.io/github/downloads/nihui/ncnn-android-rvm/total.svg)

RVM

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

https://github.com/nihui/vulkan.turnip.so  (mesa turnip driver)

## android apk file download
https://github.com/nihui/ncnn-android-rvm/releases/latest

## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
https://github.com/nihui/vulkan.turnip.so

* Download mesa-turnip-android-XYZ.zip
* Create directory **app/src/main/jniLibs/arm64-v8a** if not exists
* Extract `vulkan.turnip.so` from mesa-turnip-android-XYZ.zip into **app/src/main/jniLibs/arm64-v8a**

### step4
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## screenshot
![](screenshot0.jpg)

## guidelines for converting RVM models

TBA
