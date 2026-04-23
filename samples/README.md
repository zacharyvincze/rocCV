# rocCV Samples

## Description
The rocCV samples demonstrate the use of the operator interfaces in C++ and Python to process images and construct image processing pipelines. The code shows how to set up tensors, load images into the tensors from OpenCV mat objects and then set up and run the operators.

## Dependencies
rocCV samples requires the OpenCV development library to read and write images.
```shell
apt install libopencv-dev   # For C++ samples
apt install python3-opencv  # For Python samples
```

## Operator Samples
1. Individual process in C++: bnd_box.cpp, center_crop.cpp, composite.cpp, copy_make_border.cpp, custom_crop.cpp, gamma_contrast.cpp, normalize.cpp, warp_perspective.cpp.
2. cropandresize - Crops and resizes the input image. This sample is designed to demonstrate a simple pipeline for multiple operators.
3. pipeline/multi_op_1.py: A pipeline of color conversion, cropping, bilateral filtering, bounding box drawing, rotation and resizing.

## Building and running the samples

The samples can be built either as part of an in-tree rocCV build or standalone against an installed rocCV.

### Option 1: In-tree build with rocCV
Build rocCV as described in the main README with the `-D SAMPLES=ON` flag set:
```shell
mkdir -p build && cd build
cmake -D SAMPLES=ON ../
cmake --build . --parallel
```
Sample binaries will be placed in `build/bin/samples`.

### Option 2: Standalone build against an installed rocCV
After installing rocCV (`sudo make install` from the rocCV build directory), the samples sources are copied to `${ROCM_PATH}/share/roccv/samples` (typically `/opt/rocm/share/roccv/samples`). Build them by pointing cmake at that directory from any writable build location:
```shell
mkdir -p ~/roccv-samples-build && cd ~/roccv-samples-build
cmake ${ROCM_PATH}/share/roccv/samples
cmake --build . --parallel
```
Sample binaries will be placed in `bin/samples` under the build directory.

Run any sample with the `-h` option for usage information.