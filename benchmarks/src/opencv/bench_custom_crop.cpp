/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <opencv2/opencv.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "opencv_bench_helpers.hpp"

/**
 * @brief OpenCV reference for rocCV's CustomCrop operator.
 *
 * rocCV copies a `[x, y, width, height]` sub-rectangle of each input image into a
 * tightly packed output tensor sized to the crop. The OpenCV equivalent takes a
 * cv::Rect ROI on the source and copies it into a freshly allocated destination
 * Mat with `copyTo`, which performs the same per-row memcpy work.
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. ROI vs. copy: a bare `src(rect)` only builds a view header (no data
 *    movement). rocCV materializes a packed output buffer, so we call `copyTo`
 *    to perform the equivalent copy and time that. This is the closest match to
 *    the work rocCV's kernel actually does.
 *
 * 2. Crop rectangle uses {x, y, width, height} = {150, 50, 400, 300}, matching
 *    the Box_t used by the rocCV benchmark.
 */
template <typename T>
static roccvbench::BenchmarkResults RunCustomCropBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    int cropX = roccvbench::GetParamValue<int>(params, "cropX");
    int cropY = roccvbench::GetParamValue<int>(params, "cropY");
    int cropWidth = roccvbench::GetParamValue<int>(params, "cropWidth");
    int cropHeight = roccvbench::GetParamValue<int>(params, "cropHeight");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, cropWidth, cropHeight, outFormat);

    const cv::Rect cropRect(cropX, cropY, cropWidth, cropHeight);

    // The kernel reads only the cropped region from the input, so register the
    // output-sized footprint for reads as well (mirrors the rocCV benchmark).
    RegisterMemoryUsage(outputs, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            mats[i](cropRect).copyTo(outputs[i]);
        }
    });

    return results;
}

#define DEFINE_CUSTOM_CROP_BENCHMARK(name, T, inFormat, outFormat, cropX, cropY, cropWidth, cropHeight)        \
    BENCHMARK_P(CustomCrop, name,                                                                              \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                 \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat), BENCH_PARAM("cropX", cropX), \
                             BENCH_PARAM("cropY", cropY), BENCH_PARAM("cropWidth", cropWidth),                 \
                             BENCH_PARAM("cropHeight", cropHeight))) {                                         \
        return RunCustomCropBenchmark<T>(params);                                                              \
    }

// Mirrors the rocCV custom crop CPU benchmarks: RGB8, RGBA8, single-channel U8.
DEFINE_CUSTOM_CROP_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, 150, 50, 400, 300);
DEFINE_CUSTOM_CROP_BENCHMARK(OpenCV, uint8_t, CV_8UC4, CV_8UC4, 150, 50, 400, 300);
DEFINE_CUSTOM_CROP_BENCHMARK(OpenCV, uint8_t, CV_8UC1, CV_8UC1, 150, 50, 400, 300);
