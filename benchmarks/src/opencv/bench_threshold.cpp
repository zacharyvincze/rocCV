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
 * @brief OpenCV reference for rocCV's Threshold operator.
 *
 * rocCV applies, per element (THRESH_BINARY): `out = (in > thresh) ? maxVal : 0`,
 * with a per-image threshold and maxVal supplied through small N-length tensors.
 * cv::threshold implements the identical comparison and applies it across all
 * channels of a multi-channel image, matching rocCV's per-channel loop.
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. Per-image parameters: rocCV reads thresh/maxVal from F64 parameter tensors
 *    (one value per image in the batch). cv::threshold takes scalar `thresh` and
 *    `maxval` arguments, so we issue one call per image with that image's value.
 *
 * 2. Parameter RNG: rocCV fills its parameter tensors with rocRAND (uniform in
 *    [0, 1)); here we draw from the suite's std::mt19937 RandVector (also uniform
 *    in [0, 1)). The distribution matches but the exact values differ. Because
 *    the threshold values (< 1) sit well below the U8 input range, the branch
 *    outcome is data-independent in practice, so this does not bias timing.
 *
 * 3. Threshold-type mapping: rocCV's eThresholdType {THRESH_BINARY,
 *    THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV} map
 *    one-to-one onto cv::{THRESH_BINARY, ...}. The CPU benchmark mirrors rocCV's
 *    only registered CPU case (THRESH_BINARY).
 */
template <typename T>
static roccvbench::BenchmarkResults RunThresholdBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    int thresholdType = roccvbench::GetParamValue<int>(params, "thresholdType");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, outFormat);

    // Per-image thresh/maxVal, mirroring rocCV's N-length parameter tensors. A
    // single draw of 2*samples values is split so thresh and maxVal differ.
    std::vector<double> randParams = roccvbench::RandVector<double>(samples * 2);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);
    // Account for the per-image thresh/maxVal reads (mirrors the rocCV benchmark,
    // which registers its thresh/maxVal tensors as F64).
    results.readMemoryBytes += 2 * samples * sizeof(double);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            const double thresh = randParams[i];
            const double maxVal = randParams[samples + i];
            cv::threshold(mats[i], outputs[i], thresh, maxVal, thresholdType);
        }
    });

    return results;
}

#define DEFINE_THRESHOLD_BENCHMARK(name, T, inFormat, outFormat, thresholdType)                                    \
    BENCHMARK_P(Threshold, name,                                                                                   \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                     \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat),                                  \
                             BENCH_PARAM_STR("thresholdType", static_cast<int>(thresholdType), #thresholdType))) { \
        return RunThresholdBenchmark<T>(params);                                                                   \
    }

// Mirrors the rocCV threshold CPU benchmark: RGB8, binary threshold.
DEFINE_THRESHOLD_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::THRESH_BINARY);
