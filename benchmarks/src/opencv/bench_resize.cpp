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
 * @brief OpenCV reference for rocCV's Resize operator.
 *
 * rocCV's host resize maps each destination pixel back to source space with the
 * half-pixel convention `srcX = (x + 0.5) * scaleX - 0.5` (scaleX = inW / outW)
 * and samples the source through an InterpolationWrapper backed by a REPLICATE
 * border. cv::resize uses the exact same half-pixel center mapping for
 * INTER_LINEAR / INTER_CUBIC and also clamps out-of-range taps to the edge
 * (replicate), so the two are functionally equivalent.
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. Interpolation mode mapping: rocCV's eInterpolationType {NEAREST, LINEAR,
 *    CUBIC} map to cv::{INTER_NEAREST, INTER_LINEAR, INTER_CUBIC}. The CPU
 *    benchmark below mirrors rocCV's only registered CPU case (LINEAR).
 *
 * 2. Fixed-point vs. float: for 8-bit inputs OpenCV's INTER_LINEAR uses a
 *    fixed-point (integer SIMD) path, whereas rocCV's host kernel interpolates
 *    in float. Results are within rounding tolerance but not bit-identical, and
 *    OpenCV's fixed-point path is a meaningful performance advantage to keep in
 *    mind when reading the numbers.
 */
template <typename T>
static roccvbench::BenchmarkResults RunResizeBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    int interpolation = roccvbench::GetParamValue<int>(params, "interpolation");
    int scale = roccvbench::GetParamValue<int>(params, "scale");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width * scale, height * scale, outFormat);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    const cv::Size dsize(width * scale, height * scale);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::resize(mats[i], outputs[i], dsize, 0.0, 0.0, interpolation);
        }
    });

    return results;
}

#define DEFINE_RESIZE_BENCHMARK(name, T, inFormat, outFormat, interpolation, scale)                             \
    BENCHMARK_P(Resize, name,                                                                                   \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                  \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat),                               \
                             BENCH_PARAM_STR("interpolation", static_cast<int>(interpolation), #interpolation), \
                             BENCH_PARAM("scale", scale))) {                                                    \
        return RunResizeBenchmark<T>(params);                                                                   \
    }

// Mirrors the rocCV resize CPU benchmark: RGB8, linear interpolation, 2x upscale.
DEFINE_RESIZE_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::INTER_LINEAR, 2);
