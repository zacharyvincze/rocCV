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

#include <cmath>
#include <opencv2/opencv.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "opencv_bench_helpers.hpp"

// Matches ROCCV_NORMALIZE_SCALE_IS_STDDEV in op_normalize.hpp. Defined locally so this
// translation unit does not need to pull in the rocCV headers.
#define BENCH_NORMALIZE_SCALE_IS_STDDEV 1

/**
 * @brief OpenCV reference for rocCV's Normalize operator.
 *
 * rocCV computes, per element:
 *
 *     out = (in - base) * scale * global_scale + shift
 *
 * or, when the scale tensor is interpreted as a standard deviation
 * (ROCCV_NORMALIZE_SCALE_IS_STDDEV):
 *
 *     out = (in - base) * (1 / sqrt(scale^2 + epsilon)) * global_scale + shift
 *
 * with a final saturating cast to the output type. This benchmark reproduces the
 * `[1, 1, 1, C]` parameter case used by the rocCV benchmark, i.e. one base/scale
 * value per channel, broadcast across N/H/W.
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. No single fused call. OpenCV has no operator matching this affine-per-channel
 *    formula (`cv::normalize` is unrelated — it rescales by a vector norm / value
 *    range over the whole array). The equivalent is expressed as two element-wise
 *    passes (multiply, then add-with-saturating-cast). rocCV fuses this into a
 *    single pass, so OpenCV touches the working buffer once more than rocCV does.
 *
 * 2. Hoisted parameters. rocCV resolves the per-channel multiplier (including the
 *    stddev `1/sqrt(...)`) once per row in its fast path. We fold base, scale,
 *    global_scale, shift, and the optional stddev inversion into a per-channel
 *    multiply `m` and add `a` on the host so the timed region performs the same
 *    `in * m + a` work per element:
 *        m_c = resolve(scale_c) * global_scale,   a_c = shift - base_c * m_c
 *    This mirrors rocCV's hoisting rather than re-deriving the multiplier per pixel.
 *
 * 3. Per-channel parameters via cv::Scalar. cv::Scalar holds up to 4 channels, which
 *    covers the C1/C3/C4 formats benchmarked here; rocCV accepts an arbitrary
 *    channel count through its parameter tensor.
 */
template <typename T>
static roccvbench::BenchmarkResults RunNormalizeBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");

    // Match the rocCV benchmark's invocation: plain scale (no stddev), unit global
    // scale, zero shift, epsilon = 1e-5.
    const float globalScale = 1.0f;
    const float shift = 0.0f;
    const float epsilon = 0.00001f;
    const uint32_t flags = 0;

    const int channels = CV_MAT_CN(inFormat);

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, outFormat);

    // Per-channel base/scale, matching rocCV's [1, 1, 1, C] parameter tensors.
    std::vector<float> baseVals = roccvbench::RandVector<float>(channels);
    std::vector<float> scaleVals = roccvbench::RandVector<float>(channels);

    // Fold base/scale/global_scale/shift (and the optional stddev inversion) into a
    // per-channel multiply `m` and add `a` so the timed loop does `in * m + a`.
    cv::Scalar mul = cv::Scalar::all(0.0);
    cv::Scalar add = cv::Scalar::all(0.0);
    for (int c = 0; c < channels; c++) {
        const float resolvedScale = (flags & BENCH_NORMALIZE_SCALE_IS_STDDEV)
                                        ? 1.0f / std::sqrt(scaleVals[c] * scaleVals[c] + epsilon)
                                        : scaleVals[c];
        mul[c] = static_cast<double>(resolvedScale * globalScale);
        add[c] = static_cast<double>(shift - baseVals[c] * resolvedScale * globalScale);
    }

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);
    // Account for the per-channel base/scale parameter reads (tiny, but mirrors the
    // rocCV benchmark which registers its base/scale tensors).
    results.readMemoryBytes += 2 * channels * sizeof(float);

    const int outType = CV_MAKETYPE(CV_MAT_DEPTH(outFormat), channels);

    cv::Mat tmp;
    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            // Pass 1: tmp = in * m  (promoted to float, the work type rocCV uses).
            cv::multiply(mats[i], mul, tmp, 1.0, CV_32F);
            // Pass 2: out = saturate_cast<outType>(tmp + a).
            cv::add(tmp, add, outputs[i], cv::noArray(), outType);
        }
    });
    return results;
}

#define DEFINE_NORMALIZE_BENCHMARK(name, T, inFormat, outFormat)                     \
    BENCHMARK_P(Normalize, name,                                                     \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),       \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat))) { \
        return RunNormalizeBenchmark<T>(params);                                     \
    }

// Mirrors the rocCV benchmark's data paths: RGB8 (the host-side reference),
// single-channel U8, and single-channel F32.
DEFINE_NORMALIZE_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3);
DEFINE_NORMALIZE_BENCHMARK(OpenCV, uint8_t, CV_8UC1, CV_8UC1);
DEFINE_NORMALIZE_BENCHMARK(OpenCV, float, CV_32FC1, CV_32FC1);
