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
 * @brief OpenCV reference for rocCV's BilateralFilter operator.
 *
 * The two implementations share the same mathematical formulation. For each
 * pixel both compute a weighted average of neighbors within a circular radius,
 * weighting by `exp(spatialDist^2 * spaceCoeff + colorDist^2 * colorCoeff)`,
 * where colorDist is the L1 (sum-of-absolute-channel-difference) distance,
 * `spaceCoeff = -1 / (2 * sigmaSpace^2)`, and `colorCoeff = -1 / (2 * sigmaColor^2)`.
 * OpenCV's bilateralFilter_8u uses exactly this L1 color distance and the same
 * Gaussian coefficients, so results are very close.
 *
 * Parameter handling is matched to rocCV's:
 *   - diameter <= 0  -> both derive radius = round(sigmaSpace * 1.5), giving the
 *     same effective window. OpenCV is given d <= 0 so it auto-derives the same
 *     diameter from sigmaSpace.
 *   - sigmaColor <= 0 -> rocCV clamps it to 1.0; we pass 1.0 explicitly because
 *     OpenCV does not perform that clamp.
 *   - sigmaSpace and border type are passed through unchanged
 *     (BORDER_TYPE_REFLECT -> cv::BORDER_REFLECT).
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. Weight tables vs. live exp. OpenCV precomputes a color-weight LUT (indexed
 *    by integer L1 distance) and a per-neighbor space-weight table, then does
 *    table lookups in the inner loop. rocCV evaluates `exp` for every neighbor of
 *    every pixel. The outputs agree to rounding; OpenCV's tabulated path is a
 *    performance advantage to keep in mind.
 *
 * 2. sigmaColor clamp is reproduced on the host (see above) so both filters use
 *    the same effective sigmaColor of 1.0.
 */
template <typename T>
static roccvbench::BenchmarkResults RunBilateralFilterBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    int diameter = roccvbench::GetParamValue<int>(params, "diameter");
    float sigmaColor = roccvbench::GetParamValue<float>(params, "sigmaColor");
    float sigmaSpace = roccvbench::GetParamValue<float>(params, "sigmaSpace");
    int border = roccvbench::GetParamValue<int>(params, "border");

    // Reproduce rocCV's parameter clamping so both filters use the same effective
    // sigmas / window. rocCV clamps non-positive sigmas to 1.0; OpenCV does not.
    if (sigmaColor <= 0.0f) sigmaColor = 1.0f;
    if (sigmaSpace <= 0.0f) sigmaSpace = 1.0f;

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, outFormat);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::bilateralFilter(mats[i], outputs[i], diameter, sigmaColor, sigmaSpace, border);
        }
    });

    return results;
}

#define DEFINE_BILATERAL_FILTER_BENCHMARK(name, T, inFormat, outFormat, diameter, sigmaColor, sigmaSpace, border)    \
    BENCHMARK_P(BilateralFilter, name,                                                                               \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                       \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat), BENCH_PARAM("diameter", diameter), \
                             BENCH_PARAM("sigmaColor", sigmaColor), BENCH_PARAM("sigmaSpace", sigmaSpace),           \
                             BENCH_PARAM_STR("border", static_cast<int>(border), #border))) {                        \
        return RunBilateralFilterBenchmark<T>(params);                                                               \
    }

// Mirrors the rocCV bilateral filter CPU benchmark: RGB8, auto diameter,
// sigmaColor -1 (clamped to 1.0), sigmaSpace 1.2, reflect border.
DEFINE_BILATERAL_FILTER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, -1, -1.0f, 1.2f, cv::BORDER_REFLECT);
