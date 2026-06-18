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

/**
 * @brief OpenCV reference for rocCV's GammaContrast operator.
 *
 * rocCV computes, per element, `out = RangeCast(pow(RangeCast(in), gamma))`. For
 * U8 the RangeCasts normalize to [0, 1] and back, so the effective transform is
 * `out = 255 * (in / 255) ^ gamma` with a saturating cast. OpenCV has no gamma
 * operator; the idiomatic equivalent for an 8-bit image is a 256-entry lookup
 * table applied with cv::LUT, which for U8 produces bit-identical results because
 * there are only 256 possible inputs.
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. LUT vs. per-pixel pow. This is the most important difference. rocCV
 *    evaluates `powf` for every pixel, whereas the OpenCV idiom precomputes the
 *    transform once into a 256-entry table and then does table lookups. The
 *    outputs are identical for U8, but cv::LUT is dramatically cheaper per pixel,
 *    so OpenCV will look much faster here for a reason that is algorithmic, not
 *    a difference in raw throughput. The (tiny, constant) table-build cost is
 *    intentionally left outside the timed region, matching how a user would
 *    apply a fixed gamma to a batch.
 *
 *    A faithful per-pixel reproduction would instead normalize to a float Mat,
 *    call cv::pow, and scale back, at the cost of being non-idiomatic OpenCV.
 *    The LUT path is chosen as the canonical "how OpenCV does gamma" reference.
 *
 * 2. The LUT only models the integer (U8) path. rocCV also supports F32, where
 *    no LUT is possible; that case is not benchmarked here because rocCV only
 *    registers an RGB8 CPU gamma benchmark.
 */
template <typename T>
static roccvbench::BenchmarkResults RunGammaContrastBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    float gamma = roccvbench::GetParamValue<float>(params, "gamma");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, outFormat);

    // Precompute the gamma curve once: out = 255 * (in / 255) ^ gamma. This
    // mirrors rocCV's normalized-domain power, just hoisted into a table.
    cv::Mat lut(1, 256, CV_8U);
    uchar* lutData = lut.ptr();
    for (int i = 0; i < 256; i++) {
        lutData[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0f, gamma) * 255.0f);
    }

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::LUT(mats[i], lut, outputs[i]);
        }
    });

    return results;
}

#define DEFINE_GAMMA_CONTRAST_BENCHMARK(name, T, inFormat, outFormat, gamma)                                      \
    BENCHMARK_P(GammaContrast, name,                                                                              \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                    \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat), BENCH_PARAM("gamma", gamma))) { \
        return RunGammaContrastBenchmark<T>(params);                                                              \
    }

// Mirrors the rocCV gamma contrast CPU benchmark: RGB8, gamma = 2.2.
DEFINE_GAMMA_CONTRAST_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, 2.2f);
