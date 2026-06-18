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
 * @brief OpenCV reference for rocCV's Histogram operator.
 *
 * rocCV computes a 256-bin intensity histogram for each single-channel U8 image
 * in the batch, scattering each pixel into `histogram[bin] += 1`. cv::calcHist
 * performs the same per-image binning with 256 uniform bins over [0, 256).
 *
 * --- API differences vs. rocCV (kept as close as functionally possible) ---
 *
 * 1. Output element type. cv::calcHist always writes its result as CV_32F
 *    (float counts), whereas rocCV writes integer counts (S32 or U32). The bin
 *    arithmetic is identical and both store 256 * 4 bytes per image, so the
 *    memory footprint matches even though the element type differs. The two
 *    rocCV output-type variants (S32 / U32) are mirrored here purely so the
 *    benchmark names line up; the OpenCV work performed is identical for both.
 *
 * 2. No mask. The rocCV benchmark passes std::nullopt for the optional mask, so
 *    we pass an empty cv::Mat (count every pixel).
 *
 * 3. Per-image loop. rocCV processes the whole batch in one call; cv::calcHist
 *    is single-image, so we loop, matching the total work.
 */
static roccvbench::BenchmarkResults RunHistogramBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");

    std::vector<cv::Mat> mats = GenerateMats<uint8_t>(samples, width, height, inFormat);
    std::vector<cv::Mat> histograms(samples);

    const int channels[] = {0};
    const int histSize[] = {256};
    const float range[] = {0.0f, 256.0f};
    const float* histRange[] = {range};
    const cv::Mat noMask;

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    // 256 bins * 4 bytes per image, matching rocCV's S32/U32 output footprint.
    results.writtenMemoryBytes += static_cast<size_t>(samples) * histSize[0] * sizeof(int32_t);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::calcHist(&mats[i], 1, channels, noMask, histograms[i], 1, histSize, histRange, true, false);
        }
    });

    return results;
}

// outFormat is a label only: cv::calcHist always produces CV_32F, so the value is
// unused (passed as 0) and exists solely to mirror rocCV's S32/U32 variant names.
#define DEFINE_HISTOGRAM_BENCHMARK(name, inFormat, outLabel)                                                         \
    BENCHMARK_P(                                                                                                     \
        Histogram, name,                                                                                             \
        BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat), BENCH_PARAM_STR("outFormat", 0, outLabel))) { \
        return RunHistogramBenchmark(params);                                                                        \
    }

// Mirrors the rocCV histogram CPU benchmarks: single-channel U8 input, S32/U32 counts.
DEFINE_HISTOGRAM_BENCHMARK(OpenCV, CV_8UC1, "S32");
DEFINE_HISTOGRAM_BENCHMARK(OpenCV, CV_8UC1, "U32");
