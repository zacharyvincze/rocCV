/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "core/detail/casting.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Host {
template <typename SrcWrapper, typename DstWrapper>
void gamma_contrast(SrcWrapper input, DstWrapper output, float gamma) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;
    using work_type = MakeType<float, NumElements<src_type>>;
    constexpr int kChannels = NumElements<dst_type>;

    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // Each pixel is independent, so packed rows become unit-stride, vectorizable walks.
    const bool contiguous = input.isRowContiguous() && output.isRowContiguous();

    // Shared loop skeleton: OMP-parallel over batch + row with a contiguous fast path and a strided fallback. The
    // per-pixel transform is a template parameter, so there is no indirect-call overhead in the hot loop.
    auto runKernel = [&](auto perPixel) {
    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
        for (int b = 0; b < batches; b++) {
            for (int y = 0; y < height; y++) {
                if (contiguous) {
                    const src_type* __restrict__ inRow = &input.at(b, y, 0, 0);
                    dst_type* __restrict__ outRow = &output.at(b, y, 0, 0);
                    for (int x = 0; x < width; x++) outRow[x] = perPixel(inRow[x]);
                } else {
                    for (int x = 0; x < width; x++) output.at(b, y, x, 0) = perPixel(input.at(b, y, x, 0));
                }
            }
        }
    };

    if constexpr (std::is_same_v<base_type, uint8_t>) {
        // U8 channels have only 256 possible values, so the whole pow() collapses into a 256-entry lookup table built
        // once and indexed per pixel. Reusing the same scalar range-cast + powf the direct path uses keeps results
        // byte-for-byte identical.
        std::array<base_type, 256> lut;
        for (int v = 0; v < 256; v++) {
            float norm = ScalarRangeCast<float>(static_cast<base_type>(v));
            lut[v] = ScalarRangeCast<base_type>(powf(norm, gamma));
        }

        runKernel([&](src_type in) -> dst_type {
            dst_type out;
            for (int i = 0; i < kChannels; i++) {
                // Preserve the alpha channel (4th) unchanged, matching the direct path.
                if (kChannels == 4 && i == 3) {
                    GetElement(out, i) = static_cast<base_type>(GetElement(in, i));
                } else {
                    GetElement(out, i) = lut[GetElement(in, i)];
                }
            }
            return out;
        });
    } else {
        // Wide / floating-point types: compute pow() per pixel directly.
        runKernel([&](src_type in) -> dst_type {
            work_type inVal = RangeCast<work_type>(in);
            work_type result = math::vpowf(inVal, gamma);
            if constexpr (kChannels == 4) {
                // Preserve the alpha channel (input value), matching the original behavior.
                return RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, inVal.w});
            } else {
                return RangeCast<dst_type>(result);
            }
        });
    }
}
}  // namespace Host
}  // namespace Kernels