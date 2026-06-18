/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "core/detail/casting.hpp"

namespace Kernels {
namespace Host {
template <typename SrcWrapper, typename MaskWrapper, typename DstWrapper>
inline void composite(SrcWrapper foreground, SrcWrapper background, MaskWrapper mask, DstWrapper output) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using mask_type = typename MaskWrapper::ValueType;
    using work_type = MakeType<float, NumElements<src_type>>;

    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // Single-sourced per-pixel formula shared by both the contiguous fast path and the strided fallback.
    auto computePixel = [&](const src_type& fg, const src_type& bg, const mask_type& m) -> dst_type {
        // Range cast all input values to float to avoid overflowing values and keep them in the same range.
        auto maskFactor = RangeCast<float1>(m);
        auto fgVal = RangeCast<work_type>(fg);
        auto bgVal = RangeCast<work_type>(bg);

        // Lerp in FMA-friendly form: one multiply + one add per channel instead of two multiplies.
        work_type result = bgVal + (fgVal - bgVal) * maskFactor.x;

        // If number of channels in output is 4, ensure that the last channel (alpha in this case) is always fully on.
        if constexpr (NumElements<dst_type> == 4) {
            return RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, 1.0f});
        } else {
            return RangeCast<dst_type>(result);
        }
    };

    // Fast path is only valid when every row is densely packed; otherwise [x] indexing would skip the row stride.
    const bool contiguous = foreground.isRowContiguous() && background.isRowContiguous() && mask.isRowContiguous() &&
                            output.isRowContiguous();

#pragma omp parallel for collapse(2) schedule(static)
    for (int batch = 0; batch < batches; batch++) {
        for (int y = 0; y < height; y++) {
            if (contiguous) {
                // Packed rows become unit-stride, vectorizable walks over raw row pointers.
                const src_type* __restrict__ fgRow = &foreground.at(batch, y, 0, 0);
                const src_type* __restrict__ bgRow = &background.at(batch, y, 0, 0);
                const mask_type* __restrict__ maskRow = &mask.at(batch, y, 0, 0);
                dst_type* __restrict__ outRow = &output.at(batch, y, 0, 0);
                for (int x = 0; x < width; x++) {
                    outRow[x] = computePixel(fgRow[x], bgRow[x], maskRow[x]);
                }
            } else {
                for (int x = 0; x < width; x++) {
                    output.at(batch, y, x, 0) = computePixel(foreground.at(batch, y, x, 0),
                                                             background.at(batch, y, x, 0), mask.at(batch, y, x, 0));
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels