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

#include <stdint.h>

#include <cstring>

namespace Kernels {
namespace Host {
template <typename SrcDesc, typename DstDesc>
void copy_make_border(SrcDesc src, DstDesc dst, int32_t top, int32_t left) {
    using T = typename DstDesc::ValueType;

    const int batches = static_cast<int>(dst.batches());
    const int dstH = static_cast<int>(dst.height());
    const int dstW = static_cast<int>(dst.width());
    const int srcH = static_cast<int>(src.height());
    const int srcW = static_cast<int>(src.width());

    // The interior columns [left, left+srcW) of an interior row are a verbatim copy of the source row; only the border
    // columns (and full border rows) need the more expensive border-mode remap. Grab a border-unaware view so packed
    // rows can be bulk-copied instead of walked one bounds-checked pixel at a time.
    auto& srcImg = src.inner();
    const bool contiguous = srcImg.isRowContiguous() && dst.isRowContiguous();

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batches; n++) {
        for (int y = 0; y < dstH; y++) {
            const int srcY = y - top;
            const bool interiorRow = (srcY >= 0 && srcY < srcH);

            // Fast path: an interior row with packed storage splits into [left border | memcpy interior | right
            // border].
            if (interiorRow && contiguous) {
                T* __restrict__ outRow = &dst.at(n, y, 0, 0);
                const T* __restrict__ inRow = &srcImg.at(n, srcY, 0, 0);

                // Left border columns map to negative source x; the border wrapper resolves them.
                for (int x = 0; x < left; x++) outRow[x] = src.at(n, srcY, x - left, 0);
                // Interior columns are a contiguous, unmodified copy of the source row.
                std::memcpy(outRow + left, inRow, static_cast<size_t>(srcW) * sizeof(T));
                // Right border columns map to source x >= srcW.
                for (int x = left + srcW; x < dstW; x++) outRow[x] = src.at(n, srcY, x - left, 0);
            } else {
                // Border row, or strided storage: resolve every pixel through the border-aware accessor.
                for (int x = 0; x < dstW; x++) {
                    dst.at(n, y, x, 0) = src.at(n, srcY, x - left, 0);
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels