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

#include <cstring>

#include "operator_types.h"

namespace Kernels {
namespace Host {
template <typename SrcWrapper, typename DstWrapper>
void custom_crop(SrcWrapper input, DstWrapper output, roccv::Box_t cropRect) {
    using T = typename DstWrapper::ValueType;

    const int batches = static_cast<int>(output.batches());
    const int cropX = cropRect.x;
    const int cropY = cropRect.y;
    const int cropW = cropRect.width;
    const int cropH = cropRect.height;

    // Each output row is a verbatim, contiguous span of the corresponding source row [cropX, cropX+cropW); when rows
    // are packed it collapses to a single memcpy instead of a per-pixel copy.
    const bool contiguous = input.isRowContiguous() && output.isRowContiguous();

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int j = 0; j < cropH; j++) {
            const int sourceY = j + cropY;
            if (contiguous) {
                const T* __restrict__ inRow = &input.at(b, sourceY, cropX, 0);
                T* __restrict__ outRow = &output.at(b, j, 0, 0);
                std::memcpy(outRow, inRow, static_cast<size_t>(cropW) * sizeof(T));
            } else {
                for (int i = 0; i < cropW; i++) {
                    output.at(b, j, i, 0) = input.at(b, sourceY, i + cropX, 0);
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels