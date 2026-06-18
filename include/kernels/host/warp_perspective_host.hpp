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

#include <limits>

namespace Kernels {
namespace Host {
template <typename SrcWrapper, typename DstWrapper, typename Mat>
void warp_perspective(SrcWrapper input, DstWrapper output, Mat mat) {
    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int y = 0; y < height; y++) {
            // The matrix * y products are constant across the row. Hoist them, but keep them as separate addends so the
            // per-pixel expression stays bit-identical to the device kernel (the golden model compares byte-for-byte).
            const float m1y = mat[1] * y;
            const float m4y = mat[4] * y;
            const float m7y = mat[7] * y;

            for (int x = 0; x < width; x++) {
                const float denom = mat[6] * x + m7y + mat[8];
                const float coeff = denom == 0.0 ? std::numeric_limits<float>::max() : 1.0f / denom;
                const float ox = (mat[0] * x + m1y + mat[2]) * coeff;
                const float oy = (mat[3] * x + m4y + mat[5]) * coeff;
                output.at(b, y, x, 0) = input.at(b, oy, ox, 0);
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels