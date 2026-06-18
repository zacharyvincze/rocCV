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

#include "core/detail/swizzling.hpp"
#include "operator_types.h"

namespace Kernels::Host {

/**
 * @brief Shared skeleton for the elementwise color conversions: OMP-parallel over batch + row with a contiguous fast
 * path and a strided fallback. `perPixel` maps one source pixel to one destination pixel. Kept in Kernels::Host (rather
 * than a nested detail namespace) because the callers pull in `using namespace roccv;`, which would make a `detail`
 * namespace name ambiguous with roccv::detail.
 */
template <typename SrcWrapper, typename DstWrapper, typename Op>
inline void cvt_apply(SrcWrapper input, DstWrapper output, Op perPixel) {
    using in_type = typename SrcWrapper::ValueType;
    using out_type = typename DstWrapper::ValueType;

    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // Each pixel is independent, so packed rows become unit-stride, vectorizable walks.
    const bool contiguous = input.isRowContiguous() && output.isRowContiguous();

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int y = 0; y < height; y++) {
            if (contiguous) {
                const in_type* __restrict__ inRow = &input.at(b, y, 0, 0);
                out_type* __restrict__ outRow = &output.at(b, y, 0, 0);
                for (int x = 0; x < width; x++) outRow[x] = perPixel(inRow[x]);
            } else {
                for (int x = 0; x < width; x++) output.at(b, y, x, 0) = perPixel(input.at(b, y, x, 0));
            }
        }
    }
}

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_yuv(SrcWrapper input, DstWrapper output, float delta) {
    using namespace roccv;
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<T>>;

    cvt_apply(input, output, [delta](auto raw) {
        T val = Swizzle<S>(raw);
        work_type_t valF = StaticCast<work_type_t>(val);

        float y = valF.x * 0.299f + valF.y * 0.587f + valF.z * 0.114f;
        float cr = (valF.x - y) * 0.877f + delta;
        float cb = (valF.z - y) * 0.492f + delta;

        work_type_t out = make_float3(y, cb, cr);
        return SaturateCast<T>(out);
    });
}

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void yuv_to_rgb_or_bgr(SrcWrapper input, DstWrapper output, float delta) {
    using namespace roccv;
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<T>>;

    cvt_apply(input, output, [delta](auto raw) {
        work_type_t valF = StaticCast<work_type_t>(raw);

        // Convert from YUV to RGB
        work_type_t rgb = make_float3(valF.x + (valF.z - delta) * 1.140f,                                // R
                                      valF.x + (valF.y - delta) * -0.395f + (valF.z - delta) * -0.581f,  // G
                                      valF.x + (valF.y - delta) * 2.032f);                               // B

        // Saturate cast to type T (this clamps to proper ranges)
        return Swizzle<S>(SaturateCast<T>(rgb));
    });
}

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void reorder(SrcWrapper input, DstWrapper output) {
    using namespace roccv::detail;

    cvt_apply(input, output, [](auto raw) { return Swizzle<S>(raw); });
}

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_grayscale(SrcWrapper input, DstWrapper output) {
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<T>>;
    using out_type_t = MakeType<BaseType<T>, 1>;

    cvt_apply(input, output, [](auto raw) {
        T inVal = Swizzle<S>(raw);
        work_type_t inValF = StaticCast<work_type_t>(inVal);

        // Calculate luminance
        float y = inValF.x * 0.299f + inValF.y * 0.587f + inValF.z * 0.114f;

        return SaturateCast<out_type_t>(y);
    });
}
}  // namespace Kernels::Host