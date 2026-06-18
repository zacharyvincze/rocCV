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

#include <core/wrappers/generic_tensor_wrapper.hpp>

#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Host {
namespace detail {
/**
 * @brief Shared skeleton for every thresholding variant: OMP-parallel over batch + row with a contiguous fast path and
 * a strided fallback. `op(ip, th, mv)` computes a single channel's thresholded value in double precision; modes without
 * a max value simply ignore `mv`. thresh/maxVal are per-batch broadcasts, resolved once per row.
 */
template <typename SrcWrapper, typename DstWrapper, typename Op>
inline void threshold_apply(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh,
                            roccv::GenericTensorWrapper<double> maxVal, Op op) {
    using namespace roccv::detail;
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;
    constexpr int kChannels = NumElements<dst_type>;  // Compile-time channel count enables the inner loop to unroll.

    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // Each pixel is independent, so packed rows become unit-stride, vectorizable walks.
    const bool contiguous = input.isRowContiguous() && output.isRowContiguous();

    // Single-sourced per-pixel formula shared by both the fast path and the strided fallback.
    auto computePixel = [&](src_type in, double th, double mv) -> dst_type {
        dst_type out;
        for (int i = 0; i < kChannels; i++) {
            double ip = StaticCast<double>(GetElement(in, i));
            GetElement(out, i) = StaticCast<base_type>(op(ip, th, mv));
        }
        return out;
    };

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int y = 0; y < height; y++) {
            const double th = thresh.at(b);
            const double mv = maxVal.at(b);
            if (contiguous) {
                const src_type* __restrict__ inRow = &input.at(b, y, 0, 0);
                dst_type* __restrict__ outRow = &output.at(b, y, 0, 0);
                for (int x = 0; x < width; x++) outRow[x] = computePixel(inRow[x], th, mv);
            } else {
                for (int x = 0; x < width; x++) output.at(b, y, x, 0) = computePixel(input.at(b, y, x, 0), th, mv);
            }
        }
    }
}
}  // namespace detail

template <typename SrcWrapper, typename DstWrapper>
void binary_generic(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh,
                    roccv::GenericTensorWrapper<double> maxVal) {
    detail::threshold_apply(input, output, thresh, maxVal,
                            [](double ip, double th, double mv) { return ip > th ? mv : 0.0; });
}

template <typename SrcWrapper, typename DstWrapper>
void binary_inv_generic(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh,
                        roccv::GenericTensorWrapper<double> maxVal) {
    detail::threshold_apply(input, output, thresh, maxVal,
                            [](double ip, double th, double mv) { return ip > th ? 0.0 : mv; });
}

template <typename SrcWrapper, typename DstWrapper>
void trunc_generic(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh) {
    // No max value for this mode; pass `thresh` as a harmless stand-in since the op ignores mv.
    detail::threshold_apply(input, output, thresh, thresh,
                            [](double ip, double th, double /*mv*/) { return ip > th ? th : ip; });
}

template <typename SrcWrapper, typename DstWrapper>
void tozero_generic(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh) {
    detail::threshold_apply(input, output, thresh, thresh,
                            [](double ip, double th, double /*mv*/) { return ip > th ? ip : 0.0; });
}

template <typename SrcWrapper, typename DstWrapper>
void tozeroinv_generic(SrcWrapper input, DstWrapper output, roccv::GenericTensorWrapper<double> thresh) {
    detail::threshold_apply(input, output, thresh, thresh,
                            [](double ip, double th, double /*mv*/) { return ip > th ? 0.0 : ip; });
}
}  // namespace Host
}  // namespace Kernels