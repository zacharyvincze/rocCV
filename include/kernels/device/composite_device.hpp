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

#include <hip/hip_runtime.h>

#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/wrappers/image_wrapper.hpp"

namespace Kernels {
namespace Device {

template <typename SrcWrapper, typename MaskWrapper, typename DstWrapper>
__global__ void composite(SrcWrapper foreground, SrcWrapper background, MaskWrapper mask, DstWrapper output) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, NumElements<src_type>>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (x >= foreground.width() || y >= foreground.height()) return;

    // Range cast all input values to float to avoid overflowing values and keep them in the same range.
    auto maskFactor = RangeCast<float1>(mask.at(batch, y, x, 0));
    auto fgVal = RangeCast<work_type>(foreground.at(batch, y, x, 0));
    auto bgVal = RangeCast<work_type>(background.at(batch, y, x, 0));

    work_type result = bgVal + maskFactor.x * (fgVal - bgVal);

    // If number of channels in output is 4, ensure that the last channel (alpha in this case) is always fully on.
    if constexpr (NumElements<dst_type> == 4) {
        output.at(batch, y, x, 0) = RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, 1.0f});
    } else {
        output.at(batch, y, x, 0) = RangeCast<dst_type>(result);
    }
}

namespace detail {

using roccv::detail::MakeType;
using roccv::detail::NumElements;
using roccv::detail::RangeCast;

__device__ __forceinline__ void load_rgb24_bytes(const unsigned char* p, unsigned char b[24]) {
    const uintptr_t up = reinterpret_cast<uintptr_t>(p);
    const unsigned phase = static_cast<unsigned>(up & 3u);
    const unsigned char* p0 = p - phase;
    const int nw = static_cast<int>((phase + 24u + 3u) >> 2);
    uint32_t w[7];
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        w[i] = (i < nw) ? *reinterpret_cast<const uint32_t*>(p0 + static_cast<intptr_t>(i) * 4) : 0u;
    }
#pragma unroll
    for (int k = 0; k < 24; ++k) {
        const unsigned idx = phase + static_cast<unsigned>(k);
        const unsigned wi = idx >> 2;
        const unsigned shift = (idx & 3u) * 8u;
        b[k] = static_cast<unsigned char>((w[wi] >> shift) & 0xFFu);
    }
}

__device__ __forceinline__ void unpack_rgb24_to_float01(const unsigned char b[24], float3 out[8]) {
    constexpr float inv = 1.f / 255.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned r = b[i * 3 + 0];
        const unsigned g = b[i * 3 + 1];
        const unsigned bl = b[i * 3 + 2];
        out[i] = make_float3(static_cast<float>(r) * inv, static_cast<float>(g) * inv, static_cast<float>(bl) * inv);
    }
}

__device__ __forceinline__ void store_rgb24_aligned(unsigned char* p, const unsigned char b[24]) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(p);
    dst[0] = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) |
             (static_cast<uint32_t>(b[3]) << 24);
    dst[1] = static_cast<uint32_t>(b[4]) | (static_cast<uint32_t>(b[5]) << 8) | (static_cast<uint32_t>(b[6]) << 16) |
             (static_cast<uint32_t>(b[7]) << 24);
    dst[2] = static_cast<uint32_t>(b[8]) | (static_cast<uint32_t>(b[9]) << 8) | (static_cast<uint32_t>(b[10]) << 16) |
             (static_cast<uint32_t>(b[11]) << 24);
    dst[3] = static_cast<uint32_t>(b[12]) | (static_cast<uint32_t>(b[13]) << 8) | (static_cast<uint32_t>(b[14]) << 16) |
             (static_cast<uint32_t>(b[15]) << 24);
    dst[4] = static_cast<uint32_t>(b[16]) | (static_cast<uint32_t>(b[17]) << 8) | (static_cast<uint32_t>(b[18]) << 16) |
             (static_cast<uint32_t>(b[19]) << 24);
    dst[5] = static_cast<uint32_t>(b[20]) | (static_cast<uint32_t>(b[21]) << 8) | (static_cast<uint32_t>(b[22]) << 16) |
             (static_cast<uint32_t>(b[23]) << 24);
}

}  // namespace detail

/**
 * @brief RGB U8 (uchar3) composite with 8-pixel horizontal tiles: wide loads/stores when pointers are 4-byte aligned.
 */
template <typename MaskWrapper, typename DstWrapper>
__global__ void composite_rgb_u8_packed(roccv::ImageWrapper<uchar3> foreground, roccv::ImageWrapper<uchar3> background,
                                        MaskWrapper mask, DstWrapper output) {
    using namespace roccv::detail;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, 3>;

    const int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    const int width = static_cast<int>(foreground.width());
    const int height = static_cast<int>(foreground.height());
    if (y >= height) return;

    const int x0 = tile_x * 8;
    if (x0 >= width) return;

    const int n = width - x0 < 8 ? width - x0 : 8;

    constexpr int64_t kRgbPixelBytes = 3;
    const int64_t out_pixel_bytes = static_cast<int64_t>(NumElements<dst_type>);
    const bool dense_rgb = foreground.byte_stride_w() == kRgbPixelBytes && background.byte_stride_w() == kRgbPixelBytes &&
                           output.byte_stride_w() == out_pixel_bytes;

    if (!dense_rgb || n < 8) {
        for (int i = 0; i < n; ++i) {
            const int x = x0 + i;
            auto maskFactor = RangeCast<float1>(mask.at(batch, y, x, 0));
            auto fgVal = RangeCast<work_type>(foreground.at(batch, y, x, 0));
            auto bgVal = RangeCast<work_type>(background.at(batch, y, x, 0));
            work_type result = bgVal + maskFactor.x * (fgVal - bgVal);
            if constexpr (NumElements<dst_type> == 4) {
                output.at(batch, y, x, 0) =
                    RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, 1.0f});
            } else {
                output.at(batch, y, x, 0) = RangeCast<dst_type>(result);
            }
        }
        return;
    }

    // Eight packed RGB pixels occupy 24 contiguous bytes when horizontal pixel stride is 3.
    unsigned char* fg_p = reinterpret_cast<unsigned char*>(&foreground.at(batch, y, x0, 0));
    unsigned char* bg_p = reinterpret_cast<unsigned char*>(&background.at(batch, y, x0, 0));
    unsigned char* out_p = reinterpret_cast<unsigned char*>(&output.at(batch, y, x0, 0));

    unsigned char fg_b[24], bg_b[24];
    detail::load_rgb24_bytes(fg_p, fg_b);
    detail::load_rgb24_bytes(bg_p, bg_b);

    float3 fg_f[8], bg_f[8];
    detail::unpack_rgb24_to_float01(fg_b, fg_f);
    detail::unpack_rgb24_to_float01(bg_b, bg_f);

    float3 blended[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float m = RangeCast<float1>(mask.at(batch, y, x0 + i, 0)).x;
        blended[i].x = bg_f[i].x + m * (fg_f[i].x - bg_f[i].x);
        blended[i].y = bg_f[i].y + m * (fg_f[i].y - bg_f[i].y);
        blended[i].z = bg_f[i].z + m * (fg_f[i].z - bg_f[i].z);
    }

    if constexpr (NumElements<dst_type> == 3) {
        const bool aligned_out = (reinterpret_cast<uintptr_t>(out_p) & 3u) == 0;
        unsigned char out_b[24];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const work_type wv = {blended[i].x, blended[i].y, blended[i].z};
            const uchar3 q = RangeCast<uchar3>(wv);
            out_b[i * 3 + 0] = q.x;
            out_b[i * 3 + 1] = q.y;
            out_b[i * 3 + 2] = q.z;
        }
        if (aligned_out) {
            detail::store_rgb24_aligned(out_p, out_b);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const work_type wv = {blended[i].x, blended[i].y, blended[i].z};
                output.at(batch, y, x0 + i, 0) = RangeCast<dst_type>(wv);
            }
        }
    } else {
        static_assert(NumElements<dst_type> == 4, "composite_rgb_u8_packed expects uchar3 or uchar4 output");
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const work_type wv = {blended[i].x, blended[i].y, blended[i].z};
            output.at(batch, y, x0 + i, 0) = RangeCast<dst_type>((MakeType<float, 4>){wv.x, wv.y, wv.z, 1.0f});
        }
    }
}

}  // namespace Device
}  // namespace Kernels
