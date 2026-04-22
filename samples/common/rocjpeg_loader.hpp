#pragma once

#include <rocjpeg/rocjpeg.h>

#include <core/hip_assert.h>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <core/tensor_data.hpp>
#include <operator_types.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Owns a rocJPEG session (decoder handle + per-image stream handles) and decodes JPEG files into roccv GPU tensors.
 */
class RocJpegLoader {
   public:
    RocJpegLoader(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
    ~RocJpegLoader();

    RocJpegLoader(const RocJpegLoader&) = delete;
    RocJpegLoader& operator=(const RocJpegLoader&) = delete;

    RocJpegLoader(RocJpegLoader&& other) noexcept;
    RocJpegLoader& operator=(RocJpegLoader&& other) noexcept;

    /**
     * Reads one or more JPEGs and decodes them on the GPU as a single batched call. All images must share the same
     * width/height — a mismatch throws. Returns an NHWC tensor with N == image_paths.size() (works for N == 1 too).
     */
    roccv::Tensor loadTensor(const std::vector<std::string>& image_paths,
                             roccv::ImageFormat fmt = roccv::FMT_RGB8,
                             RocJpegOutputFormat output_format = ROCJPEG_OUTPUT_RGB);

    /** Lists JPEG files (.jpg/.jpeg, case-insensitive) directly within `directory`, sorted by filename. */
    static std::vector<std::string> listJpegFiles(const std::string& directory);

    int deviceId() const { return m_device_id; }

   private:
    void release() noexcept;
    void ensureStreamPool(size_t count);

    RocJpegHandle m_handle{};
    std::vector<RocJpegStreamHandle> m_stream_pool;  // grown on demand to match the largest batch seen
    int m_device_id = 0;
    bool m_valid = false;
};

inline RocJpegLoader::RocJpegLoader(RocJpegBackend backend, int device_id) : m_device_id(device_id) {
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_id));

    RocJpegStatus st = rocJpegCreate(backend, device_id, &m_handle);
    if (st != ROCJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("rocJpegCreate failed: ") + rocJpegGetErrorName(st));
    }

    m_valid = true;
}

inline RocJpegLoader::~RocJpegLoader() { release(); }

inline RocJpegLoader::RocJpegLoader(RocJpegLoader&& other) noexcept
    : m_handle(other.m_handle),
      m_stream_pool(std::move(other.m_stream_pool)),
      m_device_id(other.m_device_id),
      m_valid(other.m_valid) {
    other.m_handle = {};
    other.m_valid = false;
}

inline RocJpegLoader& RocJpegLoader::operator=(RocJpegLoader&& other) noexcept {
    if (this != &other) {
        release();
        m_handle = other.m_handle;
        m_stream_pool = std::move(other.m_stream_pool);
        m_device_id = other.m_device_id;
        m_valid = other.m_valid;
        other.m_handle = {};
        other.m_valid = false;
    }
    return *this;
}

inline void RocJpegLoader::release() noexcept {
    if (!m_valid) {
        return;
    }
    for (auto handle : m_stream_pool) {
        rocJpegStreamDestroy(handle);
    }
    m_stream_pool.clear();
    rocJpegDestroy(m_handle);
    m_handle = {};
    m_valid = false;
}

inline void RocJpegLoader::ensureStreamPool(size_t count) {
    while (m_stream_pool.size() < count) {
        RocJpegStreamHandle h{};
        RocJpegStatus st = rocJpegStreamCreate(&h);
        if (st != ROCJPEG_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("rocJpegStreamCreate failed: ") + rocJpegGetErrorName(st));
        }
        m_stream_pool.push_back(h);
    }
}

inline roccv::Tensor RocJpegLoader::loadTensor(const std::vector<std::string>& image_paths, roccv::ImageFormat fmt,
                                               RocJpegOutputFormat output_format) {
    if (!m_valid) {
        throw std::runtime_error("RocJpegLoader: session is not valid (was it moved from?)");
    }
    if (image_paths.empty()) {
        throw std::runtime_error("RocJpegLoader: loadTensor called with no image paths");
    }
    if (output_format == ROCJPEG_OUTPUT_RGB && fmt.channels() != 3) {
        throw std::runtime_error("RocJpegLoader: ROCJPEG_OUTPUT_RGB requires a 3-channel ImageFormat");
    }

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(m_device_id));

    const size_t batch_size = image_paths.size();
    ensureStreamPool(batch_size);

    int common_width = 0;
    int common_height = 0;

    // rocJpegStreamParse stores a raw pointer to the input buffer (no copy), so all per-image file buffers must
    // remain alive until rocJpegDecodeBatched returns. Keep them in a vector that outlives the decode call.
    std::vector<std::vector<uint8_t>> file_buffers(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        const std::string& image_path = image_paths[i];

        std::ifstream input(image_path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("RocJpegLoader: failed to open file: " + image_path);
        }
        const std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        auto& file_data = file_buffers[i];
        file_data.resize(static_cast<size_t>(file_size));
        if (!input.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
            throw std::runtime_error("RocJpegLoader: failed to read file: " + image_path);
        }

        RocJpegStatus st = rocJpegStreamParse(file_data.data(), file_data.size(), m_stream_pool[i]);
        if (st != ROCJPEG_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("rocJpegStreamParse failed for ") + image_path + ": " +
                                     rocJpegGetErrorName(st));
        }

        uint8_t num_components = 0;
        RocJpegChromaSubsampling subsampling{};
        uint32_t widths[ROCJPEG_MAX_COMPONENT]{};
        uint32_t heights[ROCJPEG_MAX_COMPONENT]{};
        st = rocJpegGetImageInfo(m_handle, m_stream_pool[i], &num_components, &subsampling, widths, heights);
        if (st != ROCJPEG_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("rocJpegGetImageInfo failed for ") + image_path + ": " +
                                     rocJpegGetErrorName(st));
        }

        const int w = static_cast<int>(widths[0]);
        const int h = static_cast<int>(heights[0]);
        if (i == 0) {
            common_width = w;
            common_height = h;
        } else if (w != common_width || h != common_height) {
            throw std::runtime_error("RocJpegLoader: batched decode requires uniform dimensions; '" + image_path +
                                     "' is " + std::to_string(w) + "x" + std::to_string(h) + " but expected " +
                                     std::to_string(common_width) + "x" + std::to_string(common_height));
        }
    }

    roccv::Tensor tensor(static_cast<int>(batch_size), roccv::Size2D{common_width, common_height}, fmt,
                         eDeviceType::GPU);
    auto strided = tensor.exportData<roccv::TensorDataStrided>();
    const size_t row_pitch = static_cast<size_t>(strided.stride(tensor.layout().height_index()));
    const int64_t batch_stride = strided.stride(tensor.layout().batch_index());
    auto* base = static_cast<uint8_t*>(strided.basePtr());

    std::vector<RocJpegImage> destinations(batch_size);
    std::vector<RocJpegDecodeParams> decode_params(batch_size);
    std::vector<RocJpegStreamHandle> stream_handles(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        destinations[i] = {};
        destinations[i].channel[0] = base + static_cast<int64_t>(i) * batch_stride;
        destinations[i].pitch[0] = static_cast<uint32_t>(row_pitch);
        decode_params[i] = {};
        decode_params[i].output_format = output_format;
        stream_handles[i] = m_stream_pool[i];
    }

    RocJpegStatus st = rocJpegDecodeBatched(m_handle, stream_handles.data(), static_cast<int>(batch_size),
                                            decode_params.data(), destinations.data());
    if (st != ROCJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("rocJpegDecodeBatched failed: ") + rocJpegGetErrorName(st));
    }

    return tensor;
}

inline std::vector<std::string> RocJpegLoader::listJpegFiles(const std::string& directory) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("RocJpegLoader: not a directory: " + directory);
    }
    std::vector<std::string> out;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg") {
            out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}
