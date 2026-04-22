#pragma once

/**
 * @file rocjpeg_loader.hpp
 * @brief RAII wrapper around a rocJPEG session that decodes one or more JPEG files directly into a roccv NHWC GPU
 *        tensor via the batched rocJPEG API. Header-only for easy reuse from samples.
 */

#include <core/hip_assert.h>
#include <operator_types.h>
#include <rocjpeg/rocjpeg.h>

#include <algorithm>
#include <cctype>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <core/tensor_data.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class RocJpegLoader
 * @brief Owns a rocJPEG decoder handle plus a pool of stream handles, and decodes one or more JPEG files into a single
 *        NHWC GPU tensor via `rocJpegDecodeBatched`.
 *
 * The class is move-only — the underlying rocJPEG handles are non-copyable. The stream pool grows on demand to
 * accommodate the largest batch the loader has been asked to process, and is reused across subsequent calls.
 */
class RocJpegLoader {
   public:
    /**
     * @brief Creates a rocJPEG decoder bound to the given backend and HIP device.
     *
     * @param backend   rocJPEG backend (hardware VCN by default).
     * @param device_id HIP device index to bind the decoder to.
     * @throws std::runtime_error If `rocJpegCreate` fails.
     */
    explicit RocJpegLoader(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);

    /** @brief Destroys the decoder and all owned stream handles. */
    ~RocJpegLoader();

    RocJpegLoader(const RocJpegLoader&) = delete;
    RocJpegLoader& operator=(const RocJpegLoader&) = delete;

    RocJpegLoader(RocJpegLoader&& other) noexcept;
    RocJpegLoader& operator=(RocJpegLoader&& other) noexcept;

    /**
     * @brief Decodes one or more JPEGs in a single batched rocJPEG call into an NHWC GPU tensor.
     *
     * Each input is parsed and verified to share the same width/height as the first image; a mismatch throws. The
     * returned tensor has `N == image_paths.size()` (works for `N == 1` too).
     *
     * @param image_paths   One or more JPEG file paths. Order is preserved in the output batch dimension.
     * @param fmt           roccv image format used to allocate the output tensor (must match the channel count of
     *                      `output_format` — e.g. RGB8 with `ROCJPEG_OUTPUT_RGB`).
     * @param output_format rocJPEG output layout (interleaved RGB by default).
     * @return NHWC GPU tensor with shape `{N, H, W, C}`.
     * @throws std::runtime_error On I/O errors, parse failures, dimension mismatches, or rocJPEG errors.
     */
    roccv::Tensor loadTensor(const std::vector<std::string>& image_paths, roccv::ImageFormat fmt = roccv::FMT_RGB8,
                             RocJpegOutputFormat output_format = ROCJPEG_OUTPUT_RGB);

    /**
     * @brief Lists `.jpg`/`.jpeg` files (case-insensitive extension) directly within `directory`, sorted by filename.
     *
     * Non-regular entries and subdirectories are skipped (no recursion).
     *
     * @param directory Path to a directory to scan.
     * @return Sorted list of absolute or relative file paths matching the JPEG extensions.
     * @throws std::runtime_error If `directory` is not a directory.
     */
    static std::vector<std::string> listJpegFiles(const std::string& directory);

    /** @brief Returns the HIP device id this loader was bound to at construction. */
    int deviceId() const { return m_device_id; }

   private:
    /** @brief Releases all rocJPEG handles. Idempotent — safe to call after a move-from. */
    void release() noexcept;

    /**
     * @brief Grows `m_stream_pool` so it contains at least `count` rocJPEG stream handles.
     *
     * @param count Minimum number of stream handles required.
     * @throws std::runtime_error If `rocJpegStreamCreate` fails.
     */
    void ensureStreamPool(size_t count);

    RocJpegHandle m_handle{};
    /** Pool of stream handles, grown on demand to match the largest batch ever requested and reused across calls. */
    std::vector<RocJpegStreamHandle> m_stream_pool;
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

namespace rocjpeg_loader_detail {

/**
 * @brief Reads the entire contents of `path` into a freshly-allocated byte buffer.
 *
 * @param path Filesystem path to read.
 * @return Byte buffer containing the file's contents.
 * @throws std::runtime_error If the file cannot be opened or fully read.
 */
inline std::vector<uint8_t> ReadFileBytes(const std::string& path) {
    std::ifstream input(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("RocJpegLoader: failed to open file: " + path);
    }
    const std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
        throw std::runtime_error("RocJpegLoader: failed to read file: " + path);
    }
    return bytes;
}

/**
 * @brief Calls `rocJpegGetImageInfo` on a parsed stream and returns the image's pixel dimensions.
 *
 * @param handle      rocJPEG decoder handle.
 * @param stream      Parsed rocJPEG stream handle for the image being inspected.
 * @param image_path  Path used purely for error messages.
 * @return `{width, height}` of the image's first component (in pixels).
 * @throws std::runtime_error If `rocJpegGetImageInfo` fails.
 */
inline std::pair<int, int> QueryImageSize(RocJpegHandle handle, RocJpegStreamHandle stream,
                                          const std::string& image_path) {
    uint8_t num_components = 0;
    RocJpegChromaSubsampling subsampling{};
    uint32_t widths[ROCJPEG_MAX_COMPONENT]{};
    uint32_t heights[ROCJPEG_MAX_COMPONENT]{};
    const RocJpegStatus st = rocJpegGetImageInfo(handle, stream, &num_components, &subsampling, widths, heights);
    if (st != ROCJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("rocJpegGetImageInfo failed for ") + image_path + ": " +
                                 rocJpegGetErrorName(st));
    }
    return {static_cast<int>(widths[0]), static_cast<int>(heights[0])};
}

}  // namespace rocjpeg_loader_detail

inline roccv::Tensor RocJpegLoader::loadTensor(const std::vector<std::string>& image_paths, roccv::ImageFormat fmt,
                                               RocJpegOutputFormat output_format) {
    using namespace rocjpeg_loader_detail;

    // ---- Validate inputs --------------------------------------------------------------------------------------------
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

    // ---- Read + parse every input, verifying uniform dimensions -----------------------------------------------------
    // NOTE: rocJpegStreamParse stores a raw pointer to the input buffer (no copy), so all per-image file buffers must
    // remain alive until rocJpegDecodeBatched returns. Keep them in a vector that outlives the decode call.
    std::vector<std::vector<uint8_t>> file_buffers(batch_size);
    int common_width = 0;
    int common_height = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        const std::string& image_path = image_paths[i];
        file_buffers[i] = ReadFileBytes(image_path);

        const RocJpegStatus st = rocJpegStreamParse(file_buffers[i].data(), file_buffers[i].size(), m_stream_pool[i]);
        if (st != ROCJPEG_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("rocJpegStreamParse failed for ") + image_path + ": " +
                                     rocJpegGetErrorName(st));
        }

        const auto [w, h] = QueryImageSize(m_handle, m_stream_pool[i], image_path);
        if (i == 0) {
            common_width = w;
            common_height = h;
        } else if (w != common_width || h != common_height) {
            throw std::runtime_error("RocJpegLoader: batched decode requires uniform dimensions; '" + image_path +
                                     "' is " + std::to_string(w) + "x" + std::to_string(h) + " but expected " +
                                     std::to_string(common_width) + "x" + std::to_string(common_height));
        }
    }

    // ---- Allocate the output tensor and build per-image rocJPEG descriptors -----------------------------------------
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

    // ---- Submit the batched decode ----------------------------------------------------------------------------------
    const RocJpegStatus st = rocJpegDecodeBatched(m_handle, stream_handles.data(), static_cast<int>(batch_size),
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
