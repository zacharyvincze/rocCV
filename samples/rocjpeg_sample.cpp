/**
 * @file rocjpeg_sample.cpp
 * @brief End-to-end sample: GPU-decode JPEGs with rocJPEG, run roccv operators (Flip + Resize) on the result, then
 *        write the output back to disk via OpenCV. Accepts either a single .jpg file or a directory of equal-sized
 *        JPEGs (decoded as a single rocJPEG batch).
 */

#include <core/hip_assert.h>

#include <chrono>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <op_cvt_color.hpp>
#include <op_flip.hpp>
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/rocjpeg_loader.hpp"

namespace {

/** Default subdirectory (created in the current working directory) for flipped + resized outputs. */
constexpr const char* kDefaultOutputDirName = "rocjpeg_flipped_output";

/** Output upscale factor applied by the resize operator after flipping. */
constexpr int kResizeFactor = 2;

using Clock = std::chrono::steady_clock;

/**
 * @brief Returns the elapsed time between two clock points in milliseconds.
 *
 * @param start Earlier time point.
 * @param end   Later time point.
 * @return Elapsed wall-clock time in milliseconds (double precision).
 */
double ElapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Prints a single labelled timing line to stdout, formatted as a uniform two-column row.
 *
 * @param label Short human-readable label describing the measured step.
 * @param ms    Elapsed time in milliseconds.
 */
void PrintTiming(const char* label, double ms) {
    std::cout << "  " << std::left << std::setw(28) << label << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << ms << " ms" << std::endl;
}

/**
 * @brief Builds an output path of the form `<output_dir>/<input_stem>_flipped.png`.
 *
 * @param input_path Path to the original input image; only its filename stem is used.
 * @param output_dir Directory the output should be written into.
 * @return Fully-qualified output path string.
 */
std::string FlippedOutputPath(const std::string& input_path, const std::filesystem::path& output_dir) {
    std::filesystem::path in_p(input_path);
    return (output_dir / (in_p.stem().string() + "_flipped.png")).string();
}

/**
 * @brief Prints CLI usage information.
 *
 * @param prog Program name (typically `argv[0]`).
 * @param os   Stream to print to (use `std::cout` for `--help`, `std::cerr` for usage errors).
 */
void PrintUsage(const char* prog, std::ostream& os) {
    os << "Usage: " << prog << " <image.jpg|directory> [output_directory]\n"
       << "  -h, --help         Show this message and exit.\n"
       << "  <input>            Path to a single .jpg/.jpeg file or a directory of them.\n"
       << "                     Directory inputs are batch-decoded; all images must share dimensions.\n"
       << "  [output_directory] Directory to write <stem>_flipped.png files into.\n"
       << "                     Defaults to ./" << kDefaultOutputDirName << " in the current working directory."
       << std::endl;
}

/**
 * @brief Creates the output directory (recursively) if it doesn't exist and verifies it is a directory.
 *
 * @param dir Directory path to ensure.
 * @throws std::runtime_error If the directory cannot be created or `dir` exists as a non-directory.
 */
void EnsureOutputDirectory(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("failed to create output directory '" + dir.string() + "': " + ec.message());
    }
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("output path is not a directory: " + dir.string());
    }
}

/**
 * @brief Resolves the CLI input argument into a list of JPEG paths to decode.
 *
 * If `input_arg` is a directory, returns all `.jpg`/`.jpeg` entries within it (sorted). If it is a regular file,
 * returns a single-element vector containing that file.
 *
 * @param input_arg User-supplied input path (file or directory).
 * @return Ordered list of JPEG paths to feed to the loader.
 * @throws std::runtime_error If the input directory contains no JPEG files.
 */
std::vector<std::string> ResolveInputPaths(const std::string& input_arg) {
    if (std::filesystem::is_directory(input_arg)) {
        auto paths = RocJpegLoader::listJpegFiles(input_arg);
        if (paths.empty()) {
            throw std::runtime_error("No .jpg/.jpeg files found in directory: " + input_arg);
        }
        return paths;
    }
    return {input_arg};
}

/**
 * @brief Copies a single image (slice `batch_idx`) of an NHWC U8 tensor from device memory to host and writes it as an
 *        image file via OpenCV.
 *
 * Supports both 3-channel RGB8 (converted to BGR before writing, since OpenCV uses BGR ordering on disk) and 1-channel
 * grayscale U8 (written as-is). Any other channel count throws.
 *
 * @param tensor    NHWC U8 source tensor on the GPU (1 or 3 channels).
 * @param batch_idx Zero-based index of the image within the batch to write.
 * @param path      Destination image path (extension determines the encoder used by OpenCV).
 * @throws std::runtime_error If the channel count is unsupported or OpenCV fails to write the file.
 */
void WriteTensorSliceToImageFile(const roccv::Tensor& tensor, int batch_idx, const std::string& path) {
    auto data = tensor.exportData<roccv::TensorDataStrided>();
    const auto& layout = tensor.layout();
    const int h = static_cast<int>(tensor.shape(layout.height_index()));
    const int w = static_cast<int>(tensor.shape(layout.width_index()));
    const int c = static_cast<int>(tensor.shape(layout.channels_index()));
    const size_t src_pitch = static_cast<size_t>(data.stride(layout.height_index()));
    const size_t row_bytes = static_cast<size_t>(w * c) * tensor.dtype().size();

    int cv_type = 0;
    switch (c) {
        case 1:
            cv_type = CV_8UC1;
            break;
        case 3:
            cv_type = CV_8UC3;
            break;
        default:
            throw std::runtime_error("WriteTensorSliceToImageFile: unsupported channel count " + std::to_string(c) +
                                     " (expected 1 or 3)");
    }

    auto* base =
        static_cast<uint8_t*>(data.basePtr()) + static_cast<int64_t>(batch_idx) * data.stride(layout.batch_index());

    cv::Mat host(h, w, cv_type);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(host.data, host.step[0], base, src_pitch, row_bytes, h, hipMemcpyDeviceToHost));

    // OpenCV expects BGR ordering on disk for color images; grayscale is written as-is.
    cv::Mat to_write;
    if (c == 3) {
        cv::cvtColor(host, to_write, cv::COLOR_RGB2BGR);
    } else {
        to_write = host;
    }

    if (!cv::imwrite(path, to_write)) {
        throw std::runtime_error("OpenCV failed to write image: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Help flag — checked before arg-count validation so `--help` works regardless of position.
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0], std::cout);
            return EXIT_SUCCESS;
        }
    }

    if (argc < 2 || argc > 3) {
        PrintUsage(argv[0], std::cerr);
        return EXIT_FAILURE;
    }

    const std::string input_arg(argv[1]);
    const std::filesystem::path output_dir =
        (argc == 3) ? std::filesystem::path(argv[2]) : std::filesystem::current_path() / kDefaultOutputDirName;

    try {
        EnsureOutputDirectory(output_dir);

        // Initialize rocJPEG loader
        std::cout << "Initializing rocJPEG loader" << std::endl;
        const auto t_loader_start = Clock::now();
        RocJpegLoader loader(ROCJPEG_BACKEND_HARDWARE, 0);
        const auto t_loader_end = Clock::now();

        // List input files
        const std::vector<std::string> input_paths = ResolveInputPaths(input_arg);

        // Batch decode images into a single tensor
        std::cout << "Decoding images into input tensor" << std::endl;
        const auto t_decode_start = Clock::now();
        roccv::Tensor input_tensor = loader.loadTensor(input_paths);
        const auto t_decode_end = Clock::now();

        const int batch = static_cast<int>(input_tensor.shape(input_tensor.layout().batch_index()));
        const int width = static_cast<int>(input_tensor.shape("W"));
        const int height = static_cast<int>(input_tensor.shape("H"));

        // Allocate output tensors
        std::cout << "Allocating output tensors" << std::endl;
        const auto t_alloc_start = Clock::now();
        roccv::Tensor flipped_tensor(batch, roccv::Size2D{width, height}, roccv::FMT_RGB8, eDeviceType::GPU);
        roccv::Tensor grayscale_tensor(batch, roccv::Size2D{width, height}, roccv::FMT_U8, eDeviceType::GPU);
        roccv::Tensor resized_tensor(batch, roccv::Size2D{width * kResizeFactor, height * kResizeFactor}, roccv::FMT_U8,
                                     eDeviceType::GPU);
        const auto t_alloc_end = Clock::now();

        hipStream_t stream{};
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        // Image preprocessing
        std::cout << "Preprocessing images" << std::endl;
        roccv::Flip flip;
        roccv::CvtColor cvt_color;
        roccv::Resize resize;

        const auto t_ops_start = Clock::now();
        flip(stream, input_tensor, flipped_tensor, -1, eDeviceType::GPU);
        cvt_color(stream, flipped_tensor, grayscale_tensor, eColorConversionCode::COLOR_RGB2GRAY, eDeviceType::GPU);
        resize(stream, grayscale_tensor, resized_tensor, eInterpolationType::INTERP_TYPE_LINEAR, eDeviceType::GPU);
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        const auto t_ops_end = Clock::now();

        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        // Write output images
        std::cout << "Writing output images" << std::endl;
        const auto t_write_start = Clock::now();
        for (int i = 0; i < batch; ++i) {
            const std::string output_path = FlippedOutputPath(input_paths[i], output_dir);
            WriteTensorSliceToImageFile(resized_tensor, i, output_path);
        }
        const auto t_write_end = Clock::now();
        std::cout << "Done! Wrote " << batch << " images to " << std::filesystem::absolute(output_dir).string()
                  << std::endl;

        // Timing summary
        const double decode_ms = ElapsedMs(t_decode_start, t_decode_end);
        const double ops_ms = ElapsedMs(t_ops_start, t_ops_end);
        const double write_ms = ElapsedMs(t_write_start, t_write_end);
        const double total_ms = ElapsedMs(t_loader_start, t_write_end);

        std::cout << "\nTimings (" << batch << " image" << (batch == 1 ? "" : "s") << " @ " << width << "x" << height
                  << "):" << std::endl;
        PrintTiming("rocJPEG init", ElapsedMs(t_loader_start, t_loader_end));
        PrintTiming("Decode (batched)", decode_ms);
        PrintTiming("  per image", decode_ms / batch);
        PrintTiming("Output tensor alloc", ElapsedMs(t_alloc_start, t_alloc_end));
        PrintTiming("Image preprocessing + sync", ops_ms);
        PrintTiming("  per image", ops_ms / batch);
        PrintTiming("Write PNGs", write_ms);
        PrintTiming("  per image", write_ms / batch);
        PrintTiming("Total", total_ms);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
