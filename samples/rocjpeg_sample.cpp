#include <core/hip_assert.h>

#include <chrono>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <op_flip.hpp>
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/rocjpeg_loader.hpp"

namespace {

void WriteRgbTensorSliceToImageFile(const roccv::Tensor& tensor, int batch_idx, const std::string& path) {
    auto data = tensor.exportData<roccv::TensorDataStrided>();
    const auto& layout = tensor.layout();
    const int h = static_cast<int>(tensor.shape(layout.height_index()));
    const int w = static_cast<int>(tensor.shape(layout.width_index()));
    const int c = static_cast<int>(tensor.shape(layout.channels_index()));
    const size_t src_pitch = static_cast<size_t>(data.stride(layout.height_index()));
    const size_t row_bytes = static_cast<size_t>(w * c) * tensor.dtype().size();

    auto* base =
        static_cast<uint8_t*>(data.basePtr()) + static_cast<int64_t>(batch_idx) * data.stride(layout.batch_index());

    cv::Mat rgb(h, w, CV_8UC3);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(rgb.data, rgb.step[0], base, src_pitch, row_bytes, h, hipMemcpyDeviceToHost));

    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, bgr)) {
        throw std::runtime_error("OpenCV failed to write image: " + path);
    }
}

std::string FlippedOutputPath(const std::string& input_path, const std::filesystem::path& output_dir) {
    std::filesystem::path in_p(input_path);
    return (output_dir / (in_p.stem().string() + "_flipped.png")).string();
}

constexpr const char* kDefaultOutputDirName = "rocjpeg_flipped_output";

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void PrintTiming(const char* label, double ms) {
    std::cout << "  " << std::left << std::setw(28) << label << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << ms << " ms" << std::endl;
}

void PrintUsage(const char* prog, std::ostream& os) {
    os << "Usage: " << prog << " <image.jpg|directory> [output_directory]\n"
       << "  -h, --help         Show this message and exit.\n"
       << "  <input>            Path to a single .jpg/.jpeg file or a directory of them.\n"
       << "                     Directory inputs are batch-decoded; all images must share dimensions.\n"
       << "  [output_directory] Directory to write <stem>_flipped.png files into.\n"
       << "                     Defaults to ./" << kDefaultOutputDirName << " in the current working directory."
       << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
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
    const bool input_is_directory = std::filesystem::is_directory(input_arg);
    const std::filesystem::path output_dir =
        (argc == 3) ? std::filesystem::path(argv[2]) : std::filesystem::current_path() / kDefaultOutputDirName;

    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Error: failed to create output directory '" << output_dir.string() << "': " << ec.message()
                      << std::endl;
            return EXIT_FAILURE;
        }
        if (!std::filesystem::is_directory(output_dir)) {
            std::cerr << "Error: output path is not a directory: " << output_dir.string() << std::endl;
            return EXIT_FAILURE;
        }
    }

    try {
        const auto t_loader_start = Clock::now();
        RocJpegLoader loader(ROCJPEG_BACKEND_HARDWARE, 0);
        const auto t_loader_end = Clock::now();

        const auto t_list_start = Clock::now();
        std::vector<std::string> input_paths;
        if (input_is_directory) {
            input_paths = RocJpegLoader::listJpegFiles(input_arg);
            if (input_paths.empty()) {
                throw std::runtime_error("No .jpg/.jpeg files found in directory: " + input_arg);
            }
            std::cout << "Batch decoding " << input_paths.size() << " images from " << input_arg << std::endl;
        } else {
            input_paths.push_back(input_arg);
        }
        const auto t_list_end = Clock::now();

        const auto t_decode_start = Clock::now();
        roccv::Tensor input_tensor = loader.loadTensor(input_paths);
        const auto t_decode_end = Clock::now();

        const int batch = static_cast<int>(input_tensor.shape(input_tensor.layout().batch_index()));
        const int width = static_cast<int>(input_tensor.shape("W"));
        const int height = static_cast<int>(input_tensor.shape("H"));

        const auto t_alloc_start = Clock::now();
        roccv::Tensor output_tensor(batch, roccv::Size2D{width, height}, roccv::FMT_RGB8, eDeviceType::GPU);
        roccv::Tensor resized_tensor(batch, roccv::Size2D{width * 2, height * 2}, roccv::FMT_RGB8, eDeviceType::GPU);
        const auto t_alloc_end = Clock::now();

        hipStream_t stream{};
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        const auto t_flip_start = Clock::now();
        roccv::Flip flip;
        roccv::Resize resize;
        flip(stream, input_tensor, output_tensor, -1, eDeviceType::GPU);
        resize(stream, output_tensor, resized_tensor, eInterpolationType::INTERP_TYPE_LINEAR, eDeviceType::GPU);
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        const auto t_flip_end = Clock::now();

        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        const auto t_write_start = Clock::now();
        for (int i = 0; i < batch; ++i) {
            const std::string output_path = FlippedOutputPath(input_paths[i], output_dir);
            WriteRgbTensorSliceToImageFile(resized_tensor, i, output_path);
        }
        const auto t_write_end = Clock::now();
        std::cout << "Wrote " << batch << " images to " << output_dir.string() << std::endl;

        const double decode_ms = ElapsedMs(t_decode_start, t_decode_end);
        const double flip_ms = ElapsedMs(t_flip_start, t_flip_end);
        const double write_ms = ElapsedMs(t_write_start, t_write_end);
        const double total_ms = ElapsedMs(t_loader_start, t_write_end);

        std::cout << "\nTimings (" << batch << " image" << (batch == 1 ? "" : "s") << " @ " << width << "x" << height
                  << "):" << std::endl;
        PrintTiming("rocJPEG init", ElapsedMs(t_loader_start, t_loader_end));
        PrintTiming("List inputs", ElapsedMs(t_list_start, t_list_end));
        PrintTiming("Decode (batched)", decode_ms);
        PrintTiming("  per image", decode_ms / batch);
        PrintTiming("Output tensor alloc", ElapsedMs(t_alloc_start, t_alloc_end));
        PrintTiming("Flip + stream sync", flip_ms);
        PrintTiming("  per image", flip_ms / batch);
        PrintTiming("Write PNGs", write_ms);
        PrintTiming("  per image", write_ms / batch);
        PrintTiming("Total", total_ms);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
