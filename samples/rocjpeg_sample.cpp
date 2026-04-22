#include <core/hip_assert.h>

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <filesystem>
#include <iostream>
#include <op_flip.hpp>
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
        RocJpegLoader loader(ROCJPEG_BACKEND_HARDWARE, 0);

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
        roccv::Tensor input_tensor = loader.loadTensor(input_paths);

        const int batch = static_cast<int>(input_tensor.shape(input_tensor.layout().batch_index()));
        const int width = static_cast<int>(input_tensor.shape("W"));
        const int height = static_cast<int>(input_tensor.shape("H"));

        roccv::Tensor output_tensor(batch, roccv::Size2D{width, height}, roccv::FMT_RGB8, eDeviceType::GPU);

        hipStream_t stream{};
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        roccv::Flip flip;
        flip(stream, input_tensor, output_tensor, 1, eDeviceType::GPU);

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        std::cout << "Synchronized stream" << std::endl;
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        for (int i = 0; i < batch; ++i) {
            const std::string output_path = FlippedOutputPath(input_paths[i], output_dir);
            WriteRgbTensorSliceToImageFile(output_tensor, i, output_path);
        }
        std::cout << "Wrote " << batch << " images to " << output_dir.string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
