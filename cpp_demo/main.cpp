#include "segmentationsession.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr char kDefaultImagePath[] =
    "~/da/skin_temp/todo/129550008_2859811450900833_5322782199742913477_n.jpg";

const std::vector<ModelKind> kMenuModels = {
    ModelKind::U2Net,
    ModelKind::U2NetP,
    ModelKind::BiRefNet,
    ModelKind::U2NetPChunks,
};

std::string readLine(const std::string &prompt, const std::string &defaultValue) {
    std::cout << prompt;
    if (!defaultValue.empty()) {
        std::cout << " [" << defaultValue << "]";
    }
    std::cout << ": ";

    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) {
        return defaultValue;
    }
    return line;
}

int readIntChoice(const std::string &prompt, int minValue, int maxValue, int defaultValue) {
    while (true) {
        const std::string line = readLine(prompt, std::to_string(defaultValue));
        try {
            const int value = std::stoi(line);
            if (value >= minValue && value <= maxValue) {
                return value;
            }
        } catch (const std::exception &) {
        }
        std::cout << "Enter a value from " << minValue << " to " << maxValue << ".\n";
    }
}

bool readBoolChoice(const std::string &prompt, bool defaultValue) {
    const std::string defaultText = defaultValue ? "y" : "n";
    while (true) {
        std::string line = readLine(prompt + " (y/n)", defaultText);
        for (char &ch : line) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        if (line == "y" || line == "yes") {
            return true;
        }
        if (line == "n" || line == "no") {
            return false;
        }
        std::cout << "Enter y or n.\n";
    }
}

cv::Mat loadRgbImage(const std::filesystem::path &imagePath) {
    const cv::Mat bgrImage = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (bgrImage.empty()) {
        throw std::runtime_error("Failed to read image: " + imagePath.string());
    }

    cv::Mat rgbImage;
    cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);
    return rgbImage;
}

std::filesystem::path ensureOutputDirectory() {
    const std::filesystem::path outputDir = std::filesystem::current_path() / "output";
    std::filesystem::create_directories(outputDir);
    return outputDir;
}

std::string modelSlug(ModelKind kind) {
    switch (kind) {
    case ModelKind::U2Net:
        return "u2net";
    case ModelKind::U2NetP:
        return "u2netp";
    case ModelKind::BiRefNet:
        return "birefnet";
    case ModelKind::U2NetPChunks:
        return "u2netp_chunks";
    }
    return "model";
}

void writeMask(const std::filesystem::path &path, const cv::Mat &mask) {
    if (!cv::imwrite(path.string(), mask)) {
        throw std::runtime_error("Failed to write image: " + path.string());
    }
}

}  // namespace

int main() {
    try {
        std::cout << "Skin Segmentation C++ Demo\n";
        std::cout << "==========================\n\n";

        std::cout << "Models:\n";
        for (std::size_t i = 0; i < kMenuModels.size(); ++i) {
            const ModelConfig &config = modelConfigFor(kMenuModels[i]);
            std::cout << "  " << (i + 1) << ". " << config.menuLabel << '\n';
            std::cout << "     " << config.modelPath << '\n';
        }
        std::cout << '\n';

        const int modelChoice = readIntChoice(
            "Select model",
            1,
            static_cast<int>(kMenuModels.size()),
            2);
        const ModelKind modelKind = kMenuModels[static_cast<std::size_t>(modelChoice - 1)];
        const bool useGpu = readBoolChoice("Use GPU if available", true);
        const bool useChunkRefinement = readBoolChoice("Run chunked refinement pass", false);
        const std::string imagePathText = readLine("Image path", kDefaultImagePath);

        const std::filesystem::path imagePath(imagePathText);
        const cv::Mat rgbImage = loadRgbImage(imagePath);
        const std::filesystem::path outputDir = ensureOutputDirectory();
        const std::string slug = modelSlug(modelKind);
        const std::string stem = imagePath.stem().string();

        std::cout << "\nLoading ONNX Runtime session...\n";
        Ort::Env env = createOrtEnv();
        auto segmentationSession = createSegmentationSession(modelKind, env, useGpu);

        std::cout << "Running base segmentation with " << segmentationSession->displayName() << "...\n";
        const cv::Mat baseMask = segmentationSession->predictMask(rgbImage);

        const std::filesystem::path baseMaskPath = outputDir / (stem + "_" + slug + "_mask.png");
        writeMask(baseMaskPath, baseMask);
        std::cout << "Base mask written to: " << baseMaskPath << '\n';

        if (useChunkRefinement) {
            std::cout << "Running chunk refinement with U2NetP Chunks...\n";
            auto chunkSession = createChunkRefinementSession(env, useGpu);
            const cv::Mat refinedMask = chunkSession->refineMask(
                rgbImage,
                baseMask,
                [](int current, int total) {
                    std::cout << "\rRefinement tiles: " << current << "/" << total << std::flush;
                    if (current == total) {
                        std::cout << '\n';
                    }
                });

            const std::filesystem::path refinedMaskPath =
                outputDir / (stem + "_" + slug + "_refined_mask.png");
            writeMask(refinedMaskPath, refinedMask);
            std::cout << "Refined mask written to: " << refinedMaskPath << '\n';
        }

        return 0;
    } catch (const Ort::Exception &err) {
        std::cerr << "ONNX Runtime error: " << err.what() << '\n';
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << '\n';
    }

    return 1;
}
