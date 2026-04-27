#include "segmentationsession.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

constexpr char kBiRefNetModelPath[] =
    "~/ai_data/skin_models/BiRefNet/birefnet.onnx";
constexpr char kU2NetModelPath[] =
    "~/ai_data/skin_models/new/u2net.onnx";
constexpr char kU2NetPModelPath[] =
    "~/ai_data/skin_models/newp/u2netp.onnx";
constexpr char kU2NetPChunksModelPath[] =
    "~/ai_data/skin_models/chunks/u2netp_chunks.onnx";

constexpr ModelConfig kModelConfigs[] = {
    {ModelKind::U2Net, "U2Net", "U2Net", kU2NetModelPath, 1024},
    {ModelKind::U2NetP, "U2NetP", "U2NetP", kU2NetPModelPath, 512},
    {ModelKind::BiRefNet, "BiRefNet", "BiRefNet", kBiRefNetModelPath, 1440},
    {ModelKind::U2NetPChunks, "U2NetP Chunks", "U2NetP Chunks", kU2NetPChunksModelPath, 512},
};

struct TileBox {
    int x0;
    int y0;
    int x1;
    int y1;
};

struct PaddedSquareImage {
    cv::Mat image;
    int contentWidth;
    int contentHeight;
};

constexpr int kChunkTileSize = 512;
constexpr int kChunkOverlap = 64;
constexpr float kChunkMinPosFrac = 0.01f;
constexpr int kChunkBinThresh = 0;
constexpr int kChunkEdgeBandPx = 12;
constexpr float kChunkEdgeTargetFrac = 0.8f;
constexpr float kChunkPosPureFrac = 0.1f;
constexpr float kChunkNegPureFrac = 0.1f;

bool hasProvider(const std::unordered_set<std::string> &available, const char *provider) {
    return available.find(provider) != available.end();
}

std::string joinProviders(const std::set<std::string> &providers) {
    std::string result;
    for (const std::string &provider : providers) {
        if (!result.empty()) {
            result += ", ";
        }
        result += provider;
    }
    return result;
}

bool appendDirectMLProvider(Ort::SessionOptions *options) {
    options->DisableMemPattern();
    options->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    try {
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
            *options, "DmlExecutionProvider", nullptr, nullptr, 0));
        return true;
    } catch (const Ort::Exception &) {
    }

    try {
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
            *options, "DML", nullptr, nullptr, 0));
        return true;
    } catch (const Ort::Exception &) {
        return false;
    }
}

Ort::SessionOptions makeSessionOptions(bool allowGpu) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    const std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    const std::set<std::string> sortedProviders(availableProviders.begin(), availableProviders.end());
    const std::unordered_set<std::string> availableSet(availableProviders.begin(), availableProviders.end());

    std::cout << "Available ONNX Runtime providers: " << joinProviders(sortedProviders) << '\n';

    if (!allowGpu) {
        std::cout << "Using ONNX Runtime provider: CPUExecutionProvider\n";
        return options;
    }

    if (hasProvider(availableSet, "CUDAExecutionProvider")) {
        try {
            OrtCUDAProviderOptions cudaOptions{};
            options.AppendExecutionProvider_CUDA(cudaOptions);
            std::cout << "Using ONNX Runtime provider: CUDAExecutionProvider\n";
            return options;
        } catch (const Ort::Exception &err) {
            std::cerr << "Failed to append CUDAExecutionProvider: " << err.what() << '\n';
        }
    }

    if (hasProvider(availableSet, "DmlExecutionProvider") || hasProvider(availableSet, "DML")) {
        if (appendDirectMLProvider(&options)) {
            std::cout << "Using ONNX Runtime provider: DmlExecutionProvider\n";
            return options;
        }
    }

    std::cout << "GPU execution requested but unavailable, falling back to CPU\n";
    return options;
}

std::vector<float> matToNchwFloat(const cv::Mat &mat) {
    const int width = mat.cols;
    const int height = mat.rows;
    const int channels = mat.channels();
    std::vector<float> values(static_cast<std::size_t>(channels) * height * width);

    for (int y = 0; y < height; ++y) {
        const std::uint8_t *line = mat.ptr<std::uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            for (int channel = 0; channel < channels; ++channel) {
                values[static_cast<std::size_t>(channel) * height * width + idx] =
                    static_cast<float>(line[x * channels + channel]) / 255.0f;
            }
        }
    }

    return values;
}

cv::Mat resizeMat(const cv::Mat &source, int width, int height) {
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LANCZOS4);
    return resized;
}

cv::Mat resizeMaskToOriginal(const cv::Mat &mask, const cv::Mat &originalMat) {
    return resizeMat(mask, originalMat.cols, originalMat.rows);
}

PaddedSquareImage makePaddedSquareImage(const cv::Mat &source, int targetSize, int type) {
    int resizedWidth = targetSize;
    int resizedHeight = targetSize;
    if (source.rows >= source.cols) {
        resizedHeight = targetSize;
        resizedWidth = std::max(
            1,
            static_cast<int>(
                std::lround(static_cast<double>(source.cols) * targetSize / source.rows)));
    } else {
        resizedWidth = targetSize;
        resizedHeight = std::max(
            1,
            static_cast<int>(
                std::lround(static_cast<double>(source.rows) * targetSize / source.cols)));
    }

    PaddedSquareImage result{
        cv::Mat::zeros(targetSize, targetSize, type),
        resizedWidth,
        resizedHeight,
    };
    resizeMat(source, resizedWidth, resizedHeight)
        .copyTo(result.image(cv::Rect(0, 0, resizedWidth, resizedHeight)));
    return result;
}

std::vector<int> genTileStarts(int length, int window, int overlap) {
    const int stride = std::max(1, window - overlap);
    std::vector<int> starts;

    for (int pos = 0; pos < std::max(1, length - window + 1); pos += stride) {
        starts.push_back(pos);
    }

    const int finalStart = std::max(0, length - window);
    if (starts.empty() || starts.back() != finalStart) {
        starts.push_back(finalStart);
    }

    return starts;
}

std::vector<TileBox> selectChunkTiles(const cv::Mat &baseMaskImage) {
    cv::Mat binMask;
    cv::threshold(baseMaskImage, binMask, kChunkBinThresh, 1, cv::THRESH_BINARY);

    cv::Mat invBinMask = 1 - binMask;
    cv::Mat distFg;
    cv::Mat distBg;
    cv::distanceTransform(binMask, distFg, cv::DIST_L2, 3);
    cv::distanceTransform(invBinMask, distBg, cv::DIST_L2, 3);

    cv::Mat distToBoundary;
    cv::min(distFg, distBg, distToBoundary);

    const int height = baseMaskImage.rows;
    const int width = baseMaskImage.cols;
    const std::vector<int> xs = genTileStarts(width, kChunkTileSize, kChunkOverlap);
    const std::vector<int> ys = genTileStarts(height, kChunkTileSize, kChunkOverlap);

    const int total = kChunkTileSize * kChunkTileSize;
    std::vector<std::pair<TileBox, float>> edgeTiles;
    std::vector<TileBox> posPureTiles;
    std::vector<TileBox> negPureTiles;

    for (int y0 : ys) {
        for (int x0 : xs) {
            const int x1 = x0 + kChunkTileSize;
            const int y1 = y0 + kChunkTileSize;
            if (x1 > width || y1 > height) {
                continue;
            }

            cv::Mat tileBin = binMask(cv::Rect(x0, y0, kChunkTileSize, kChunkTileSize));
            const int pos = cv::countNonZero(tileBin);
            const int neg = total - pos;
            const float posFrac = static_cast<float>(pos) / static_cast<float>(total);

            if (posFrac < kChunkMinPosFrac) {
                continue;
            }

            TileBox box{x0, y0, x1, y1};
            if (pos == total) {
                posPureTiles.push_back(box);
                continue;
            }
            if (neg == total) {
                negPureTiles.push_back(box);
                continue;
            }

            cv::Mat edgeMask;
            cv::compare(
                distToBoundary(cv::Rect(x0, y0, kChunkTileSize, kChunkTileSize)),
                kChunkEdgeBandPx,
                edgeMask,
                cv::CMP_LE);
            const float edgeFrac =
                static_cast<float>(cv::countNonZero(edgeMask)) / static_cast<float>(total);
            edgeTiles.push_back({box, edgeFrac});
        }
    }

    std::sort(
        edgeTiles.begin(),
        edgeTiles.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });

    const int poolSize = static_cast<int>(edgeTiles.size() + posPureTiles.size() + negPureTiles.size());
    if (poolSize == 0) {
        return {};
    }

    const float weightSum = kChunkEdgeTargetFrac + kChunkPosPureFrac + kChunkNegPureFrac;
    const int targetEdge = std::min(
        static_cast<int>(edgeTiles.size()),
        static_cast<int>(std::lround((kChunkEdgeTargetFrac / weightSum) * poolSize)));
    const int targetPos = std::min(
        static_cast<int>(posPureTiles.size()),
        static_cast<int>(std::lround((kChunkPosPureFrac / weightSum) * poolSize)));
    const int targetNeg = std::min(
        static_cast<int>(negPureTiles.size()),
        static_cast<int>(std::lround((kChunkNegPureFrac / weightSum) * poolSize)));

    std::vector<TileBox> picked;
    for (int i = 0; i < targetEdge; ++i) {
        picked.push_back(edgeTiles[i].first);
    }
    picked.insert(picked.end(), posPureTiles.begin(), posPureTiles.begin() + targetPos);
    picked.insert(picked.end(), negPureTiles.begin(), negPureTiles.begin() + targetNeg);

    for (int i = targetEdge; i < static_cast<int>(edgeTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i) {
        picked.push_back(edgeTiles[i].first);
    }
    for (int i = targetPos; i < static_cast<int>(posPureTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i) {
        picked.push_back(posPureTiles[i]);
    }
    for (int i = targetNeg; i < static_cast<int>(negPureTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i) {
        picked.push_back(negPureTiles[i]);
    }

    return picked;
}

cv::Mat makeBlendWindow(int size) {
    cv::Mat win1d(1, size, CV_32FC1);
    for (int i = 0; i < size; ++i) {
        const float value =
            0.5f - 0.5f * std::cos(
                2.0f * static_cast<float>(CV_PI) * static_cast<float>(i) /
                static_cast<float>(size - 1));
        win1d.at<float>(0, i) = std::max(value, 1e-3f);
    }

    cv::Mat win2d = win1d.t() * win1d;
    double maxValue = 1.0;
    cv::minMaxLoc(win2d, nullptr, &maxValue);
    win2d /= static_cast<float>(maxValue);
    return win2d;
}

int maskOutputHeight(const std::vector<std::int64_t> &shape, int fallback) {
    if (shape.size() >= 2 && shape[shape.size() - 2] > 0) {
        return static_cast<int>(shape[shape.size() - 2]);
    }
    return fallback;
}

int maskOutputWidth(const std::vector<std::int64_t> &shape, int fallback) {
    if (shape.size() >= 1 && shape[shape.size() - 1] > 0) {
        return static_cast<int>(shape[shape.size() - 1]);
    }
    return fallback;
}

cv::Mat outputToMask(Ort::Value &output, int fallbackSize, bool normalize) {
    float *outData = output.GetTensorMutableData<float>();
    const auto outputInfo = output.GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> outputShape = outputInfo.GetShape();
    const int maskHeight = maskOutputHeight(outputShape, fallbackSize);
    const int maskWidth = maskOutputWidth(outputShape, fallbackSize);

    float minValue = 0.0f;
    float maxValue = 1.0f;
    if (normalize) {
        minValue = std::numeric_limits<float>::max();
        maxValue = std::numeric_limits<float>::lowest();
        for (int i = 0; i < maskHeight * maskWidth; ++i) {
            minValue = std::min(minValue, outData[i]);
            maxValue = std::max(maxValue, outData[i]);
        }
    }

    const float range = maxValue - minValue;
    cv::Mat mask(maskHeight, maskWidth, CV_8UC1);
    for (int y = 0; y < maskHeight; ++y) {
        std::uint8_t *line = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < maskWidth; ++x) {
            const int idx = y * maskWidth + x;
            float value = outData[idx];
            if (normalize) {
                value = range > std::numeric_limits<float>::epsilon()
                    ? (outData[idx] - minValue) / range
                    : outData[idx];
            }
            line[x] = static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f);
        }
    }

    return mask;
}

class OnnxSegmentationSession : public SegmentationSession {
public:
    OnnxSegmentationSession(
        const std::string &modelPath,
        Ort::Env &env,
        int fallbackInputSize,
        bool allowGpu
    )
        : inputSize_(fallbackInputSize),
          session_(createSession(modelPath, env, allowGpu)) {
        readInputSize(fallbackInputSize);
    }

protected:
    static Ort::Session createSession(const std::string &modelPath, Ort::Env &env, bool allowGpu) {
        auto createWithOptions = [&](bool enableGpu) {
            Ort::SessionOptions options = makeSessionOptions(enableGpu);
#ifdef _WIN32
            const std::wstring wideModelPath(modelPath.begin(), modelPath.end());
            return Ort::Session(env, wideModelPath.c_str(), options);
#else
            return Ort::Session(env, modelPath.c_str(), options);
#endif
        };

        if (!allowGpu) {
            return createWithOptions(false);
        }

        try {
            return createWithOptions(true);
        } catch (const Ort::Exception &err) {
            std::cerr << "GPU session initialization failed, retrying on CPU: "
                      << err.what() << '\n';
            return createWithOptions(false);
        }
    }

    void readInputSize(int fallbackInputSize) {
        const Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
        const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> inputShape = inputTensorInfo.GetShape();

        if (inputShape.size() == 4 && inputShape[2] > 0 && inputShape[3] > 0) {
            inputSize_ = static_cast<int>(inputShape[2]);
        } else {
            inputSize_ = fallbackInputSize;
        }
    }

    cv::Mat runMask(const cv::Mat &inputMat, bool normalizeOutput) {
        std::vector<float> inputTensorValues = matToNchwFloat(inputMat);
        const std::array<std::int64_t, 4> shape{
            1,
            inputMat.channels(),
            inputMat.rows,
            inputMat.cols,
        };

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            shape.data(),
            shape.size());

        auto inputName = session_.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
        auto outputName = session_.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
        const std::array<const char *, 1> inputNames{inputName.get()};
        const std::array<const char *, 1> outputNames{outputName.get()};

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            1,
            outputNames.data(),
            1);

        return outputToMask(outputs.front(), inputSize_, normalizeOutput);
    }

    int inputSize_;
    Ort::Session session_;
};

class U2NetSession final : public OnnxSegmentationSession {
public:
    U2NetSession(
        const std::string &modelPath,
        Ort::Env &env,
        int fallbackInputSize,
        bool allowGpu,
        std::string name
    )
        : OnnxSegmentationSession(modelPath, env, fallbackInputSize, allowGpu),
          modelName_(std::move(name)) {
    }

    std::string displayName() const override {
        return modelName_;
    }

    cv::Mat predictMask(const cv::Mat &rgbImage) override {
        return predictMaskFromMat(rgbImage);
    }

    cv::Mat predictMaskFromMat(const cv::Mat &inputRgbOrRgba) {
        const cv::Mat inputMat = resizeMat(inputRgbOrRgba, inputSize_, inputSize_);
        const cv::Mat mask = runMask(inputMat, true);
        return resizeMaskToOriginal(mask, inputRgbOrRgba);
    }

private:
    std::string modelName_;
};

class BiRefNetSession final : public OnnxSegmentationSession {
public:
    BiRefNetSession(const std::string &modelPath, Ort::Env &env, bool allowGpu)
        : OnnxSegmentationSession(modelPath, env, 1440, allowGpu) {
    }

    std::string displayName() const override {
        return "BiRefNet";
    }

    cv::Mat predictMask(const cv::Mat &rgbImage) override {
        const PaddedSquareImage paddedInput = makePaddedSquareImage(rgbImage, inputSize_, CV_8UC3);
        const cv::Mat mask = runMask(paddedInput.image, false);

        const int croppedWidth = std::min(paddedInput.contentWidth, mask.cols);
        const int croppedHeight = std::min(paddedInput.contentHeight, mask.rows);
        const cv::Mat croppedMask = mask(cv::Rect(0, 0, croppedWidth, croppedHeight));
        return resizeMaskToOriginal(croppedMask, rgbImage);
    }
};

class U2NetPChunkRefinementSession final : public ChunkRefinementSession {
public:
    U2NetPChunkRefinementSession(const std::string &modelPath, Ort::Env &env, bool allowGpu)
        : tileSession_(modelPath, env, kChunkTileSize, allowGpu, "U2NetP Chunks") {
    }

    cv::Mat refineMask(
        const cv::Mat &rgbImage,
        const cv::Mat &baseMask,
        const std::function<void(int, int)> &progressCallback
    ) override {
        const std::vector<TileBox> boxes = selectChunkTiles(baseMask);
        if (boxes.empty()) {
            std::cerr << "No refinement tiles selected; returning empty mask\n";
            return cv::Mat::zeros(baseMask.rows, baseMask.cols, CV_8UC1);
        }

        if (rgbImage.cols < kChunkTileSize || rgbImage.rows < kChunkTileSize) {
            std::cerr << "Image smaller than chunk size; returning empty mask\n";
            return cv::Mat::zeros(rgbImage.rows, rgbImage.cols, CV_8UC1);
        }

        cv::Mat acc = cv::Mat::zeros(rgbImage.rows, rgbImage.cols, CV_32FC1);
        cv::Mat weightSum = cv::Mat::zeros(rgbImage.rows, rgbImage.cols, CV_32FC1);
        const cv::Mat blendWindow = makeBlendWindow(kChunkTileSize);

        for (int tileIndex = 0; tileIndex < static_cast<int>(boxes.size()); ++tileIndex) {
            if (progressCallback) {
                progressCallback(tileIndex + 1, static_cast<int>(boxes.size()));
            }

            const TileBox &box = boxes[tileIndex];
            const cv::Mat tileMat =
                rgbImage(cv::Rect(box.x0, box.y0, kChunkTileSize, kChunkTileSize)).clone();
            const cv::Mat tileBaseMask =
                baseMask(cv::Rect(box.x0, box.y0, kChunkTileSize, kChunkTileSize)).clone();

            std::vector<cv::Mat> rgbChannels;
            cv::split(tileMat, rgbChannels);
            rgbChannels.push_back(tileBaseMask);

            cv::Mat tileWithMask;
            cv::merge(rgbChannels, tileWithMask);

            cv::Mat tileMask = tileSession_.predictMaskFromMat(tileWithMask);
            if (tileMask.cols != kChunkTileSize || tileMask.rows != kChunkTileSize) {
                tileMask = resizeMat(tileMask, kChunkTileSize, kChunkTileSize);
            }

            cv::Mat tileFloat;
            tileMask.convertTo(tileFloat, CV_32FC1);

            const cv::Rect roi(box.x0, box.y0, kChunkTileSize, kChunkTileSize);
            acc(roi) += tileFloat.mul(blendWindow);
            weightSum(roi) += blendWindow;
        }

        cv::Mat safeWeightSum = weightSum.clone();
        cv::max(safeWeightSum, 1e-6f, safeWeightSum);

        cv::Mat refinedFloat;
        cv::divide(acc, safeWeightSum, refinedFloat);

        cv::Mat refinedMask;
        refinedFloat.convertTo(refinedMask, CV_8UC1);

        cv::morphologyEx(
            refinedMask,
            refinedMask,
            cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        cv::GaussianBlur(
            refinedMask,
            refinedMask,
            cv::Size(3, 3),
            1.0,
            1.0,
            cv::BORDER_DEFAULT);

        return refinedMask;
    }

private:
    U2NetSession tileSession_;
};

}  // namespace

Ort::Env createOrtEnv() {
    return Ort::Env(ORT_LOGGING_LEVEL_WARNING, "skin-segmentation-cpp-demo");
}

const ModelConfig &modelConfigFor(ModelKind kind) {
    for (const ModelConfig &config : kModelConfigs) {
        if (config.kind == kind) {
            return config;
        }
    }
    throw std::invalid_argument("Unknown model kind");
}

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env,
    bool allowGpu
) {
    const ModelConfig &config = modelConfigFor(kind);
    switch (kind) {
    case ModelKind::U2Net:
    case ModelKind::U2NetP:
    case ModelKind::U2NetPChunks:
        return std::make_unique<U2NetSession>(
            config.modelPath,
            env,
            config.fallbackInputSize,
            allowGpu,
            config.sessionName);
    case ModelKind::BiRefNet:
        return std::make_unique<BiRefNetSession>(config.modelPath, env, allowGpu);
    }

    throw std::invalid_argument("Unknown model kind");
}

std::unique_ptr<ChunkRefinementSession> createChunkRefinementSession(
    Ort::Env &env,
    bool allowGpu
) {
    return std::make_unique<U2NetPChunkRefinementSession>(
        kU2NetPChunksModelPath,
        env,
        allowGpu);
}
