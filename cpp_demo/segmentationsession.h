#pragma once

#include <functional>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "ortcompat.h"

enum class ModelKind {
    U2Net,
    U2NetP,
    BiRefNet,
    U2NetPChunks,
};

struct ModelConfig {
    ModelKind kind;
    const char *menuLabel;
    const char *sessionName;
    const char *modelPath;
    int fallbackInputSize;
};

class SegmentationSession {
public:
    virtual ~SegmentationSession() = default;

    virtual std::string displayName() const = 0;
    virtual cv::Mat predictMask(const cv::Mat &rgbImage) = 0;
};

class ChunkRefinementSession {
public:
    virtual ~ChunkRefinementSession() = default;

    virtual cv::Mat refineMask(
        const cv::Mat &rgbImage,
        const cv::Mat &baseMask,
        const std::function<void(int, int)> &progressCallback
    ) = 0;
};

Ort::Env createOrtEnv();

const ModelConfig &modelConfigFor(ModelKind kind);

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env,
    bool allowGpu
);

std::unique_ptr<ChunkRefinementSession> createChunkRefinementSession(
    Ort::Env &env,
    bool allowGpu
);
