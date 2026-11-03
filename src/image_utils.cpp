#include "image_utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace image_utils {

std::vector<std::string> list_images(const std::string& folder) {
    if (!fs::is_directory(folder)) {
        throw std::runtime_error("Image folder does not exist: " + folder);
    }

    std::vector<std::string> paths;
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        // Skip macOS resource-fork files such as "._image.png".
        const std::string name = entry.path().filename().string();
        if (name.rfind("._", 0) == 0) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".ppm") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

Tensor4D load_image(const std::string& path, bool grayscale, const cv::Size& resize_to) {
    cv::Mat img = cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    if (resize_to.area() > 0 && img.size() != resize_to) {
        cv::resize(img, img, resize_to, 0, 0, cv::INTER_LINEAR);
    }

    Tensor4D out;
    out.n = 1;
    out.c = grayscale ? 1 : 3;
    out.h = img.rows;
    out.w = img.cols;
    out.data.resize(out.size());

    if (grayscale) {
        cv::Mat f32;
        img.convertTo(f32, CV_32FC1, 1.0 / 255.0);
        std::copy(f32.ptr<float>(), f32.ptr<float>() + f32.total(), out.data.begin());
        return out;
    }

    // OpenCV loads BGR; models expect RGB, normalised per channel.
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    img.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdev[3] = {0.229f, 0.224f, 0.225f};

    // Interleaved HWC -> planar CHW while normalising.
    const size_t plane = static_cast<size_t>(out.h) * out.w;
    for (int y = 0; y < out.h; ++y) {
        const float* row = f32.ptr<float>(y);
        for (int x = 0; x < out.w; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                out.data[ch * plane + static_cast<size_t>(y) * out.w + x] =
                    (row[x * 3 + ch] - mean[ch]) / stdev[ch];
            }
        }
    }
    return out;
}

Tensor4D load_batch(const std::vector<std::string>& paths, bool grayscale, const cv::Size& resize_to) {
    if (paths.empty()) {
        throw std::runtime_error("No images to load.");
    }

    Tensor4D batch;
    for (const auto& path : paths) {
        Tensor4D img = load_image(path, grayscale, resize_to);
        if (batch.n == 0) {
            batch.c = img.c;
            batch.h = img.h;
            batch.w = img.w;
            batch.data.reserve(img.size() * paths.size());
        } else if (img.h != batch.h || img.w != batch.w) {
            throw std::runtime_error("Image sizes differ within the batch (" + path +
                                     "). All images in a batch must have the same size.");
        }
        batch.data.insert(batch.data.end(), img.data.begin(), img.data.end());
        ++batch.n;
    }
    return batch;
}

}  // namespace image_utils
