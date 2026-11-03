// Image loading helpers: file discovery and conversion to NCHW float tensors.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "common.hpp"

namespace image_utils {

// Recursively lists every .jpg/.jpeg/.png/.bmp/.ppm file under `folder`
// (sorted, so results are reproducible). Throws if the folder does not exist.
std::vector<std::string> list_images(const std::string& folder);

// Loads an image as a 1xCxHxW float tensor.
//  - color (default): RGB, scaled to [0,1], then normalised with the ImageNet
//    mean/std, which is what torchvision models expect (C = 3).
//  - grayscale: scaled to [0,1] only (C = 1).
// If `resize_to` is non-empty the image is resized to that size first.
Tensor4D load_image(const std::string& path, bool grayscale = false,
                    const cv::Size& resize_to = cv::Size());

// Loads several images and stacks them into one NxCxHxW tensor. All images must
// end up with the same size (pass `resize_to` if they do not).
Tensor4D load_batch(const std::vector<std::string>& paths, bool grayscale = false,
                    const cv::Size& resize_to = cv::Size());

}  // namespace image_utils
