#pragma once

#include <vector>

#include <libimages/color.h>
#include <libimages/image.h>

template <typename T>
Image<T> downsample(const Image<T> &image, int w, int h);

template <typename T>
std::vector<Color<T>> downsample(const std::vector<Color<T>> &colors, int n);
