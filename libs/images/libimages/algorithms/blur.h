#pragma once

#include <vector>

#include <libimages/color.h>
#include <libimages/image.h>

template <typename T>
Image<T> blur(const Image<T> &image, float strength);

template <typename T>
std::vector<Color<T>> blur(const std::vector<Color<T>> &colors, float strength);
