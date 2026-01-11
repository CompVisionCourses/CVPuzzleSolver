#pragma once

#include <string>
#include <vector>

#include <libbase/point2.h>
#include <libimages/color.h>
#include <libimages/image.h>


std::vector<color8u> extractColors(const image8u &image, const std::vector<point2i> &pixels);

void drawImage(image8u &image, image8u &image_part, point2i offset);

void drawRGBLine(image8u &image, std::vector<color8u> &a, point2i offset, int height);

void drawGraph(image8u &image, std::vector<color8u> &a, point2i offset, int height);

void drawGraph(image8u &image, std::vector<float> &a, point2i offset, int height, float maxValue=-1.0f);

// pad with zeros so that string has at least minLength symbols
std::string pad(int v, int minLength);
