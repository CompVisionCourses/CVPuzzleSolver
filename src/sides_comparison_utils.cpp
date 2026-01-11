#include "sides_comparison_utils.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

inline void set_rgb(image8u& img, int x, int y, const color8u& c) {
    img(y, x, 0) = c(0);
    img(y, x, 1) = c(1);
    img(y, x, 2) = c(2);
}

inline void set_rgb(image8u& img, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    img(y, x, 0) = r;
    img(y, x, 1) = g;
    img(y, x, 2) = b;
}

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

inline int sample_index_round(int i, int n, int m) {
    // Map i in [0..n-1] to idx in [0..m-1], preserving endpoints.
    // n>=2, m>=2
    const double pos = (static_cast<double>(i) * static_cast<double>(m - 1)) / static_cast<double>(n - 1);
    int idx = static_cast<int>(std::lround(pos));
    return clampi(idx, 0, m - 1);
}

} // namespace

std::vector<color8u> extractColors(const image8u &image, const std::vector<point2i> &pixels) {
    rassert(image.channels() == 1 || image.channels() == 3, 983417231, image.channels());

    std::vector<color8u> out;
    out.reserve(pixels.size());

    const int w = image.width();
    const int h = image.height();
    const int c = image.channels();

    for (const auto& p : pixels) {
        rassert(p.x >= 0 && p.x < w && p.y >= 0 && p.y < h, 983417232);

        if (c == 3) {
            out.emplace_back(image(p.y, p.x, 0), image(p.y, p.x, 1), image(p.y, p.x, 2));
        } else {
            uint8_t v = image(p.y, p.x);
            out.emplace_back(v, v, v);
        }
    }

    return out;
}

void drawImage(image8u &image, image8u &image_part, point2i offset) {
    rassert(offset.y + image_part.height() <= image.height(), 1231412431);
    rassert(offset.x + image_part.width() <= image.width(), 64534524523);
    rassert(image.channels() == image_part.channels(), 3427823974238);
    for (int j = 0; j < image_part.height(); ++j) {
        for (int i = 0; i < image_part.width(); ++i) {
            for (int c = 0; c < image_part.channels(); ++c) {
                image(offset.y + j, offset.x + i, c) = image_part(j, i, c);
            }
        }
    }
}

void drawRGBLine(image8u &image, std::vector<color8u> &a, point2i offset, int height) {
    rassert(image.channels() == 3, 981273641);

    rassert(offset.y + height <= image.height(), 1231412445631);
    rassert(offset.x + a.size() <= image.width(), 64534524522353);

    for (int x = 0; x < a.size(); ++x) {
        for (int y = 0; y < height; ++y) {
            set_rgb(image, offset.x + x, offset.y + y, a[x]);
        }
    }
}

void drawGraph(image8u &image, std::vector<color8u> &a, point2i offset, int height) {
    rassert(image.channels() == 3, 981273642);

    rassert(offset.y + height <= image.height(), 54656234);
    rassert(offset.x + a.size() <= image.width(), 247426342);

    auto y_from_val = [&](int v) -> int {
        v = clampi(v, 0, 255);
        const double t = static_cast<double>(v) / 255.0; // 0..1
        const int yy = (height - 1 - static_cast<int>(std::lround(t * (height - 1))));
        rassert(yy >= 0 && yy < height, 43524231412);
        return yy;
    };

    for (int x = 0; x < a.size(); ++x) {
        const int yr = y_from_val(a[x](0));
        const int yg = y_from_val(a[x](1));
        const int yb = y_from_val(a[x](2));

        // R channel graph (red)
        set_rgb(image, offset.x + x, offset.y + yr, 255, 0, 0);
        // G channel graph (green)
        set_rgb(image, offset.x + x, offset.y + yg, 0, 255, 0);
        // B channel graph (blue)
        set_rgb(image, offset.x + x, offset.y + yb, 0, 0, 255);
    }
}

void drawGraph(image8u &image, std::vector<float> &a, point2i offset, int height, float maxValue) {
    rassert(image.channels() == 3, 981273643);

    rassert(offset.y + height <= image.height(), 45654732452);
    rassert(offset.x + a.size() <= image.width(), 6584563634);

    if (maxValue == -1.0f) {
        maxValue = 0.0f;
        for (float v : a) maxValue = std::max(maxValue, v);
        if (maxValue <= 0.0f) maxValue = 1.0f;
    }

    auto y_from_val = [&](float v) -> int {
        float t = v / maxValue;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        const int yy = (height - 1 - static_cast<int>(std::lround(t * (height - 1))));
        rassert(yy >= 0 && yy < height, 34718947128321);
        return yy;
    };

    for (int x = 0; x < a.size(); ++x) {
        const int yy = y_from_val(a[x]);
        set_rgb(image, offset.x + x, offset.y + yy, 255, 255, 255);
    }
}

// pad with zeros so that string has at least minLength symbols
std::string pad(int v, int minLength) {
    const std::string s = std::to_string(v);
    if (s.size() >= minLength) return s;
    return std::string(minLength - s.size(), '0') + s;
}
