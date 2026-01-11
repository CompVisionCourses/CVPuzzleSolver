// downsample.cpp
#include "downsample.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline int clampi(int v, int lo, int hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

inline int map_index_round(int i, int n, int m) {
    // Map i in [0..n-1] to idx in [0..m-1], preserving endpoints with rounding.
    // n>=2, m>=2
    const double pos = (static_cast<double>(i) * static_cast<double>(m - 1)) / static_cast<double>(n - 1);
    int idx = static_cast<int>(std::lround(pos));
    return clampi(idx, 0, m - 1);
}

template <typename T>
inline int safe_mid_index(int m) {
    return (m <= 0) ? 0 : (m / 2);
}

} // namespace

template <typename T>
Image<T> downsample(const Image<T> &image, int w, int h) {
    rassert(w > 0 && h > 0, 781234981);

    const int srcW = image.width();
    const int srcH = image.height();
    const int ch = image.channels();
    rassert(srcW > 0 && srcH > 0, 781234982);
    rassert(ch == 1 || ch == 3, 781234983, ch);

    Image<T> out(w, h, ch);

    // Handle degenerate mappings (target size 1) by sampling center in that axis.
    const int sx_center = safe_mid_index<T>(srcW);
    const int sy_center = safe_mid_index<T>(srcH);

    for (int y = 0; y < h; ++y) {
        const int sy = (h == 1) ? sy_center : map_index_round(y, h, srcH);
        for (int x = 0; x < w; ++x) {
            const int sx = (w == 1) ? sx_center : map_index_round(x, w, srcW);

            if (ch == 1) {
                out(y, x) = image(sy, sx);
            } else {
                out(y, x, 0) = image(sy, sx, 0);
                out(y, x, 1) = image(sy, sx, 1);
                out(y, x, 2) = image(sy, sx, 2);
            }
        }
    }

    return out;
}

template <typename T>
std::vector<Color<T>> downsample(const std::vector<Color<T>> &colors, int n) {
    if (n <= 0) return {};
    if (colors.empty()) return {};

    const int m = static_cast<int>(colors.size());
    if (n >= m) return colors;

    std::vector<Color<T>> out;
    out.reserve(static_cast<size_t>(n));

    if (n == 1) {
        out.push_back(colors[m / 2]);
        return out;
    }

    for (int i = 0; i < n; ++i) {
        const int idx = map_index_round(i, n, m);
        out.push_back(colors[idx]);
    }

    return out;
}

// ---- explicit instantiations ----
template Image<std::uint8_t> downsample(const Image<std::uint8_t>& image, int w, int h);
template Image<float>        downsample(const Image<float>& image, int w, int h);
template Image<int>          downsample(const Image<int>& image, int w, int h);

template std::vector<Color<std::uint8_t>> downsample(const std::vector<Color<std::uint8_t>>& colors, int n);
template std::vector<Color<float>>        downsample(const std::vector<Color<float>>& colors, int n);
