// blur.cpp
#include "blur.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

inline int clampi(int v, int lo, int hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

struct Kernel1D {
    std::vector<float> w;
    int r = 0;
};

inline Kernel1D makeGaussianKernel(float sigma) {
    Kernel1D k;
    if (!(sigma > 0.0f)) return k;

    const float s = std::max(0.001f, sigma);
    k.r = std::max(0, static_cast<int>(std::ceil(3.0f * s)));
    const int R = k.r;

    k.w.resize(static_cast<size_t>(2 * R + 1), 0.0f);

    const float inv2s2 = 1.0f / (2.0f * s * s);
    float sum = 0.0f;
    for (int i = -R; i <= R; ++i) {
        const float v = std::exp(-(float)(i * i) * inv2s2);
        k.w[static_cast<size_t>(i + R)] = v;
        sum += v;
    }
    if (sum > 0.0f) {
        for (float& v : k.w) v /= sum;
    }
    return k;
}

template <typename T>
inline float to_f(T v) noexcept {
    if constexpr (std::is_same_v<T, float>) return v;
    return static_cast<float>(v);
}

template <typename T>
inline T from_f(float v) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        v = std::clamp(v, 0.0f, 255.0f);
        return static_cast<std::uint8_t>(std::lround(v));
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::lround(v));
    } else {
        return static_cast<T>(v);
    }
}

} // namespace

template <typename T>
Image<T> blur(const Image<T> &image, float strength) {
    if (!(strength > 0.0f)) return image;

    const int W = image.width();
    const int H = image.height();
    const int C = image.channels();
    rassert(W > 0 && H > 0, 981234001);
    rassert(C == 1 || C == 3, 981234002, C);

    const Kernel1D k = makeGaussianKernel(strength);
    if (k.r == 0) return image;

    const int R = k.r;

    std::vector<float> tmp(static_cast<size_t>(W) * H * C, 0.0f);

    auto idx = [&](int x, int y, int c) -> size_t {
        return (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * static_cast<size_t>(C)
             + static_cast<size_t>(c);
    };

    // Horizontal pass -> tmp
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (C == 1) {
                float acc = 0.0f;
                for (int dx = -R; dx <= R; ++dx) {
                    const int sx = clampi(x + dx, 0, W - 1);
                    acc += k.w[static_cast<size_t>(dx + R)] * to_f(image(y, sx));
                }
                tmp[idx(x, y, 0)] = acc;
            } else {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
                for (int dx = -R; dx <= R; ++dx) {
                    const int sx = clampi(x + dx, 0, W - 1);
                    const float w = k.w[static_cast<size_t>(dx + R)];
                    acc0 += w * to_f(image(y, sx, 0));
                    acc1 += w * to_f(image(y, sx, 1));
                    acc2 += w * to_f(image(y, sx, 2));
                }
                tmp[idx(x, y, 0)] = acc0;
                tmp[idx(x, y, 1)] = acc1;
                tmp[idx(x, y, 2)] = acc2;
            }
        }
    }

    Image<T> out(W, H, C);

    // Vertical pass: tmp -> out
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (C == 1) {
                float acc = 0.0f;
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = clampi(y + dy, 0, H - 1);
                    acc += k.w[static_cast<size_t>(dy + R)] * tmp[idx(x, sy, 0)];
                }
                out(y, x) = from_f<T>(acc);
            } else {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = clampi(y + dy, 0, H - 1);
                    const float w = k.w[static_cast<size_t>(dy + R)];
                    acc0 += w * tmp[idx(x, sy, 0)];
                    acc1 += w * tmp[idx(x, sy, 1)];
                    acc2 += w * tmp[idx(x, sy, 2)];
                }
                out(y, x, 0) = from_f<T>(acc0);
                out(y, x, 1) = from_f<T>(acc1);
                out(y, x, 2) = from_f<T>(acc2);
            }
        }
    }

    return out;
}

template <typename T>
std::vector<Color<T>> blur(const std::vector<Color<T>> &colors, float strength) {
    if (!(strength > 0.0f)) return colors;
    if (colors.empty()) return {};

    const Kernel1D k = makeGaussianKernel(strength);
    if (k.r == 0) return colors;

    const int R = k.r;
    const int n = static_cast<int>(colors.size());
    const int C = colors[0].channels();
    rassert(C == 1 || C == 3, 981234003, C);

    std::vector<std::vector<float>> tmp(static_cast<size_t>(C), std::vector<float>(static_cast<size_t>(n), 0.0f));

    // 1D convolution with clamp boundary
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < C; ++c) {
            float acc = 0.0f;
            for (int d = -R; d <= R; ++d) {
                const int si = clampi(i + d, 0, n - 1);
                acc += k.w[static_cast<size_t>(d + R)] * to_f(colors[static_cast<size_t>(si)](c));
            }
            tmp[static_cast<size_t>(c)][static_cast<size_t>(i)] = acc;
        }
    }

    std::vector<Color<T>> out;
    out.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        if (C == 1) {
            out.emplace_back(from_f<T>(tmp[0][static_cast<size_t>(i)]));
        } else {
            out.emplace_back(
                from_f<T>(tmp[0][static_cast<size_t>(i)]),
                from_f<T>(tmp[1][static_cast<size_t>(i)]),
                from_f<T>(tmp[2][static_cast<size_t>(i)])
            );
        }
    }

    return out;
}

// explicit instantiations
template Image<std::uint8_t> blur(const Image<std::uint8_t>& image, float strength);
template Image<float>        blur(const Image<float>& image, float strength);

template std::vector<Color<std::uint8_t>> blur(const std::vector<Color<std::uint8_t>>& colors, float strength);
template std::vector<Color<float>>        blur(const std::vector<Color<float>>& colors, float strength);
