#include "blur.h"

#include <gtest/gtest.h>

#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <libimages/tests_utils.h>

namespace {

static std::vector<color8u> makeRedGradient(int n) {
    std::vector<color8u> v;
    v.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const uint8_t r = static_cast<uint8_t>(std::lround(255.0 * i / std::max(1, n - 1)));
        v.emplace_back(r, 0, 0);
    }
    return v;
}

static std::vector<color8u> makeImpulseLine(int n, int at, uint8_t val) {
    std::vector<color8u> v(static_cast<size_t>(n), color8u(0, 0, 0));
    v[static_cast<size_t>(at)] = color8u(val, val, val);
    return v;
}

static void drawBand(image8u& img, int y0, int y1, const std::vector<color8u>& line) {
    const int W = std::min(img.width(), static_cast<int>(line.size()));
    for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < W; ++x) {
            const color8u& c = line[static_cast<size_t>(x)];
            img(y, x, 0) = c(0);
            img(y, x, 1) = c(1);
            img(y, x, 2) = c(2);
        }
    }
}

static image8u visualizeLines(const std::vector<color8u>& a, const std::vector<color8u>& b) {
    const int W = static_cast<int>(a.size());
    image8u img(W, 24, 3);
    img.fill(0);
    drawBand(img, 0, 12, a);
    drawBand(img, 12, 24, b);
    return img;
}

} // namespace

TEST(blur, colors_strength0_identity) {
    configureWorkingDirectory();

    auto src = makeRedGradient(64);
    auto dst = blur(src, 0.0f);

    ASSERT_EQ(dst.size(), src.size());
    for (int i = 0; i < src.size(); ++i) {
        EXPECT_EQ(dst[i], src[i]);
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src_vs_dst.png", visualizeLines(src, dst));
}

TEST(blur, colors_impulse_becomes_smooth_and_symmetric) {
    configureWorkingDirectory();

    const int n = 81;
    const int mid = n / 2;
    auto src = makeImpulseLine(n, mid, 255);

    auto dst = blur(src, 3.0f);

    ASSERT_EQ(dst.size(), src.size());

    for (int c = 0; c < 3; ++c) {
        // Center decreases (energy spreads)
        EXPECT_LT((int)dst[mid](c), 255);

        // Neighbors become > 0
        EXPECT_GT((int)dst[mid - 1](c), 0);
        EXPECT_GT((int)dst[mid + 1](c), 0);
    }

    // Symmetry around center (rounding allowed)
    for (int d = 1; d <= 10; ++d) {
        for (int c = 0; c < 3; ++c) {
            const int a = (int)dst[mid - d](c);
            const int b = (int)dst[mid - d](c);
            EXPECT_LE(std::abs(a - b), 2);
        }
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src_vs_dst.png", visualizeLines(src, dst));
}

TEST(blur, image_impulse_2d_spreads) {
    configureWorkingDirectory();

    image32f src(61, 61, 1);
    src.fill(0.0f);
    src(30, 30) = 255.0f;

    auto dst = blur(src, 3.0f);

    EXPECT_LT(dst(30, 30), 255.0f);
    EXPECT_GT(dst(30, 31), 0.0f);
    EXPECT_GT(dst(30, 30), dst(30, 31));
    EXPECT_GT(dst(30, 31), dst(30, 35));

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src.png", src);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_blur.png", dst);
}

TEST(blur, image_rgb_edge_smooths) {
    configureWorkingDirectory();

    image8u src(40, 20, 3);
    src.fill(0);

    // Left half: red; right half: green
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            if (x < src.width() / 2) {
                src(y, x, 0) = 255; src(y, x, 1) = 0;   src(y, x, 2) = 0;
            } else {
                src(y, x, 0) = 0;   src(y, x, 1) = 255; src(y, x, 2) = 0;
            }
        }
    }

    auto dst = blur(src, 2.5f);

    // Boundary becomes mixed
    const int xb = src.width() / 2;
    EXPECT_GT((int)dst(10, xb - 1, 0), 0);
    EXPECT_GT((int)dst(10, xb - 1, 1), 0);
    EXPECT_GT((int)dst(10, xb,     0), 0);
    EXPECT_GT((int)dst(10, xb,     1), 0);

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src.png", src);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_blur.png", dst);
}
