// downsample_tests.cpp
#include "downsample.h"

#include <gtest/gtest.h>

#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/tests_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline void set_rgb(image8u& img, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    img(y, x, 0) = r;
    img(y, x, 1) = g;
    img(y, x, 2) = b;
}

static std::vector<color8u> makeRedGradient(int m) {
    std::vector<color8u> v;
    v.reserve(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        const uint8_t r = static_cast<uint8_t>(std::lround(255.0 * i / std::max(1, m - 1)));
        v.push_back(color8u{r, 0, 0});
    }
    return v;
}

static std::vector<int> expectedIndicesRound(int m, int n) {
    std::vector<int> idx;
    idx.reserve(static_cast<size_t>(n));
    if (n <= 0 || m <= 0) return idx;

    if (n == 1) {
        idx.push_back(m / 2);
        return idx;
    }

    for (int i = 0; i < n; ++i) {
        const double pos = (static_cast<double>(i) * static_cast<double>(m - 1)) / static_cast<double>(n - 1);
        int k = static_cast<int>(std::lround(pos));
        k = std::max(0, std::min(m - 1, k));
        idx.push_back(k);
    }
    return idx;
}

static std::vector<color8u> stretchToWidth(const std::vector<color8u>& small, int W) {
    std::vector<color8u> out;
    out.resize(static_cast<size_t>(std::max(0, W)));
    if (small.empty() || W <= 0) return out;
    if (W == 1) {
        out[0] = small[small.size() / 2];
        return out;
    }
    const int n = static_cast<int>(small.size());
    for (int x = 0; x < W; ++x) {
        const double pos = (static_cast<double>(x) * static_cast<double>(n - 1)) / static_cast<double>(W - 1);
        int k = static_cast<int>(std::lround(pos));
        k = std::max(0, std::min(n - 1, k));
        out[static_cast<size_t>(x)] = small[static_cast<size_t>(k)];
    }
    return out;
}

static image8u visualizeColorDownsample(const std::vector<color8u>& src, const std::vector<color8u>& ds) {
    const int W = static_cast<int>(src.size());
    const int H = 30;
    image8u vis(W, H, 3);
    vis.fill(0);

    const int bandH = 10;

    // Top: src colors
    for (int x = 0; x < W; ++x) {
        auto c = src[static_cast<size_t>(x)];
        for (int y = 0; y < bandH; ++y) {
            set_rgb(vis, x, y, c(0), c(1), c(2));
        }
    }

    // Middle: downsampled stretched to width
    auto stretched = stretchToWidth(ds, W);
    for (int x = 0; x < W; ++x) {
        auto c = stretched[static_cast<size_t>(x)];
        for (int y = bandH; y < 2 * bandH; ++y) {
            set_rgb(vis, x, y, c(0), c(1), c(2));
        }
    }

    // Bottom: markers at selected indices (white ticks)
    // compute chosen indices by matching exact colors in src (safe here because we made unique gradient)
    // just mark the stretched transitions, and also mark expected picks using a search
    for (int x = 0; x < W; ++x) {
        for (int y = 2 * bandH; y < 3 * bandH; ++y) {
            set_rgb(vis, x, y, 0, 0, 0);
        }
    }
    // mark expected chosen indices by nearest match to src
    for (const auto& c : ds) {
        int bestX = 0;
        int bestD = 1e9;
        for (int x = 0; x < W; ++x) {
            const auto s = src[static_cast<size_t>(x)];
            const int dr = std::abs(int(s(0)) - int(c(0)));
            const int dg = std::abs(int(s(1)) - int(c(1)));
            const int db = std::abs(int(s(2)) - int(c(2)));
            const int d = dr + dg + db;
            if (d < bestD) { bestD = d; bestX = x; }
        }
        for (int y = 2 * bandH; y < 3 * bandH; ++y) {
            set_rgb(vis, bestX, y, 255, 255, 255);
        }
    }

    return vis;
}

} // namespace

TEST(downsample, colors_preserve_endpoints_and_round_mapping) {
    configureWorkingDirectory();

    const int m = 10;
    const int n = 4;

    auto src = makeRedGradient(m);
    auto ds = downsample(src, n);

    ASSERT_EQ(static_cast<int>(ds.size()), n);

    // endpoints preserved for n>=2
    EXPECT_EQ(ds.front()(0), src.front()(0));
    EXPECT_EQ(ds.back()(0),  src.back()(0));

    // indices exactly as round-mapping
    auto idx = expectedIndicesRound(m, n);
    for (int i = 0; i < n; ++i) {
        const auto& a = ds[static_cast<size_t>(i)];
        const auto& b = src[static_cast<size_t>(idx[static_cast<size_t>(i)])];
        for (int c = 0; c < 3; ++c) {
            EXPECT_EQ(a(c), b(c));
        }
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "00_colors_src_vs_ds.png",
                         visualizeColorDownsample(src, ds));
}

TEST(downsample, colors_n1_takes_middle) {
    configureWorkingDirectory();

    const int m = 9;
    auto src = makeRedGradient(m);

    auto ds = downsample(src, 1);
    ASSERT_EQ(ds.size(), 1u);

    const int mid = m / 2;
    for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(ds[0](c), src[static_cast<size_t>(mid)](c));
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "00_colors_src_vs_ds.png",
                         visualizeColorDownsample(src, ds));
}

TEST(downsample, image_gray_5x5_to_3x3_round_mapping) {
    configureWorkingDirectory();

    image8u src(5, 5, 1);
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            src(y, x) = static_cast<uint8_t>(x + 10 * y);
        }
    }

    auto ds = downsample(src, 3, 3);

    ASSERT_EQ(ds.width(), 3);
    ASSERT_EQ(ds.height(), 3);
    ASSERT_EQ(ds.channels(), 1);

    // expected mapping for 5->3: indices [0,2,4]
    EXPECT_EQ(ds(0, 0), src(0, 0));
    EXPECT_EQ(ds(0, 1), src(0, 2));
    EXPECT_EQ(ds(0, 2), src(0, 4));

    EXPECT_EQ(ds(1, 0), src(2, 0));
    EXPECT_EQ(ds(1, 1), src(2, 2));
    EXPECT_EQ(ds(1, 2), src(2, 4));

    EXPECT_EQ(ds(2, 0), src(4, 0));
    EXPECT_EQ(ds(2, 1), src(4, 2));
    EXPECT_EQ(ds(2, 2), src(4, 4));

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src.png", src);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_ds_3x3.png", ds);
}

TEST(downsample, image_rgb_preserves_channels) {
    configureWorkingDirectory();

    image8u src(4, 3, 3);
    src.fill(0);

    // Encode position into channels to verify correct sampling.
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            src(y, x, 0) = static_cast<uint8_t>(10 * x);
            src(y, x, 1) = static_cast<uint8_t>(20 * y);
            src(y, x, 2) = static_cast<uint8_t>(x + y);
        }
    }

    auto ds = downsample(src, 2, 2);
    ASSERT_EQ(ds.channels(), 3);

    // 4->2 => x indices [0,3]; 3->2 => y indices [0,2]
    EXPECT_EQ(ds(0, 0, 0), src(0, 0, 0));
    EXPECT_EQ(ds(0, 0, 1), src(0, 0, 1));
    EXPECT_EQ(ds(0, 0, 2), src(0, 0, 2));

    EXPECT_EQ(ds(0, 1, 0), src(0, 3, 0));
    EXPECT_EQ(ds(0, 1, 1), src(0, 3, 1));
    EXPECT_EQ(ds(0, 1, 2), src(0, 3, 2));

    EXPECT_EQ(ds(1, 0, 0), src(2, 0, 0));
    EXPECT_EQ(ds(1, 0, 1), src(2, 0, 1));
    EXPECT_EQ(ds(1, 0, 2), src(2, 0, 2));

    EXPECT_EQ(ds(1, 1, 0), src(2, 3, 0));
    EXPECT_EQ(ds(1, 1, 1), src(2, 3, 1));
    EXPECT_EQ(ds(1, 1, 2), src(2, 3, 2));

    debug_io::dump_image(getUnitCaseDebugDir() + "00_src.png", src);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_ds_2x2.png", ds);
}
