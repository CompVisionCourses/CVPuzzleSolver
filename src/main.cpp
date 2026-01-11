#include <libimages/algorithms/grayscale.h>
#include <libimages/algorithms/threshold_masking.h>
#include <libimages/algorithms/morphology.h>

#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libbase/runtime_assert.h>
#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include <iostream>

int main() {
    try {
        configureWorkingDirectory();

        Timer total_t;
        Timer t;

        image8u image = load_image("data/00_photo_six_parts_downscaled_x4.jpg");
        auto [w, h, c] = image.size();
        rassert(c == 3, 237045347618912, image.channels());
        std::cout << "image loaded in " << t.elapsed() << " sec" << std::endl;
        debug_io::dump_image("debug/00_input.jpg", image);

        // TODO: convert image to grayscale and save it into debug file

        std::vector<float> intensities_on_border;
        // TODO: find grayscale intensities on borders - they are from dark background
        for (int j = 0; j < h; ++j) {
            // TODO
        }

        // TODO what invariant we can check with rassert about intensities_on_border.size()?
        rassert(intensities_on_border.size() >= 0, 7283197129381312);
        std::cout << "intensities on border: " << stats::summaryStats(intensities_on_border) << std::endl;

        // TODO find background_threshold
        double background_threshold = 0.0;
        std::cout << "background threshold=" << background_threshold << std::endl;

        // TODO build is_foreground_mask + save its visualization on disk + print into log percent of background
        image8u is_foreground_mask;

        // TODO print to console percent of foreground pixels
        // TODO save mask visualization to debug file

        // TODO make mask more smooth thanks to morphology
        // TODO firstly try dilation + erosion, is the mask robust? no outliers?

        std::cout << "processed in " << total_t.elapsed() << " sec" << std::endl;

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
