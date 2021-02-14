#include <iostream>
#include "kernel_separable_float.h"
#include "image.h"

void
convSeparableHost(float* kdata, const int& kernel_supp_half, const Image& image, Image& image_conv) {
    if(1 == kernel_supp_half) {
        ThreeDKernelSeparable<1> kernel(kdata);
        for(int row = 0; row < image.n_rows; row++) {
            for(int col = 0; col < image.n_cols; col++) {
                for(int dept = 0; dept < image.n_depts; dept++) {
                    kernel.apply(row, col, dept, &image.data[0], &image_conv.data[0], image.n_rows, image.n_depts, image.it);
                }
            }
        }
    }
    else if(2 == kernel_supp_half){
        ThreeDKernelSeparable<2> kernel(kdata);
        for(int row = 0; row < image.n_rows; row++) {
            for(int col = 0; col < image.n_cols; col++) {
                for(int dept = 0; dept < image.n_depts; dept++) {
                    kernel.apply(row, col, dept, &image.data[0], &image_conv.data[0], image.n_rows, image.n_depts, image.it);
                }
            }
        }
    } else {
        std::cerr << "convSeparableHost() :: kernel size not implemented." << std::endl;
    }
}