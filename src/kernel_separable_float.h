#ifndef ASS4_KERNEL_SEPARABLE_FLOAT_H
#define ASS4_KERNEL_SEPARABLE_FLOAT_H

#include <cassert>
#include <limits>
#include "image.h"

template<int SuppHalf>
class ThreeDKernelSeparable {
public:
#ifndef __CUDACC__

    ThreeDKernelSeparable(float *kdata) {
        float weight_total = 0.0;

        for(int i = 0; i < (2 * SuppHalf + 1) * (2 * SuppHalf + 1) * (2 * SuppHalf + 1); i++) {
            kernel[i] = kdata[i];
            weight_total += kernel[i];
        }
        assert((weight_total - 1.0) <= std::numeric_limits<float>::epsilon());
    }


    ~ThreeDKernelSeparable() {}

#endif

public:

#ifdef __CUDACC__
    __host__ __device__
#endif

    void
    apply(const int &row, const int &col, const int &dept,
          const float *image, float *image_conv,
          const int &imagesize, const int &image_dept, const ImageType it) {

        // kernel pointer
        int pointer = 0;

        // init val
        if(it == PPM) {
            for (int l = 0; l < 3; l++) {
                image_conv[(row * imagesize * image_dept + col * image_dept + dept) * 3 + l] = 0;
            }
        } else {
            image_conv[row * imagesize * image_dept + col * image_dept + dept] = 0.0;
        }

        for (int i = row - SuppHalf; i <= row + SuppHalf; i++) {
            for (int j = col - SuppHalf; j <= col + SuppHalf; j++) {
                if (it == PPM) {
                    for (int k = dept - SuppHalf; k <= dept + SuppHalf; k++, pointer++) {

                        if ((i < 0 || j < 0 || k < 0) || (i >= imagesize) || (j >= imagesize) || (k >= image_dept))
                            continue;

                        for (int l = 0; l < 3; l++) {
                            image_conv[(row * imagesize * image_dept + col * image_dept + dept) * 3 + l] +=
                                    kernel[pointer] * image[(i * imagesize * image_dept + j * image_dept + k) * 3 + l];
                        }
                    }
                } else {
                    for (int k = dept - SuppHalf; k <= dept + SuppHalf; k++, pointer++) {

                        if ((i < 0 || j < 0 || k < 0) || (i >= imagesize) || (j >= imagesize) || (k >= image_dept))
                            continue;

                        image_conv[row * imagesize * image_dept + col * image_dept + dept] +=
                                kernel[pointer] * image[i * imagesize * image_dept + j * image_dept + k];
                    }
                }
            }
        }
    }



public:
    float kernel[(2 * SuppHalf + 1) * (2 * SuppHalf + 1) * (2 * SuppHalf + 1)];

private:

    ThreeDKernelSeparable();

    ThreeDKernelSeparable(const ThreeDKernelSeparable &other);

    ThreeDKernelSeparable &operator=(const ThreeDKernelSeparable &other);
};

#endif
