#ifndef ASS4_KERNEL_SEPARABLE_HALF_H
#define ASS4_KERNEL_SEPARABLE_HALF_H


#include "image.h"
#include <cuda_fp16.h>

template<int SuppHalf>
class HalfKernelSeparable {
public:
#ifndef __CUDACC__

    HalfKernelSeparable(__half *kdata) {
        __half weight_total = 0.0;

        for(int i = 0; i < (2 * SuppHalf + 1) * (2 * SuppHalf + 1) * (2 * SuppHalf + 1); i++) {
            kernel[i] = kdata[i];
            weight_total += kernel[i];
        }
        //assert((__hadd(weight_total,__float2half(-1.0f))) <= std::numeric_limits<__half>::epsilon());
    }


    ~HalfKernelSeparable() {}

#endif

public:

#ifdef __CUDACC__
    __host__ __device__
#endif

    void
    apply(const int &row, const int &col, const int &dept,
          const __half *image, __half *image_conv,
          const int &imagesize, const int &image_dept, const ImageType it) {

        // kernel pointer
        int pointer = 0;

        // init val
        image_conv[row * imagesize * image_dept + col * image_dept + dept] = __float2half(0.0); // init val

        for (int i = row - SuppHalf; i <= row + SuppHalf; i++) {
            for (int j = col - SuppHalf; j <= col + SuppHalf; j++) {

                    for (int k = dept - SuppHalf; k <= dept + SuppHalf; k++, pointer++) {

                        if ((i < 0 || j < 0 || k < 0) || (i >= imagesize) || (j >= imagesize) || (k >= image_dept))
                            continue;

                        image_conv[row * imagesize * image_dept + col * image_dept + dept] = __hadd(image_conv[row * imagesize * image_dept + col * image_dept + dept],__hmul(kernel[pointer],image[i * imagesize * image_dept + j * image_dept + k]));
                    }
            }
        }
    }



public:
    __half kernel[(2 * SuppHalf + 1) * (2 * SuppHalf + 1) * (2 * SuppHalf + 1)];

private:

    HalfKernelSeparable();

    HalfKernelSeparable(const HalfKernelSeparable &other);

    HalfKernelSeparable &operator=(const HalfKernelSeparable &other);
};

#endif
