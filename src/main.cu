
#include <chrono>
#include "cuda_util.h"
#include <cuda_fp16.h>
#include <fstream>

#include "timer.h"
#include "image.h"
#include "kernel_separable_float.h"
#include "kernel_separable_half.h"

extern void
convSeparableHost(float *kdata, const int &kernel_supp_half, const Image &image, Image &image_conv);

template<int KernelSuppHalf>
__global__
void
convSeparableHalf(const unsigned int image_size, const int image_dept, const ImageType it, __half *gkernel, __half *image, __half *image_conv) {
    __shared__ __half kernel[(KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1)];

    if (((blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z)<
         ((KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1)))) {
        kernel[blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z] = gkernel[
                blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z];
    }

    __syncthreads();

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pixel_x < image_size && pixel_y < image_size && pixel_z < image_dept) {
        HalfKernelSeparable<KernelSuppHalf> *skernel = (HalfKernelSeparable<KernelSuppHalf> *) kernel;
        skernel->apply(pixel_x, pixel_y, pixel_z, image, image_conv, image_size, image_dept, it);
    }
}

template<int KernelSuppHalf, class T>
__global__
void
convSeparable(T *gkernel, T *image, T *image_conv, const unsigned int image_size, const int image_dept, const ImageType it) {

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pixel_x < image_size && pixel_y < image_size && pixel_z < image_dept) {
        ThreeDKernelSeparable<KernelSuppHalf> *skernel = (ThreeDKernelSeparable<KernelSuppHalf> *) gkernel;
        skernel->apply(pixel_x, pixel_y, pixel_z, image, image_conv, image_size, image_dept, it);
    }

}

template<int KernelSuppHalf, class T>
__global__
void
convSeparableShared(T *gkernel, T *gimage, T *image_conv, const unsigned int image_size, const int image_dept, const ImageType it) {
    __shared__ T kernel[(KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1)];

    if (((blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z) <
         ((KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1) * (KernelSuppHalf * 2 + 1)))) {
        kernel[blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z] = gkernel[
                blockDim.z * blockDim.x * threadIdx.y + blockDim.z * threadIdx.x + threadIdx.z];
    }

    __syncthreads();

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (pixel_x < image_size && pixel_y < image_size && pixel_z < image_dept) {
        ThreeDKernelSeparable<KernelSuppHalf> *skernel = (ThreeDKernelSeparable<KernelSuppHalf> *) kernel;
        skernel->apply(pixel_x, pixel_y, pixel_z, gimage, image_conv, image_size, image_dept, it);
    }
}

template<int kernel_supp>
void floatConv(float *kdata, Image &image, std::fstream &file) {

    // set output image
    Image image_conv(image.n_rows, image.n_cols, image.n_depts, image.it);
    int image_size = image.data.size();

    // set kernel layout
    const int kernel_supp_half = kernel_supp / 2;
    const int kernel_size = kernel_supp * kernel_supp * kernel_supp;

    // setup timer
    Timer t = Timer();

    // run on cpu
    t.start();
    convSeparableHost(kdata, kernel_supp_half, image, image_conv);
    std::cerr << "CPU Convolution: ";
    file << t.stop() << " ";
    // delete content of output_cpu folder
    //system("exec rm -r ../images/output_cpu/*");
    t.start();
    image_conv.write("../images/output_cpu/head_cpu");
    std::cerr << "CPU Convolution: Writing: ";
    t.stop();

    // check execution environment
    int device_handle = 0;
    int max_threads_per_block = 0;
    if (!initDevice(device_handle, max_threads_per_block)) {
        exit(EXIT_FAILURE);
    }

    // initialize memory
    float *kernel_device = nullptr;
    float *image_device = nullptr;
    float *image_conv_device = nullptr;

    // allocate device memory
    checkErrorsCuda(cudaMalloc((void **) &kernel_device, sizeof(float) * kernel_size));
    checkErrorsCuda(cudaMalloc((void **) &image_device, sizeof(float) * image_size));
    checkErrorsCuda(
            cudaMalloc((void **) &image_conv_device, sizeof(float) * image_size));


    // copy device memory
    checkErrorsCuda(cudaMemcpy((void *) kernel_device, kdata,
                               sizeof(float) * kernel_size,
                               cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy((void *) image_device, &(image.data[0]),
                               sizeof(float) * image_size,
                               cudaMemcpyHostToDevice));

    // determine thread layout

    // split 1024 threads into equal parts -> 10 * 10 * 10 = 1000
    int max_threads_per_block_cbrt = std::cbrt(
            max_threads_per_block);

    // determine threads per block: 10*10*10 or less (when image is smaller)
    dim3 num_threads_per_block(std::min(image.n_rows, max_threads_per_block_cbrt),
                               std::min(image.n_cols, max_threads_per_block_cbrt),
                               std::min(image.n_depts, max_threads_per_block_cbrt));

    // determine block layout: image size / threads per block
    dim3 num_blocks(image.n_rows / num_threads_per_block.x,
                    image.n_cols / num_threads_per_block.y,
                    image.n_depts / num_threads_per_block.z);

    // round up if num_block is zero or when there are less threads per axis than pixels -> 256 pixel -> 26 blocks instead of 25
    if (0 == num_blocks.x || num_blocks.x * num_threads_per_block.x < image.n_rows) {
        num_blocks.x++;
    }
    if (0 == num_blocks.y || num_blocks.y * num_threads_per_block.y < image.n_cols) {
        num_blocks.y++;
    }
    if (0 == num_blocks.z || num_blocks.z * num_threads_per_block.z < image.n_depts) {
        num_blocks.z++;
    }

    // print thread layout
    std::cout << "Thread-Layout:" << std::endl;
    std::cout << "Blocks per Axis (x,y,z) = " << num_blocks.x << " / " << num_blocks.y << " / " << num_blocks.z << std::endl;
    std::cout << "Threads per Block (x,y,z) = " << num_threads_per_block.x << " / "
              << num_threads_per_block.y << " / " << num_threads_per_block.z << std::endl;

    // run kernel

    t.start();

    // ENTWEDER SHARED ODER OHNE SHARED MEMORY
    convSeparableShared<kernel_supp_half><<< num_blocks, num_threads_per_block >>> (kernel_device, image_device, image_conv_device, image.n_rows, image.n_depts, image.it);
    //convSeparable<kernel_supp_half><<< num_blocks, num_threads_per_block >>>(kernel_device, image_device,image_conv_device, image.n_rows,image.n_depts, image.it);

    checkLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();
    std::cerr << "GPU Convolution: ";
    file << t.stop() << "\n";


    // copy result back to host
    checkErrorsCuda(cudaMemcpy(&image_conv.data[0], image_conv_device,
                               sizeof(float) * image_size,
                               cudaMemcpyDeviceToHost));

    // write result
    // delete content of output_gpu folder
    //system("exec rm -r ../images/output_gpu/*");
    t.start();
    image_conv.write("../images/output_gpu/head_gpu");
    std::cerr << "GPU Convolution: Writing: ";
    t.stop();

    // clean up device memory
    checkErrorsCuda(cudaFree(kernel_device));
    checkErrorsCuda(cudaFree(image_device));
    checkErrorsCuda(cudaFree(image_conv_device));
}

template <int kernel_supp>
void halfConv(float *kdata, Image &image, std::fstream &file) {

    if(image.it == PPM) {
        std::cerr << "PPM not implemented for halfs" << std::endl;
        exit (EXIT_FAILURE);
    }

    // set output image
    Image image_conv(image.n_rows, image.n_cols, image.n_depts, image.it);
    image_conv.data.resize(image.n_rows * image.n_cols * image.n_depts); // WEIRD BUG

    // set kernel layout
    const int kernel_supp_half = kernel_supp / 2;
    const int kernel_size = kernel_supp * kernel_supp * kernel_supp;

    // setup timer
    Timer t = Timer();

    // run on cpu
    t.start();
    convSeparableHost(kdata, kernel_supp_half, image, image_conv);
    std::cerr << "CPU Convolution: ";
    t.stop();
    //system("exec rm -r ../images/output_cpu/*");
    t.start();
    image_conv.write("../images/output_cpu/head_cpu");
    std::cerr << "CPU Convolution:  Writing: ";
    t.stop();
    // check execution environment
    int device_handle = 0;
    int max_threads_per_block = 0;
    if (!initDevice(device_handle, max_threads_per_block)) {
        exit(EXIT_FAILURE);
    }


    // initialize memory
    __half *kernel_device = nullptr;
    __half *image_device = nullptr;
    __half *image_conv_device = nullptr;

    // convert kernel and image to __half

    __half hkdata[kernel_size];

    for(int i = 0; i < kernel_size; i++) {
        hkdata[i] = __float2half(kdata[i]);
    }

    int image_size = image.data.size();
    __half* himage = (__half*) malloc(image_size*sizeof(__half));

    //std::cout << image.data.front() << std::endl;
    for(int i = 0; i < image.data.size(); i++) {
        himage[i] = __float2half(image.data[i]);
    }

    // allocate device memory
    checkErrorsCuda(cudaMalloc((void **) &kernel_device, sizeof(__half) * kernel_size));
    checkErrorsCuda(cudaMalloc((void **) &image_device, sizeof(__half) * image.n_cols * image.n_rows * image.n_depts));
    checkErrorsCuda(
            cudaMalloc((void **) &image_conv_device, sizeof(__half) * image.n_cols * image.n_rows * image.n_depts));


    // copy device memory
    checkErrorsCuda(cudaMemcpy((void *) kernel_device, hkdata,
                               sizeof(__half) * kernel_size,
                               cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy((void *) image_device, himage,
                               sizeof(__half) * image.n_cols * image.n_rows * image.n_depts,
                               cudaMemcpyHostToDevice));

    // determine thread layout

    // split 1024 threads into equal parts -> 10 * 10 * 10 = 1000
    int max_threads_per_block_cbrt = std::cbrt(
            max_threads_per_block);


    // determine threads per block: 10*10*10 or less (when image is smaller)
    dim3 num_threads_per_block(std::min(image.n_rows, max_threads_per_block_cbrt),
                               std::min(image.n_cols, max_threads_per_block_cbrt),
                               std::min(image.n_depts, max_threads_per_block_cbrt));

    // determine block layout: image size / threads per block
    dim3 num_blocks(image.n_rows / num_threads_per_block.x,
                    image.n_cols / num_threads_per_block.y,
                    image.n_depts / num_threads_per_block.z);

    // round up if num_block is zero or when there are less threads per axis than pixels -> 256 pixel -> 26 blocks instead of 25
    if (0 == num_blocks.x || num_blocks.x * num_threads_per_block.x < image.n_rows) {
        num_blocks.x++;
    }
    if (0 == num_blocks.y || num_blocks.y * num_threads_per_block.y < image.n_cols) {
        num_blocks.y++;
    }
    if (0 == num_blocks.z || num_blocks.z * num_threads_per_block.z < image.n_depts) {
        num_blocks.z++;
    }

    // print thread layout
    std::cout << "Thread-Layout:" << std::endl;
    std::cout << "Blocks per Axis (x,y,z) = " << num_blocks.x << " / " << num_blocks.y << " / " << num_blocks.z << std::endl;
    std::cout << "Threads per Block (x,y,z) = " << num_threads_per_block.x << " / "
              << num_threads_per_block.y << " / " << num_threads_per_block.z << std::endl;

    // run kernel

    t.start();

    // nur mit shared memory verfuegbar
    convSeparableHalf<kernel_supp_half><<< num_blocks, num_threads_per_block >>> (image.n_rows, image.n_depts, image.it,kernel_device, image_device, image_conv_device);

    checkLastCudaError("Kernel execution failed");
    cudaDeviceSynchronize();

    std::cerr << "GPU Convolution: ";
    file << t.stop() << "\n";

    // copy result back to host
    checkErrorsCuda(cudaMemcpy(himage, image_conv_device,
                               sizeof(__half) * image_size,
                               cudaMemcpyDeviceToHost));

    // change result back to float
    for(int i = 0; i < image_conv.data.size(); i++) {
        image_conv.data[i] = __half2float(himage[i]);
    }

    free(himage);

    // write result
    //system("exec rm -r ../images/output_gpu/*");
    t.start();
    image_conv.write("../images/output_gpu/head_gpu");
    std::cerr << "CPU Convolution:  Writing: ";
    t.stop();

    // clean up device memory
    checkErrorsCuda(cudaFree(kernel_device));
    checkErrorsCuda(cudaFree(image_device));
    checkErrorsCuda(cudaFree(image_conv_device));
}

int main() {
    const int kernel_supp = 5; // DAS HIER MUSS DEM KERNEL ENTSPRECHEN -> SIEHE KERNEL NAME
    const int kernel_size = kernel_supp * kernel_supp * kernel_supp;

    float kdata[kernel_size];

    //std::ifstream kernel("../kernels/identity3x3x3.txt");
    //std::ifstream kernel("../kernels/sharpen3x3x3.txt");
    //std::ifstream kernel("../kernels/3RDedgedetection3x3x3.txt");
    //std::ifstream kernel("../kernels/2NDedgedetection3x3x3.txt");
    //std::ifstream kernel("../kernels/edgedetection3x3x3.txt");
    std::ifstream kernel("../kernels/gaussian5x5x5.txt");
    //std::ifstream kernel("../kernels/boxblur3x3x3.txt");

    if (!kernel) {
        std::cout << "file not found\n";
        return 1;
    }

    std::string line;
    for (float &i : kdata) {
        std::getline(kernel, line);
        i = std::stof(line);
    }
    kernel.close();

    // saving time results
    std::fstream file;
    file.open(("../benchmark_results.txt"), std::fstream::app);

    //for(int i = 50; i <= 1000; i+=50) { // benchmarking

    Timer t = Timer();

    //file << i << " "; // benchmarking
    t.start();

    Image image("../images/head/head", 99, PGM);
    //Image image("../images/head2/head", 33, PGM);
    //Image image("../images/abdomen/image-00", 361, PGM);
    //Image image("../images/benchmark/test", i, PGM); // benchmarking
    //Image image("../images/colored_head/head", 6, PPM); // farbbild
    //Image image("../images/chicken/chicken", 25, PGM);

    std::cerr << "Reading: ";
    t.stop();

    // AUSFÜHREN entweder mit floats oder halfs
    floatConv<kernel_supp>(kdata, image, file);
    //halfConv<kernel_supp>(kdata, image, file);

    //} // benchmarking

    file.close();
    return 0;
}
