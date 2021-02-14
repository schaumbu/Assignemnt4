#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H


// includes, system
#include <iostream>


template< typename T >
void checkError(T result, char const *const func, const char *const file, int const line) {

    if (result)  {
        std::cerr << "CUDA error at " << file << "::" << line << " with error code "
                  <<  static_cast<int>(result) << " for " << func << "()." << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkErrorsCuda(val) checkError( (val), #val, __FILE__, __LINE__ )


inline void
checkLastCudaErrorFunc(const char *errorMessage, const char *file, const int line){
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkLastCudaError(msg)  checkLastCudaErrorFunc(msg, __FILE__, __LINE__)

bool
initDevice(int &device_handle, int &max_threads_per_block) {

    int deviceCount = 0;
    checkErrorsCuda(cudaGetDeviceCount(&deviceCount));

    if (0 == deviceCount) {
        std::cerr << "initDevice() : No CUDA device found." << std::endl;
        return false;
    }

    // one could implement more complex logic here to find the fastest device
    if (deviceCount > 1) {
        std::cerr << "initDevice() : Multiple CUDA devices found. Using first one." << std::endl;
    }

    // set the device
    checkErrorsCuda(cudaSetDevice(device_handle));

    cudaDeviceProp device_props;
    checkErrorsCuda(cudaGetDeviceProperties(&device_props, device_handle));
    max_threads_per_block = device_props.maxThreadsPerBlock;

    return true;
}

#endif //_CUDA_UTIL_H
