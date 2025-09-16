#ifndef FUNC_N
#define FUNC_N


#define CUDA_KERNAL_LOOP(i,n)\
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)

template<typename T>
__global__ void relu_gpu(T* in, T* out, int size){
    CUDA_KERNAL_LOOP(i, size){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

template<typename T>
__global__ void relu_gpu_backward(T* in_grad, T* out_grad, T* x, int size){
    CUDA_KERNAL_LOOP(i, size){
        in_grad[i] = x[i] > 0 ? out_grad[i] : 0;
    }
}

template<typename T>
__global__ void sigmoid_gpu(T* in, T* out, int size){
    CUDA_KERNAL_LOOP(i, size){
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

template<typename T>
__global__ void sigmoid_gpu_backward(T* in_grad, T* out_grad, T* x, int size){
    CUDA_KERNAL_LOOP(i, size){
        T sig = 1 / (1 + exp(-x[i]));
        in_grad[i] = sig * (1 - sig) * out_grad[i];
    }
}

#endif